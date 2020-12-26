from pathlib import Path
from typing import Any, Dict, List, Tuple

import ignite.contrib.handlers.tensorboard_logger as tbl
import torch
import torch.nn.functional as F
from ignite.engine import Events
from ignite.engine.engine import Engine
from ignite.metrics.accuracy import Accuracy
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics.loss import Loss
from ignite.metrics.metric import Metric
from texttable import Texttable
from torch.utils.data.dataloader import DataLoader

from gradcam import ExecuteGradCAM
from modules import T
from modules import functions as fns
from modules import torch_utils as tutils
from modules import utils
from modules.global_config import GlobalConfig


def train_step(
    engine: Engine,
    minibatch: tutils.MiniBatch,
    model: tutils.Model,
    subdivisions: int,
    gc: GlobalConfig,
    pbar: utils.MyProgressBar,
    use_amp: bool = False,
    *,
    save_img_fn: fns.save_img_fn_t = fns.dummy_save_mistaken_image,  # from functions.py
    non_blocking: bool = True,
):
    model.net.train()
    model.optimizer.zero_grad()
    total_loss = 0.0

    epoch = engine.state.epoch
    max_epochs = engine.state.max_epochs

    for x, y in utils.subdivide_batch(
        minibatch.batch, model.device, subdivisions, non_blocking=non_blocking
    ):
        with torch.cuda.amp.autocast(enabled=use_amp):  # type: ignore
            y_pred = model.net(x)
            loss = model.criterion(y_pred, y)
        model.scaler.scale(loss).backward()
        total_loss += float(loss)

        # save mistaken predicted image
        save_img_fn((x, y), y_pred, str(minibatch.path[0]))

        # TODO: now, output labels are under 10. in file, all output?
        # softmax
        pred_labels = [str(torch.argmax(x).item()) for x in F.softmax(y_pred[:10], dim=1)]
        pred_labels = " ".join(pred_labels)
        ans_labels = utils.tensor2np(y[:10])
        pbar.log_message(
            f"Epoch [{epoch}/{max_epochs}]: pred [{pred_labels}] | ans {ans_labels}, loss: {round(total_loss / subdivisions, 3)}",
            stdout=gc.option.is_show_batch_result,
        )
    model.scaler.step(model.optimizer)
    model.scaler.update()
    return total_loss / subdivisions


def validate_loader(batch: T._batch, model: tutils.Model, *, non_blocking: bool = True,) -> Any:
    model.net.eval()
    with torch.no_grad():
        x, y = utils.prepare_batch(batch, model.device, non_blocking=non_blocking)
        y_pred = model.net(x)
    return y_pred, y


def validate_model(
    engine: Engine,
    collect_list: List[Tuple[DataLoader, str]],
    gc: GlobalConfig,
    pbar: utils.MyProgressBar,
    classes: List[str],
    gcam: ExecuteGradCAM,
    model: tutils.Model,
    valid_schedule: List[bool],
    *,
    exec_gcam_fn: fns.exec_gcam_fn_t = fns.dummy_execute_gradcam,  # from functions.py
    exec_softmax_fn: fns.exec_softmax_fn_t = fns.dummy_execute_softmax,  # from functions.py
    save_cm_fn: fns.save_cm_fn_t = fns.dummy_save_cm_image,  # from functions.py
    non_blocking: bool = True,
    # tb_logger: Optional[tbl.TensorboardLogger],
) -> None:
    epoch = engine.state.epoch
    max_epochs = engine.state.max_epochs
    # iteration = engine.state.iteration

    gc.logfile.writeline(f"--- Epoch: {epoch}/{max_epochs} ---")
    gc.ratefile.write(f"{epoch}")

    # for evaluator, loader, phase in collect_list:
    for loader, phase in collect_list:
        # table
        table = Texttable()
        table.header(["No.", "Name", "Accuracy", "Detail"])
        table.set_deco(Texttable.HEADER)
        table.set_cols_align(["c", "l", "r", "l"])

        softmaxfile = utils.LogFile(
            gc.path.softmax.joinpath(f"epoch{epoch}_{phase.lower()}.csv"), stdout=False
        )

        classes_ = ",".join([f"[{cls}]" for cls in classes])
        softmaxfile.writeline(f"correct class,predict class,filename,,softmax,{classes_}")

        metrics: Dict[str, Metric] = {}

        # compute only validation.
        if valid_schedule[epoch - 1]:
            metrics = {
                "acc": Accuracy(device=model.device),
                "loss": Loss(model.criterion, device=model.device),
                "cm": ConfusionMatrix(len(classes), device=model.device),
            }

        for x, y, path in loader:
            y_pred, y = validate_loader((x, y), model)
            ans, pred = utils.get_label(y, y_pred)

            path = str(path[0])
            exec_softmax_fn(path, gc, y_pred, y, gcam.classes, softmaxfile)
            exec_gcam_fn(engine, gc, gcam, model, path, ans, pred, phase.lower())

            # update metrics
            for metric in metrics.values():
                metric.update(output=(y_pred, y))

        if valid_schedule[epoch - 1]:
            # conpute average
            avg_acc = metrics["acc"].compute()
            avg_loss = metrics["loss"].compute()
            pbar.log_message(
                f"\n{phase.capitalize()} Validation Results -  Avg accuracy: {avg_acc:.3f} Avg loss: {avg_loss:.3f}"
            )
            cm = utils.tensor2np(metrics["cm"].compute())

            # confusion matrix
            title = f"Confusion Matrix - {phase} (Epoch {epoch})"
            fig = utils.plot_confusion_matrix(cm, classes, title=title)

            save_cm_fn(gc.path.cm, phase, epoch, fig)
            gc.ratefile.write(f",,")

            for i, cls_name in enumerate(classes):
                n_all = sum(cm[i])
                n_acc = cm[i][i]
                acc = n_acc / n_all

                table.add_row([i, cls_name, acc, f"{n_acc} / {n_all} images"])
                gc.ratefile.write(f"{acc:<.3f},")
            gc.ratefile.write(f"{avg_acc},")
            table.add_row(["*", "[avg]", round(avg_acc, 3), "-"])

            pbar.log_message(f"\n{table.draw()}")

        softmaxfile.flush()
        softmaxfile.close()

    pbar.log_message("\n")
    pbar.n = pbar.last_print_n = 0  # type: ignore

    gc.ratefile.writeline()

    gc.logfile.flush()
    gc.ratefile.flush()


def save_model(model: tutils.Model, classes: List[str], gc: GlobalConfig, epoch: int):
    path = utils.create_filepath(Path(gc.path.model), f"epoch{epoch}", ext="pt")
    print(f"Saving model to '{path}'...")

    save_cfg = {"classes": classes, "model_state_dict": model.net.state_dict()}
    torch.save(save_cfg, path)


def attach_log_to_tensorboard(
    tb_logger: tbl.TensorboardLogger,
    trainer: Engine,
    _list: List[Tuple[Engine, DataLoader, str]],
    model: tutils.Model,
) -> None:
    # attach_num = 1

    # tb_logger.attach_output_handler(
    #     trainer,
    #     event_name=Events.ITERATION_COMPLETED(every=attach_num),
    #     tag="training",  # type: ignore
    #     output_transform=lambda loss: {"batchloss": loss},  # type: ignore
    #     metric_names="all",  # type: ignore
    # )

    for evaluator, _, tag in _list:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,  # type: ignore
            metric_names=["loss", "acc"],  # type: ignore
            global_step_transform=tbl.global_step_from_engine(trainer),  # type: ignore
        )

    # tb_logger.attach_opt_params_handler(
    #     trainer, event_name=Events.ITERATION_COMPLETED(every=attach_num), optimizer=model.optimizer  # type: ignore
    # )

    # tb_logger.attach(
    #     trainer,
    #     log_handler=tbl.WeightsScalarHandler(model.net),
    #     event_name=Events.ITERATION_COMPLETED(every=attach_num),
    # )

    # tb_logger.attach(
    #     trainer,
    #     log_handler=tbl.WeightsHistHandler(model.net),
    #     event_name=Events.EPOCH_COMPLETED(every=attach_num),
    # )

    # tb_logger.attach(
    #     trainer,
    #     log_handler=tbl.GradsScalarHandler(model.net),
    #     event_name=Events.ITERATION_COMPLETED(every=attach_num),
    # )

    # tb_logger.attach(
    #     trainer,
    #     log_handler=tbl.GradsHistHandler(model.net),
    #     event_name=Events.EPOCH_COMPLETED(every=attach_num),
    # )


def show_network_difinition(
    gc: GlobalConfig, model: tutils.Model, dataset: tutils.CreateDataset, stdout: bool = False
) -> None:
    r"""Show network difinition on console.

    Args:
        model (tu.Model): model.
        dataset (tu.CreateDataset): dataset.
    """

    def show_config(dict_: Dict[str, Any], header: str = "") -> None:
        r"""execute.

        Args:
            dict_ (Dict[str, Any]): show contents.
            header (str, optional): show before showing contents. Defaults to ''.
        """
        gc.logfile.writeline(header, stdout=stdout)
        max_len = max([len(x) for x in dict_.keys()])  # adjust to max length of key

        for k, v in dict_.items():
            # format for structure of network
            if isinstance(v, str) and v.find("\n") > -1:
                v = v.replace("\n", "\n" + " " * (max_len + 3)).rstrip()

            gc.logfile.writeline(f"{k.center(max_len)} : {v}", stdout=stdout)
        gc.logfile.writeline("", stdout=stdout)

    base_conf = {
        "run time": gc.filename_base,
        "dataset path": (
            f"train - {gc.path.train_dir}\nvalid - {gc.path.valid_dir}"
            if gc.path.is_pre_splited
            else str(gc.path.dataset)
        ),
        "result path": gc.path.result_dir,
    }
    dataset_conf = {
        "limit dataset size": gc.dataset.limit_size,
        "shuffle dataset per epoch is": gc.dataset.is_shuffle_per_epoch,
        "supported extensions": gc.dataset.extensions,
    }
    gradcam_conf = {
        "Grad-CAM is": gc.gradcam.enabled,
        "Grad-CAM execute only mistaken": gc.gradcam.only_mistaken,
        "Grad-CAM layer": gc.gradcam.layer,
        "Grad-CAM cycle": gc.gradcam.cycle,
    }
    network_conf = {
        "input size": f"(h: {gc.network.height}, w: {gc.network.width})",
        "channels": gc.network.channels,
        "epoch": gc.network.epoch,
        "batch size": gc.network.batch,
        "subdivisions": gc.network.subdivisions,
        "cycle of saving": gc.network.save_cycle,
        "cycle of validation": gc.network.valid_cycle,
        "GPU available": torch.cuda.is_available(),  # type: ignore
        "GPU used": gc.network.gpu_enabled,
        "saving final pth is": gc.network.is_save_final_model,
        "amp enabled": gc.network.amp,
    }
    option_conf = {
        "saving mistaken prediction is": gc.option.is_save_mistaken_pred,
        "saving log is": gc.option.is_save_log,
        "saving config is": gc.option.is_save_config,
        "logging TensorBoard is": gc.option.log_tensorboard,
        "saving Confusion-Matrix is": gc.option.is_save_cm,
    }
    model_conf = {
        "net": str(model.net),
        "optimizer": str(model.optimizer),
        "criterion": str(model.criterion),
        "train dataset size": dataset.train_size,
        "unknown dataset size": dataset.unknown_size,
        "known dataset size": dataset.known_size,
    }
    classes = {str(k): v for k, v in dataset.classes.items()}

    collect_list = [
        (classes, "--- Classes ---"),
        (base_conf, "--- Base ---"),
        (dataset_conf, "--- Dataset ---"),
        (gradcam_conf, "--- Grad-CAM ---"),
        (network_conf, "--- Network ---"),
        (option_conf, "--- Option ---"),
        (model_conf, "--- Model ---"),
    ]
    print()
    for dict_, title in collect_list:
        show_config(dict_, title)

    gc.logfile.flush()
