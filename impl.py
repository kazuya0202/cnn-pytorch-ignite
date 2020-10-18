from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ignite.contrib.handlers.tensorboard_logger as tbl
import numpy as np
import torch
import tqdm
from ignite.engine import Events
from ignite.engine.engine import Engine
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image

from gradcam import ExecuteGradCAM
from modules import T
from modules import torch_utils as tutils
from modules import utils
from modules.global_config import GlobalConfig


@dataclass
class Model:
    net: T._net_t
    optimizer: T._optim_t
    criterion: T._criterion_t
    device: torch.device


def train_step(
    minibatch: tutils.MiniBatch,
    model: Model,
    subdivisions: int,
    *,
    is_save_mistaken_pred: bool = False,
    non_blocking: bool = True,
):
    model.net.train()
    model.optimizer.zero_grad()

    total_loss = 0.0

    for (x, y), iter_size in utils.subdivide_batch(
        minibatch.batch, model.device, subdivisions, non_blocking=non_blocking
    ):
        y_pred = model.net(x)
        loss = model.criterion(y_pred, y) / iter_size
        loss.backward()
        total_loss += loss.item()

        # save mistaken predicted image
        if not is_save_mistaken_pred:
            continue

        ans, pred = utils.get_label(y, y_pred)
        if ans != pred:
            save_image(x, str(minibatch.path))

    model.optimizer.step()
    return total_loss / subdivisions


def validation_step(
    engine: Engine,
    minibatch: tutils.MiniBatch,
    model: Model,
    gc: GlobalConfig,
    gcam: ExecuteGradCAM,
    phase: str = "known",
    *,
    non_blocking: bool = True,
) -> T._batch_t:
    model.net.eval()

    with torch.no_grad():
        x, y = utils.prepare_batch(minibatch.batch, model.device, non_blocking=non_blocking)
        y_pred = model.net(x)

        ans, pred = utils.get_label(y, y_pred)
        execute_gradcam(engine, gc, gcam, model, str(minibatch.path[0]), ans, pred, phase)

        return y_pred, y


def validate_model(
    engine: Engine,
    collect_list: List[Tuple[Engine, DataLoader, str]],
    gc: GlobalConfig,
    pbar: utils.MyProgressBar,
    classes: List[str],
    tb_logger: Optional[tbl.TensorboardLogger],
) -> None:
    epoch_num = engine.state.epoch
    gc.logfile.writeline(f"--- Epoch: {epoch_num}/{engine.state.max_epochs} ---")
    gc.ratefile.write(f"{epoch_num}")

    for evaluator, loader, phase in collect_list:
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        avg_acc = metrics["acc"]
        avg_loss = metrics["loss"]

        pbar.log_message("")
        pbar.log_message(f"{phase} Results -  Avg accuracy: {avg_acc:.3f} Avg loss: {avg_loss:.3f}")

        cm = metrics["cm"]
        phase_name = phase.split(" ")[0].lower()

        # confusion matrix
        title = f"Confusion Matrix - {phase} (Epoch {epoch_num})"
        fig = utils.plot_confusion_matrix(cm, classes, title=title)

        if gc.option.is_save_cm:
            p = gc.path.cm.joinpath(phase_name, f"epoch{epoch_num}_{phase_name}.jpg")
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(p))  # type: ignore

        if tb_logger:
            title = "Confusion Matrix " + phase_name.capitalize()
            utils.add_image_to_tensorboard(tb_logger, fig, title, epoch_num)

        gc.ratefile.write(f",,")

        align_size = max([len(v) for v in classes]) + 2  # "[class_name]"
        cm = cm.cpu().numpy()
        for i, cls_name in enumerate(classes):
            n_all = sum(cm[i])
            n_acc = cm[i][i]
            acc = n_acc / n_all

            cls_name = f"[{cls_name}]".ljust(align_size)
            s = f" {cls_name} -> acc: {acc:<.3f} ({n_acc} / {n_all} images.)"
            pbar.log_message(s)

            gc.ratefile.write(f"{acc:<.3f},")

        gc.ratefile.write(f"{avg_acc},")

    pbar.log_message("\n")
    pbar.n = pbar.last_print_n = 0  # type: ignore

    gc.logfile.flush()
    gc.ratefile.writeline()
    gc.ratefile.flush()


def execute_gradcam(
    engine: Engine,
    gc: GlobalConfig,
    gcam: ExecuteGradCAM,
    model: Model,
    path: T._path_t,
    ans: int,
    pred: int,
    name: str,
) -> None:
    # do not execute / execute only mistaken
    is_correct = ans == pred
    if any([(not gc.gradcam.enabled), (gc.gradcam.only_mistaken and is_correct)]):
        return
    epoch = engine.state.epoch
    iteration = engine.state.iteration

    gcam_base_dir = Path(gc.path.gradcam)
    epoch_str = f"epoch{epoch}"

    dir_name = "correct" if is_correct else "mistaken"
    base_dir = utils.concat_path(
        gcam_base_dir, concat=[f"{name}_{dir_name}", epoch_str], is_make=True
    )

    ret = gcam.main(model.net, str(path))
    pbar = tqdm.tqdm(ret.items(), desc="Grad-CAM", leave=False)

    for name, data_list in pbar:  # name: "gcam", "gbp" ...
        for i, img in enumerate(data_list):
            is_png = name == "gbp"
            ext = "png" if is_png else "jpg"

            s = f"{iteration}_{gcam.classes[i]}_{name}_pred[{pred}]_correct[{ans}].{ext}"
            path_ = base_dir.joinpath(s)

            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("RGBA" if is_png else "RGB")
            img.save(str(path_))
            # print(path_)
    del ret


def save_model(model: Model, classes: List[str], gc: GlobalConfig, epoch: int):
    path = utils.create_filepath(Path(gc.path.model), f"epoch{epoch}", ext="pt")
    print(f"Saving model to '{path}'...")

    save_cfg = {"classes": classes, "model_state_dict": model.net.state_dict()}
    torch.save(save_cfg, path)


def log_tensorboard(
    tb_logger: tbl.TensorboardLogger,
    trainer: Engine,
    _list: List[Tuple[Engine, DataLoader, str]],
    model: Model,
) -> None:
    attach_num = 1

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=attach_num),
        tag="training",  # type: ignore
        output_transform=lambda loss: {"batchloss": loss},  # type: ignore
        metric_names="all",  # type: ignore
    )

    for evaluator, _, tag in _list:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,  # type: ignore
            metric_names=["loss", "acc"],  # type: ignore
            global_step_transform=tbl.global_step_from_engine(trainer),  # type: ignore
        )

    tb_logger.attach_opt_params_handler(
        trainer, event_name=Events.ITERATION_COMPLETED(every=attach_num), optimizer=model.optimizer  # type: ignore
    )

    tb_logger.attach(
        trainer,
        log_handler=tbl.WeightsScalarHandler(model.net),
        event_name=Events.ITERATION_COMPLETED(every=attach_num),
    )

    tb_logger.attach(
        trainer,
        log_handler=tbl.WeightsHistHandler(model.net),
        event_name=Events.EPOCH_COMPLETED(every=attach_num),
    )

    tb_logger.attach(
        trainer,
        log_handler=tbl.GradsScalarHandler(model.net),
        event_name=Events.ITERATION_COMPLETED(every=attach_num),
    )

    tb_logger.attach(
        trainer,
        log_handler=tbl.GradsHistHandler(model.net),
        event_name=Events.EPOCH_COMPLETED(every=attach_num),
    )


def show_network_difinition(
    gc: GlobalConfig, model: Model, dataset: tutils.CreateDataset, stdout: bool = False
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
            f"train - {gc.dataset.train_dir}\nvalid - {gc.dataset.valid_dir}"
            if gc.dataset.is_pre_splited
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
