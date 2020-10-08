from dataclasses import dataclass
from gcam import ExecuteGradCAM
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ignite.contrib.handlers.tensorboard_logger as tbl
import torch
from ignite.engine import Events
from ignite.engine.engine import Engine
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import numpy as np

import torch_utils as tutils
import utils
from gradcam import ExecuteOnlyGradCAM
from my_typings import T
from utils import prepare_batch
from yaml_parser import GlobalConfig


@dataclass
class Model:
    net: T._net_t
    optimizer: T._optim_t
    criterion: T._criterion_t
    device: torch.device


@dataclass
class Predict:
    pass


def train_step(
    minibatch: tutils.MiniBatch,
    model: Model,
    device: torch.device,
    subdivisions: int,
    non_blocking: bool = False,
):
    model.net.train()
    model.optimizer.zero_grad()

    pred_chunk = torch.tensor([], device=torch.device("cpu"))

    # total_loss = 0.0  # total loss of one batch
    for x, _ in utils.subdivide_batch(
        minibatch.batch, device, subdivisions=subdivisions + 1, non_blocking=non_blocking
    ):
        y_pred = model.net(x)
        pred_chunk = torch.cat((pred_chunk, y_pred.cpu()))

        # loss = criterion(y_pred, y)
        # loss.backward()
        # total_loss += float(loss)
    loss = model.criterion(pred_chunk.to(device), minibatch.batch[1].to(device))
    loss.backward()
    ret = loss.item()
    del loss

    model.optimizer.step()
    return ret
    # return total_loss / subdivisions  # avg loss in batch.


def validation_step(
    engine: Engine,
    minibatch: tutils.MiniBatch,
    model: Model,
    device: torch.device,
    gc: GlobalConfig,
    # gcam: ExecuteOnlyGradCAM,
    gcam: ExecuteGradCAM,
    name: str = "known",
    non_blocking: bool = False,
) -> T._batch_t:
    model.net.eval()
    with torch.no_grad():
        x, y = prepare_batch(minibatch.batch, device, non_blocking=non_blocking)
        y_pred = model.net(x)

        ans = int(y[0].item())
        pred = int(torch.max(y_pred.data, 1)[1].item())
        execute_gradcam(engine, gc, gcam, model, str(minibatch.path[0]), ans, pred, name)

        return y_pred, y


def validate_model(
    engine: Engine,
    _list: List[Tuple[Engine, DataLoader, str]],
    pbar: utils.MyProgressBar,
    classes: List[str],
    tb_logger: Optional[tbl.TensorboardLogger],
    verbose: bool = True,
) -> None:
    pbar.logfile.writeline(f"--- Epoch: {engine.state.epoch}/{engine.state.max_epochs} ---")

    for evaluator, loader, phase in _list:
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        avg_acc = metrics["acc"]
        avg_loss = metrics["loss"]

        pbar.log_message("", stdout=verbose)
        pbar.log_message(f"{phase} Results -  Avg accuracy: {avg_acc:.3f} Avg loss: {avg_loss:.3f}")

        cm = metrics["cm"]

        # confusion matrix
        if tb_logger:
            title = f"Confusion Matrix - {phase} (Epoch {engine.state.epoch})"
            fig = utils.plot_confusion_matrix(cm, classes, title=title)

            title = "Confusion Matrix " + phase.split(" ")[0]
            utils.add_to_tensorboard(tb_logger, fig, title, engine.state.epoch)

        if verbose:
            align_size = max([len(v) for v in classes]) + 2  # "[class_name]"
            cm = cm.cpu().numpy()
            for i, cls_name in enumerate(classes):
                n_all = sum(cm[i])
                n_acc = cm[i][i]
                acc = n_acc / n_all

                cls_name = f"[{cls_name}]".ljust(align_size)
                s = f" {cls_name} -> acc: {acc:<.3f} ({n_acc} / {n_all} images.)"
                pbar.log_message(s)

    pbar.log_message("")
    pbar.log_message("", stdout=verbose)
    pbar.n = pbar.last_print_n = 0  # type: ignore


def execute_gradcam(
    engine: Engine,
    gc: GlobalConfig,
    # gcam: ExecuteOnlyGradCAM,
    gcam: ExecuteGradCAM,
    model: Model,
    path: T._path_t,
    ans: int,
    pred: int,
    phase: str,
) -> None:
    # do not execute / execute only mistaken
    if any([(not gc.gradcam.enabled), (gc.gradcam.only_mistaken and ans == pred)]):
        return
    epoch = engine.state.epoch
    iteration = engine.state.iteration

    gcam_base_dir = Path(gc.path.gradcam)
    epoch_str = f"epoch{epoch}"

    mistaken_dir = utils.concat_path_and_mkdir(
        gcam_base_dir, concat=[f"{phase}_mistaken", epoch_str], is_make=True
    )
    correct_dir = None
    if not gc.gradcam.only_mistaken:
        correct_dir = utils.concat_path_and_mkdir(
            gcam_base_dir, concat=[f"{phase}_correct", epoch_str], is_make=True
        )

    ret = gcam.main(model.net, str(path))
    for phase, dat_list in ret.items():  # phase: "gcam", "gbp" ...
        for i, img_dat in enumerate(dat_list):
            is_png = phase == "gbp"
            ext = "png" if is_png else "jpg"

            s = f"{iteration}_{gcam.classes[i]}_{phase}_pred[{pred}]_correct[{ans}].{ext}"
            path_ = correct_dir.joinpath(s) if ans == pred else mistaken_dir.joinpath(s)

            if isinstance(img_dat, np.ndarray):
                img = Image.fromarray(img_dat)
            img = img_dat.convert("RGBA" if is_png else "RGB")
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

    global_conf = {
        "run time": gc.filename_base,
        "image path": gc.path.dataset,
        "supported extensions": gc.dataset.extensions,
        "saving debug log is": gc.option.is_save_log,
        # "saving rate log is": gc.option.is_save_rate_log,
        "pt save cycle": gc.network.save_cycle,
        "valid cycle": gc.network.valid_cycle,
        "saving final pth is": gc.network.is_save_final_model,
        "Grad-CAM is": gc.gradcam.enabled,
        "Grad-CAM layer": gc.gradcam.layer,
        # "load model is ": gc.option.load_model_path,
        # "load path": gc.option.load_model_path if gc.option.re_training else "None",
    }

    dataset_conf = {
        "limit dataset size": gc.dataset.limit_size,
        "train dataset size": dataset.train_size,
        "unknown dataset size": dataset.unknown_size,
        "known dataset size": dataset.known_size,
    }

    model_conf = {
        "net": str(model.net),
        "optimizer": str(model.optimizer),
        "criterion": str(model.criterion),
        "input size": f"(h: {gc.network.height}, w: {gc.network.width})",
        "epoch": gc.network.epoch,
        "batch size": gc.network.batch,
        "subdivisions": gc.network.subdivisions,
        "GPU available": torch.cuda.is_available(),
        "GPU used": gc.network.gpu_enabled,
        # "re-training": ("available" if gc.option.is_available_re_training else "not available"),
    }

    def _inner_execute(_dict: Dict[str, Any], header: str = "") -> None:
        r"""execute.

        Args:
            _dict (Dict[str, Any]): show contents.
            head (str, optional): show before showing contents. Defaults to ''.
        """
        gc.log.writeline(header, stdout=stdout)

        # adjust to max length of key
        max_len = max([len(x) for x in _dict.keys()])
        # _format = f'%-{max_len}s : %s'

        for k, v in _dict.items():
            # format for structure of network
            if isinstance(v, str) and v.find("\n") > -1:
                v = v.replace("\n", "\n" + " " * (max_len + 3)).rstrip()

            gc.log.writeline(f"{k.center(max_len)} : {v}", stdout=stdout)
        gc.log.writeline("", stdout=stdout)

    classes = {str(k): v for k, v in dataset.classes.items()}

    print()
    _inner_execute(classes, "--- Classes ---")
    _inner_execute(global_conf, "--- Global Configuration ---")
    _inner_execute(dataset_conf, "--- Dataset Configuration ---")
    _inner_execute(model_conf, "--- Model Configuration ---")
