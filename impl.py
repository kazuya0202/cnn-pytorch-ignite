from datetime import datetime
from typing import List, Tuple
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine.engine import Engine
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data.dataloader import DataLoader

import cnn
import utils
from my_typings import Names, T
from utils import prepare_batch


def train_step(
    batch: T._BATCH,
    model: cnn.Net,
    optimizer: optim.Adam,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    subdivision: int,
    non_blocking: bool = False,
):
    # https://pytorch.org/ignite/quickstart.html#explanation

    model.train()
    optimizer.zero_grad()

    total_loss = 0.0  # total loss of one batch
    for x, y in utils.subdivide_batch(
        batch, device, subdivision=subdivision + 1, non_blocking=non_blocking
    ):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss / subdivision  # avg loss in batch.


def validation_step(
    batch: T._BATCH, model: cnn.Net, device: torch.device, non_blocking: bool = False,
) -> Tuple[Tensor, Tensor]:
    model.eval()
    with torch.no_grad():
        x, y = prepare_batch(batch, device, non_blocking=non_blocking)
        y_pred = model(x)
        return y_pred, y


def log_results(
    engine: Engine,
    _list: List[Tuple[Engine, DataLoader, str]],
    pbar: utils.MyProgressBar,
    classes: List[str],
    tb_logger: TensorboardLogger,
    verbose: bool = True,
) -> None:
    for evaluator, loader, name in _list:
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        avg_acc = metrics["acc"]
        avg_loss = metrics["loss"]

        pbar.log_message("", verbose)
        pbar.log_message(f"{name} Results -  Avg accuracy: {avg_acc:.3f} Avg loss: {avg_loss:.3f}")

        if name == Names.TRAINING:
            continue

        cm = metrics.pop("cm", None)
        if cm is None:
            continue

        # confusion matrix
        title = f"Confusion Matrix - {name} (Epoch {engine.state.epoch})"
        fig = utils.plot_confusion_matrix(cm, classes, title=title)

        title = "Confusion Matrix " + name.split(" ")[0]
        utils.add_to_tensorboard(tb_logger, fig, title, engine.state.epoch)

        if verbose:
            cm = cm.cpu().numpy()
            for i, cls_name in enumerate(classes):
                n_all = sum(cm[i])
                n_acc = cm[i][i]
                acc = n_acc / n_all

                s = "  %-12s -> " % f"[{cls_name}]"
                s += f"acc: {acc:<.3f} ({n_acc} / {n_all} images.)"
                pbar.log_message(s)

    pbar.log_message("")
    pbar.log_message("", verbose)
    pbar.n = pbar.last_print_n = 0  # type: ignore


def save_checkpoint():
    pass
