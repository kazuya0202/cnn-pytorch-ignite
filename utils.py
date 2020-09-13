import itertools
from typing import Any, Dict, Iterator, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
# import torchvision.utils as vutils
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine.engine import Engine
from mpl_toolkits.axes_grid1 import make_axes_locatable

from my_typings import T


class MyProgressBar(ProgressBar):
    def __init__(
        self,
        persist=False,
        bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]",
        **tqdm_kwargs
    ) -> None:
        super().__init__(persist, bar_format, **tqdm_kwargs)

    def log_message(self, message: str = "", stdout_verbose: bool = True):
        """
        Logs a message, preserving the progress bar correct output format.

        Args:
            message (str): string you wish to log.
            verbose (bool): output verbose stdout.
        """
        if not stdout_verbose:
            # 必要ならファイルにだけ書き出す
            return

        from tqdm import tqdm

        tqdm.write(message, file=self.tqdm_kwargs.get("file", None))


def prepare_batch(batch: T._BATCH, device: torch.device, non_blocking: bool = False,) -> T._BATCH:
    x, y = batch
    return (
        x.to(device=device, non_blocking=non_blocking),
        y.to(device=device, non_blocking=non_blocking),
    )


def subdivide_batch(
    batch: T._BATCH, device: torch.device, subdivision: int, non_blocking: bool = False,
) -> Iterator[T._BATCH]:
    x, y = batch
    sep = np.linspace(start=0, stop=len(x), num=subdivision + 1, dtype=np.int)

    for n, m in zip(sep[:-1], sep[1:]):
        batch = (x[n:m], y[n:m])
        yield prepare_batch(batch, device, non_blocking=non_blocking)


def attach_metrics(evaluator: Engine, metrics: Dict[str, Any]) -> None:
    for name, metric in metrics.items():
        metric.attach(evaluator, name)


# def log_generated_images(tag, fake_buffer):
#     def wrapper(engine, logger, event_name):
#         res = vutils.make_grid(torch.cat(fake_buffer, dim=0), padding=2, normalize=True)
#         res = res.detach().cpu()
#         state = engine.state
#         global_step = state.get_event_attrib_value(event_name)
#         logger.writer.add_image(tag=tag, img_tensor=res, global_step=global_step, dataformats="CHW")

#     return wrapper


def add_to_tensorboard(
    tb_logger: TensorboardLogger, fig: plt.Figure, title: str, step: int = 0
) -> None:
    r"""Add plot image to TensorBoard.

    Args:
        writer (tbx.SummaryWriter): tensorboard writer.
        fig (plt.Figure): plot figure.
        title (str): title of plotted figure.
        step (int, optional): step. Defaults to 0.
    """
    fig.canvas.draw()
    img = fig.canvas.renderer._renderer
    img_ar = np.array(img).transpose(2, 0, 1)

    tb_logger.writer.add_image(title, img_ar, step)
    plt.close()  # clear plot


def plot_confusion_matrix(
    cm: Union[torch.Tensor, np.ndarray],
    classes: List[str],
    normalize: bool = False,
    title: str = "Confusion matrix",
    cmap: plt.cm = plt.cm.Greens,  # type: ignore
) -> plt.Figure:
    r"""Plot confusion matrix.

    Args:
        cm (Union[Tensor, np.ndarray]): array of confusion matrix.
        classes (List[str]): class list.
        normalize (bool, optional): normalize. Defaults to False.
        title (str, optional): plot title. Defaults to 'Confusion matrix'.
        cmap (plt.cm, optional): using color map. Defaults to plt.cm.Greens.

    Returns:
        plt.Figure: plotted figure.
    """
    # Tensor to np.ndarray
    _cm: np.ndarray = cm if not isinstance(cm, torch.Tensor) else cm.cpu().numpy()

    if normalize:
        _cm = _cm.astype("float") / _cm.sum(axis=1)[:, np.newaxis]

    # change font size
    plt.rcParams["font.size"] = 18

    fig, axes = plt.subplots(figsize=(10, 10))

    # ticklabels
    tick_marks = np.arange(len(classes))

    plt.setp(axes, xticks=tick_marks, xticklabels=classes)
    plt.setp(axes, yticks=tick_marks, yticklabels=classes)
    # rotate xticklabels
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # title
    plt.suptitle(title)

    # label
    axes.set_ylabel("True label")
    axes.set_xlabel("Predicted label")

    # grid
    # axes.grid(which='minor', color='b', linestyle='-', linewidth=3)

    img = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    # adjust color bar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(img, cax=cax)

    thresh = cm.max() / 2.0
    fmt = ".2f" if normalize else "d"

    # plot text
    for i, j in itertools.product(range(len(classes)), range(len(classes))):
        clr = "white" if cm[i, j] > thresh else "black"
        axes.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color=clr)

    plt.tight_layout()
    fig = plt.gcf()
    return fig
