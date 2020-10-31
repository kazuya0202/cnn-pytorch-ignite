import itertools
import os
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine.engine import Engine
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from tqdm import tqdm

from . import T


@dataclass
class LogFile:
    path: Optional[T._path] = None
    stdout: bool = False
    clear: bool = False

    _is_write: bool = field(init=False)
    _path: Path = field(init=False)
    _file: TextIOWrapper = field(init=False)

    def __post_init__(self):
        if self.path is None:
            self._is_write = False
            return

        self._path = Path(self.path)
        self._is_write = True

        if self.clear:
            self.__clear()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

        # open file as append mode.
        self._file = self._path.open("a")

    def write(self, line: object = "", stdout: Optional[bool] = None) -> None:
        self.__write(f"{line}", f"\r{line}", stdout=self.__is_stdout(stdout))

    def writeline(self, line: object = "", stdout: Optional[bool] = None) -> None:
        self.__write(content=f"{line}\n", line=f"{line}\n", stdout=self.__is_stdout(stdout))

    def __write(self, content: str, line: object, stdout: bool) -> None:
        """
        Args:
            content (str): write `content` to file.
            line (object): output `line` as stdout.
        """
        if self._is_write:
            self._file.write(content)

        if stdout:
            print(line, end="")

    def flush(self) -> None:
        if self._is_write:
            self._file.flush()

    def __clear(self) -> None:
        if not self._is_write:
            return

        if self._path.exists():
            self._path.unlink()  # delete
        self._path.touch()  # create

    def __is_stdout(self, stdout: Optional[bool]) -> bool:
        # priority `stdout` argument.
        return stdout if stdout is not None else self.stdout

    def close(self) -> None:
        if self._is_write:
            self._file.close()

    def __enter__(self) -> "LogFile":
        return self

    def __exit__(self, t, v, tb) -> None:
        self.close()


class MyProgressBar(ProgressBar):
    def __init__(
        self,
        logfile: LogFile = LogFile(),
        persist=False,
        bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]",
        **tqdm_kwargs,
    ) -> None:
        super().__init__(persist, bar_format, **tqdm_kwargs)
        self.logfile = logfile

    def log_message(self, message: str = "", *, stdout: bool = True):
        """
        Logs a message, preserving the progress bar correct output format.

        Args:
            message (str): string you wish to log.
            stdout (bool): output verbose stdout.
        """
        self.logfile.writeline(message, stdout=False)
        if not stdout:
            return

        # tqdm.write(message, file=self.tqdm_kwargs.get("file", None))
        tqdm.write(message)


def prepare_batch(
    batch: T._batch, device: torch.device, *, non_blocking: bool = False,
) -> T._batch:
    x, y = batch
    return (
        x.to(device=device, non_blocking=non_blocking),
        y.to(device=device, non_blocking=non_blocking),
    )


def subdivide_batch(
    batch: T._batch,
    device: torch.device,
    subdivisions: int,
    *,
    non_blocking: bool = False,
    # ) -> Iterator[Tuple[T._batch, int]]:
) -> Iterator[T._batch]:
    x, y = batch
    batch_len = x.size()[0]
    sep = np.linspace(start=0, stop=batch_len, num=subdivisions + 1, dtype=np.int)

    for n, m in zip(sep[:-1], sep[1:]):
        batch = (x[n:m], y[n:m])
        if batch[0].size()[0] == 0:
            continue
        # yield prepare_batch(batch, device, non_blocking=non_blocking), m - n
        yield prepare_batch(batch, device, non_blocking=non_blocking)


def attach_metrics(evaluator: Engine, metrics: Dict[str, Any]) -> None:
    for name, metric in metrics.items():
        metric.attach(evaluator, name)


def create_schedule(max_epoch: int, cycle: int) -> List[bool]:
    if cycle == 0:
        return [*[False] * (max_epoch - 1), True]  # [-1] is only True.

    # range(N - 1) -> last epoch is True.
    _ = [(i + 1) % cycle == 0 for i in range(max_epoch - 1)]
    return [*_, True]


def create_filepath(dir_: T._path, name: str, is_prefix_seq: bool = False, ext: str = "txt") -> str:
    if isinstance(dir_, str):
        dir_ = Path(dir_)
    prefix = "" if not is_prefix_seq else f"{len([x for x in dir_.glob(f'*.{ext}')])}_"
    path = dir_.joinpath(f"{prefix}{name}.{ext}")
    return str(path)


def concat_path(
    base_path: T._path, concat: Union[T._path, List[T._path]], *, is_make: bool = False
) -> Path:
    fp = Path(base_path, *concat) if isinstance(concat, list) else Path(base_path, concat)
    if is_make:
        fp.mkdir(parents=True, exist_ok=True)
    return fp


def check_existence(path: Union[T._path, List[T._path]]) -> None:
    def _inner(path: T._path):
        if not is_exists(path):
            raise FileNotFoundError(f"'{path}' does not exist.")

    if isinstance(path, list):
        for p in path:
            _inner(p)
    else:
        _inner(path)


def is_exists(path: T._path) -> bool:
    return os.path.exists(path)


def replace_backslash(s: T._path) -> Path:
    return Path(str(s).replace("\\\\", "\\"))


def add_image_to_tensorboard(
    tb_logger: TensorboardLogger, fig: plt.Figure, title: str, step: int = 0
) -> None:
    r"""Add plot image to TensorBoard.

    Args:
        writer (tbx.SummaryWriter): tensorboard writer.
        fig (plt.Figure): plot figure.
        title (str): title of plotted figure.
        step (int, optional): step. Defaults to 0.
    """
    fig.canvas.draw()  # type: ignore
    img = fig.canvas.renderer._renderer  # type: ignore
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
    plt.rcParams["font.size"] = 18  # type: ignore

    fig, axes = plt.subplots(figsize=(10, 10))

    # ticklabels
    tick_marks = np.arange(len(classes))

    plt.setp(axes, xticks=tick_marks, xticklabels=classes)  # type: ignore
    plt.setp(axes, yticks=tick_marks, yticklabels=classes)  # type: ignore
    # rotate xticklabels
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")  # type: ignore

    # title
    plt.suptitle(title)

    # label
    axes.set_ylabel("True label")  # type: ignore
    axes.set_xlabel("Predicted label")  # type: ignore

    # grid
    # axes.grid(which='minor', color='b', linestyle='-', linewidth=3)

    img = plt.imshow(cm, interpolation="nearest", cmap=cmap)  # type: ignore

    # adjust color bar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(img, cax=cax)  # type: ignore

    thresh = cm.max() / 2.0
    fmt = ".2f" if normalize else "d"

    # plot text
    for i, j in itertools.product(range(len(classes)), range(len(classes))):
        clr = "white" if cm[i, j] > thresh else "black"
        axes.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color=clr)  # type: ignore

    plt.tight_layout()
    fig = plt.gcf()
    return fig


def get_label(ans_label: Tensor, pred_label: Tensor) -> Tuple[int, int]:
    ans = int(ans_label[0].item())
    pred = int(torch.max(pred_label.data, 1)[1].item())  # type: ignore
    return ans, pred
