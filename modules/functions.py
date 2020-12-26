from pathlib import Path
from typing import Callable, List

import torch
import torch.nn.functional as F
from gradcam import ExecuteGradCAM, GradCamType
from ignite.engine.engine import Engine
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.utils import save_image

from modules import torch_utils as tutils
from modules.global_config import GlobalConfig

from . import T, utils

save_img_fn_t = Callable[[T._batch, Tensor, str], None]
save_cm_fn_t = Callable[[Path, str, int, Figure], None]

exec_gcam_fn_t = Callable[
    [Engine, GlobalConfig, ExecuteGradCAM, tutils.Model, T._path, int, int, str], None
]
exec_softmax_fn_t = Callable[
    [T._path, GlobalConfig, Tensor, Tensor, List[str], utils.LogFile], None
]


def save_mistaken_image(batch: T._batch, y_pred: Tensor, path: str) -> None:
    x, y = batch
    ans, pred = utils.get_label(y, y_pred)
    if ans != pred:
        save_image(x, path)


def dummy_save_mistaken_image(batch: T._batch, y_pred: Tensor, path: str) -> None:
    pass


def save_cm_image(base_dir: Path, phase: str, epoch: int, fig: Figure) -> None:
    phase_lower = phase.lower()
    path = base_dir.joinpath(phase_lower, f"epoch{epoch}_{phase_lower}.jpg")
    fig.savefig(str(path))  # type: ignore


def dummy_save_cm_image(base_dir: Path, phase: str, epoch: int, fig: Figure) -> None:
    pass


def execute_gradcam(
    # epoch: int,
    # iteration: int,
    engine: Engine,
    gc: GlobalConfig,
    gcam: ExecuteGradCAM,
    model: tutils.Model,
    path: T._path,
    ans: int,
    pred: int,
    phase: str,
) -> None:
    epoch = engine.state.epoch
    # iteration = engine.state.iteration

    if not gcam.schedule[epoch - 1]:
        return

    # do not execute / execute only mistaken
    is_correct = ans == pred
    if gc.gradcam.only_mistaken and is_correct:
        return

    dir_name = "correct" if is_correct else "mistaken"
    cls_name = gcam.classes[ans]
    base_dir = gc.path.gradcam.joinpath(f"{phase}_{dir_name}", f"epoch{epoch}", cls_name)

    path = Path(path)
    ret = gcam.process_single_image(model.net, path)
    ret.pop(GradCamType.CAM)  # ignore cam

    print("\rExecute Grad-CAM...", end="")

    path = path.name
    path_stem = path[: path.rfind(".")]

    for phase, data in ret.items():  # phase: heatmap, heatmap_on_image
        # filename = f"{iteration}_{path_stem}_pred[{pred}]_ans[{ans}]_{phase}.jpg"
        filename = f"{path_stem}_pred[{pred}]_ans[{ans}]_{phase}.jpg"
        out = base_dir.joinpath(filename)
        img = data.convert("RGB")  # ?
        img.save(str(out))
    del ret

    # ret = gcam.process_multi_image(model.net, str(path))
    # ret.pop(GradCamType.CAM)
    # for phase, data_list in pbar:  # name: "heatmap", "heatmap_on_image" ...
    # ext = "jpg"
    # for phase, data_list in ret.items():
    #     for i, img in enumerate(data_list):
    #         # is_png = name == "gbp"
    #         # ext = "png" if is_png else "jpg"

    #         # s = f"{iteration}_{gcam.classes[i]}_{phase}_pred[{pred}]_correct[{ans}].{ext}"
    #         s = f"{iteration}_{phase}_"
    #         path_ = gc.gcam_base_dir.joinpath(s)

    #         # img = img.convert("RGBA" if is_png else "RGB")
    #         img = img.convert("RGB")
    #         img.save(str(path_))
    #         # print(path_)
    # del ret


def dummy_execute_gradcam(
    engine: Engine,
    gc: GlobalConfig,
    gcam: ExecuteGradCAM,
    model: tutils.Model,
    path: T._path,
    ans: int,
    pred: int,
    phase: str,
) -> None:
    pass


def execute_softmax(
    path: T._path,
    gc: GlobalConfig,
    y_pred: Tensor,
    y: Tensor,
    classes: List[str],
    softmaxfile: utils.LogFile,
) -> None:
    # batch is only 1.
    pred_softmax = F.softmax(y_pred, dim=1)[0]
    path = Path(path).name
    filename = path[: path.rfind(".")]
    softmax_list = list(map(lambda x: str(round(x.item(), 5)), pred_softmax))
    correct_cls = classes[int(y.item())]
    pred_cls = classes[int(torch.argmax(pred_softmax).item())]
    onehot = ",".join(softmax_list)
    softmaxfile.writeline(f"{correct_cls},{pred_cls},{filename},,,{onehot}")
    # print(f"{correct_cls}, {pred_cls}, {filename}, {onehot}")


def dummy_execute_softmax(
    path: T._path,
    gc: GlobalConfig,
    y_pred: Tensor,
    y: Tensor,
    classes: List[str],
    softmaxfile: utils.LogFile,
) -> None:
    pass
