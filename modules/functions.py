from pathlib import Path
from typing import Callable

from gradcam import ExecuteGradCAM
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


def save_mistaken_image(batch: T._batch, y_pred: Tensor, path: str) -> None:
    x, y = batch
    ans, pred = utils.get_label(y, y_pred)
    if ans != pred:
        save_image(x, path)


def dummy_save_mistaken_image(batch: T._batch, y_pred: Tensor, path: str) -> None:
    return


def save_cm_image(base_dir: Path, phase: str, epoch: int, fig: Figure) -> None:
    phase_lower = phase.lower()
    path = base_dir.joinpath(phase_lower, f"epoch{epoch}_{phase_lower}.jpg")
    fig.savefig(str(path))  # type: ignore


def dummy_save_cm_image(base_dir: Path, phase: str, epoch: int, fig: Figure) -> None:
    return


def execute_gradcam(
    engine: Engine,
    gc: GlobalConfig,
    gcam: ExecuteGradCAM,
    model: tutils.Model,
    path: T._path,
    ans: int,
    pred: int,
    phase: str,
) -> None:
    if not gcam.schedule[engine.state.epoch - 1]:
        return

    # do not execute / execute only mistaken
    is_correct = ans == pred
    if gc.gradcam.only_mistaken and is_correct:
        return
    epoch = engine.state.epoch
    iteration = engine.state.iteration

    gcam_base_dir = Path(gc.path.gradcam)
    epoch_str = f"epoch{epoch}"

    dir_name = "correct" if is_correct else "mistaken"
    base_dir = utils.concat_path(
        gcam_base_dir, concat=[f"{phase}_{dir_name}", epoch_str], is_make=True
    )

    ret = gcam.main(model.net, str(path))
    ret.pop("gbp")  # ignore gbp
    ret.pop("ggcam")  # ignore ggcam
    # pbar = tqdm.tqdm(ret.items(), desc="Grad-CAM", leave=False)
    print("\rExecute Grad-CAM...", end="")

    # for name, data_list in pbar:  # name: "gcam", "gbp" ...
    ext = "jpg"
    for phase, data_list in ret.items():  # name: "gcam", "gbp" ...
        for i, img in enumerate(data_list):
            # is_png = name == "gbp"
            # ext = "png" if is_png else "jpg"

            s = f"{iteration}_{gcam.classes[i]}_{phase}_pred[{pred}]_correct[{ans}].{ext}"
            path_ = base_dir.joinpath(s)

            # img = img.convert("RGBA" if is_png else "RGB")
            img = img.convert("RGB")
            img.save(str(path_))
            # print(path_)
    del ret


def dummy_execute_gradcam(
    engine: Engine,
    gc: GlobalConfig,
    gcam: ExecuteGradCAM,
    model: tutils.Model,
    path: T._path,
    ans: int,
    pred: int,
    name: str,
) -> None:
    return
