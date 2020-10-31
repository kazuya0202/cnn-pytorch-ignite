from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from torch import optim

import cnn
import torch

from . import T, utils


@dataclass
class Path_:
    dataset: T._path = r"./dataset"
    result_dir: T._path = r"./results"
    tb_log_dir: T._path = r"./runs"

    mistaken: Path = field(init=False)
    model: Path = field(init=False)
    config: Path = field(init=False)
    gradcam: Path = field(init=False)
    log: Path = field(init=False)
    cm: Path = field(init=False)

    def __post_init__(self) -> None:
        self.dataset = utils.replace_backslash(self.dataset)
        self.result_dir = utils.replace_backslash(self.result_dir)


@dataclass
class Dataset_:
    limit_size: Optional[int] = -1
    valid_size: Union[int, float] = 0.1

    is_pre_splited: bool = False
    train_dir: T._path = r"./dataset/train"
    valid_dir: T._path = r"./dataset/valid"

    is_shuffle_per_epoch: bool = True
    extensions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.limit_size == -1:
            self.limit_size = None

        if not self.extensions:
            self.extensions = ["jpg", "png", "jpeg", "bmp"]

        if self.is_pre_splited:
            self.train_dir = utils.replace_backslash(self.train_dir)
            self.valid_dir = utils.replace_backslash(self.valid_dir)


@dataclass
class Gradcam_:
    enabled: bool = False
    only_mistaken: bool = True
    layer: str = "conv5"
    cycle: int = 10


@dataclass
class Network_:
    height: int = 60
    width: int = 60
    channels: int = 3

    epoch: int = 10
    batch: int = 128
    subdivisions: int = 4

    save_cycle: int = 0
    valid_cycle: int = 1

    gpu_enabled: bool = True
    is_save_final_model: bool = True

    net_name: str = "Net"
    net_: T._type_net = cnn.Net

    optim_name: str = "Adam"
    optim_: T._type_optim = optim.Adam

    amp: bool = True
    # fp16: bool = True

    input_size: Tuple[int, int] = field(init=False)  # height, width
    device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        self.input_size = (self.height, self.width)
        self.gpu_enabled = torch.cuda.is_available() and self.gpu_enabled  # type: ignore
        self.device = torch.device("cuda" if self.gpu_enabled else "cpu")

        self.net_ = eval(f"cnn.{self.net_name}")
        if self.optim_name == "RAdam":
            from modules.radam import RAdam

            self.optim_ = RAdam  # type: ignore
        else:
            self.optim_ = eval(f"optim.{self.optim_name}")

        if self.amp:
            try:
                from torch.cuda import amp  # noqa
            except ImportError:
                self.amp = False


@dataclass
class Option_:
    is_show_network_difinition: bool = True
    is_save_log: bool = True
    is_save_mistaken_pred: bool = False
    is_save_config: bool = False
    log_tensorboard: bool = False
    is_save_cm: bool = False


@dataclass
class GlobalConfig:
    __data: dict

    path: Path_ = field(init=False)
    dataset: Dataset_ = field(init=False)
    gradcam: Gradcam_ = field(init=False)
    network: Network_ = field(init=False)
    option: Option_ = field(init=False)

    filename_base: str = field(init=False)
    logfile: utils.LogFile = field(init=False)

    def __post_init__(self):
        if not self.__data:  # empty dict
            return

        def create_instance(cls_obj: Any, key: str) -> Any:
            dict_ = self.__data.pop(key)
            return cls_obj(**dict_) if dict_ is not None else cls_obj()

        self.path = create_instance(Path_, "path")
        self.dataset = create_instance(Dataset_, "dataset")
        self.gradcam = create_instance(Gradcam_, "gradcam")
        self.network = create_instance(Network_, "network")
        self.option = create_instance(Option_, "option")

        self.filename_base = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        self.logfile = utils.LogFile(stdout=False)
        self.ratefile = utils.LogFile(stdout=False)

        def mk_path(path: T._path, is_make: bool) -> Path:
            # return utils.concat_path(path, self.filename_base, is_make=is_make)
            return utils.concat_path(base_path, path, is_make=is_make)

        # determine directory structure / make directory
        base_path = Path(self.path.result_dir, self.filename_base)
        self.path.mistaken = mk_path("mistaken", is_make=self.option.is_save_mistaken_pred)
        self.path.model = mk_path(
            "models", is_make=self.network.save_cycle != 0 or self.network.is_save_final_model,
        )
        self.path.config = mk_path("config", is_make=self.option.is_save_config)
        self.path.gradcam = mk_path("GradCAM", is_make=self.gradcam.enabled)
        self.path.cm = mk_path("confusion_matrix", is_make=self.option.is_save_cm)

        self.path.log = base_path
        if self.option.is_save_log:
            self.path.log.mkdir(parents=True, exist_ok=True)

    def check_dataset_path(self, is_show: bool = True) -> None:
        if self.dataset.is_pre_splited:
            utils.check_existence([self.dataset.train_dir, self.dataset.valid_dir])
            if is_show:
                print(f"Creating dataset from")
                print(f"  train - '{self.dataset.train_dir}'")
                print(f"  valid - '{self.dataset.valid_dir}'...")
        else:
            utils.check_existence(self.path.dataset)
            if is_show:
                print(f"Creating dataset from '{self.path.dataset}'...")
