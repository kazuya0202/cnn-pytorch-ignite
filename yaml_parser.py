from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import yaml

import cnn
import utils
from my_typings import T


@dataclass
class Path_:
    dataset: T._path_t = r"./dataset"
    mistaken: T._path_t = r"./mistaken"
    model: T._path_t = r"./models"
    config: T._path_t = r"./config"
    log: T._path_t = r"./logs"
    gradcam: T._path_t = r"./GradCAM_results"


@dataclass
class Dataset_:
    limit_size: Optional[int] = -1
    valid_size: Union[int, float] = 0.1
    extensions: List[str] = field(default_factory=list)

    is_shuffle_per_epoch: bool = True

    def __post_init__(self) -> None:
        if self.limit_size == -1:
            self.limit_size = None

        if not self.extensions:
            self.extensions = ["jpg", "png", "jpeg", "bmp"]


@dataclass
class Gradcam_:
    enabled: bool = False
    only_mistaken: bool = True
    layer: str = "conv5"

    gpu_enabled: bool = True


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

    class_name: str = "Net"
    class_: T._type_net_t = cnn.Net

    input_size: Tuple[int, int] = field(init=False)  # height, width
    device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        self.input_size = (self.height, self.width)
        self.gpu_enabled = torch.cuda.is_available() and self.gpu_enabled
        self.device = torch.device("cuda" if self.gpu_enabled else "cpu")

        if self.class_name == "VGG16":
            self.class_ = cnn.VGG16
        elif self.class_name == "LightNet":
            self.class_ = cnn.LightNet


@dataclass
class Option_:
    is_show_network_difinition: bool = True
    verbose: bool = True

    # is_debug: bool = False
    is_save_log: bool = True
    is_save_mistaken_pred: bool = False
    is_save_config: bool = True
    log_tensorboard: bool = False


@dataclass
class GlobalConfig:
    __data: dict

    path: Path_ = field(init=False)
    dataset: Dataset_ = field(init=False)
    gradcam: Gradcam_ = field(init=False)
    network: Network_ = field(init=False)
    option: Option_ = field(init=False)

    filename_base: str = field(init=False)
    log: utils.LogFile = field(init=False)

    def __post_init__(self):
        if not self.__data:  # empty dict
            return

        path_ = self.__data.pop("path")
        dataset_ = self.__data.pop("dataset")
        gradcam_ = self.__data.pop("gradcam")
        network_ = self.__data.pop("network")
        option_ = self.__data.pop("option")

        def create_instance(cls_obj: Any, dict_: dict) -> Any:
            return cls_obj(**dict_) if dict_ is not None else cls_obj()

        self.path = create_instance(Path_, path_)
        self.dataset = create_instance(Dataset_, dataset_)
        self.gradcam = create_instance(Gradcam_, gradcam_)
        self.network = create_instance(Network_, network_)
        self.option = create_instance(Option_, option_)

        self.filename_base = datetime.now().strftime("%Y%b%d_%Hh%Mm%Ss")
        self.log = utils.LogFile(stdout=False)

        # determine directory structure / make directory
        self.path.mistaken = utils.concat_path_and_mkdir(
            self.path.mistaken, self.filename_base, is_make=self.option.is_save_mistaken_pred
        )
        self.path.model = utils.concat_path_and_mkdir(
            self.path.model,
            self.filename_base,
            is_make=self.network.save_cycle != 0 or self.network.is_save_final_model,
        )
        self.path.config = utils.concat_path_and_mkdir(
            self.path.config, self.filename_base, is_make=self.option.is_save_config
        )
        self.path.gradcam = utils.concat_path_and_mkdir(
            self.path.gradcam, self.filename_base, is_make=self.gradcam.enabled
        )

        Path(self.path.log).mkdir(parents=True, exist_ok=True)


def parse(path: str) -> GlobalConfig:
    path = path if path is not None else "user_config.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return GlobalConfig(data)
