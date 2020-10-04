from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

import torch
import yaml

import cnn
import utils
from my_typings import T


@dataclass
class Path_:
    dataset: str = r"./dataset"
    mistaken: str = r"./mistaken"
    model: str = r"./"
    config: str = r"./config"
    log: str = r"./logs"
    gradcam: str = r"./GradCAM_results"


@dataclass
class Dataset_:
    limit_size: Optional[int] = -1
    valid_size: Union[int, float] = 0.1
    extensions: List[str] = field(default_factory=list)

    is_shuffle_per_epoch: bool = True

    # use_enum_txt: bool = False
    # train_txt: T._path_t = ""
    # valid_txt: T._path_t = ""

    def __post_init__(self) -> None:
        if self.limit_size == -1:
            self.limit_size = None

        if not self.extensions:
            self.extensions = ["jpg", "png", "jpeg", "bmp"]

        # if self.use_enum_txt:
        #     self.train_txt = Path(self.train_txt)
        #     self.valid_txt = Path(self.valid_txt)

        #     self.check_exist(self.train_txt)
        #     self.check_exist(self.valid_txt)

    # def check_exist(self, path: Path) -> None:
    #     if not path.exists():
    #         raise FileNotFoundError(f"'{path}' is not exist.")


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


@dataclass
class Option_:
    is_show_network_difinition: bool = True
    verbose: bool = True

    is_debug: bool = False
    is_save_log: bool = True
    # is_save_rate_log: bool = True
    is_save_mistaken_pred: bool = False
    is_save_config: bool = True
    log_tensorboard: bool = False

    # is_available_re_training: bool = False
    # re_training: bool = False
    # load_model_path: str = r""


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


def parse(path: str) -> GlobalConfig:
    path = path if path is not None else "user_config.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return GlobalConfig(data)
