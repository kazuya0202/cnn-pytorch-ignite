from pathlib import Path
from typing import Tuple, Type, Union
from torch import Tensor
from torch import nn

from torch import optim
import cnn


class T:
    _path_t = Union[Path, str]

    _batch_t = Tuple[Tensor, Tensor]
    _batch_path_t = Tuple[Tensor, Tensor, Tensor]

    _net_t = Union[cnn.Net, cnn.VGG16]
    _type_net_t = Union[Type[cnn.Net], Type[cnn.VGG16], Type[cnn.LightNet]]
    _optim_t = Union[optim.Adam, optim.SGD]
    _criterion_t = Union[nn.CrossEntropyLoss]


class State:
    TRAINING = "Training"
    UNKNOWN_VALID = "Unkonwn Validation"
    KNOWN_VALID = "Known Validation"
