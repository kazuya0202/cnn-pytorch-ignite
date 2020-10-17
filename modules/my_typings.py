from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Type, Union

import cnn
from torch import Tensor, nn, optim
from modules.radam import RAdam


@dataclass(frozen=True)
class T:
    _path_t = Union[Path, str]

    _batch_t = Tuple[Tensor, Tensor]
    _batch_path_t = Tuple[Tensor, Tensor, Tensor]

    _net_t = Union[cnn.Net, cnn.LightNet]
    _optim_t = Union[optim.Adam, RAdam]
    _criterion_t = Union[nn.CrossEntropyLoss]

    _type_optim_t = Union[Type[optim.Adam], Type[RAdam]]
    _type_net_t = Union[Type[cnn.Net], Type[cnn.LightNet]]


@dataclass(frozen=True)
class State:
    TRAINING = "Training"
    UNKNOWN_VALID = "Unkonwn Validation"
    KNOWN_VALID = "Known Validation"
