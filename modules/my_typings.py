from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Type, Union

import cnn
from torch import Tensor, nn, optim
from modules.radam import RAdam


@dataclass(frozen=True)
class T:
    _path = Union[Path, str]

    _batch = Tuple[Tensor, Tensor]
    _batch_path = Tuple[Tensor, Tensor, Tensor]

    _net = Union[cnn.Net, cnn.LightNet]
    _optim = Union[optim.Adam, RAdam]
    _criterion = Union[nn.CrossEntropyLoss]

    _type_optim = Union[Type[optim.Adam], Type[RAdam]]
    _type_net = Union[Type[cnn.Net], Type[cnn.LightNet]]


@dataclass(frozen=True)
class State:
    TRAINING = "Training"
    UNKNOWN_VALID = "Unkonwn Validation"
    KNOWN_VALID = "Known Validation"
