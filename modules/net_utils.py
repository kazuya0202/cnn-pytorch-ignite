from typing import Union, Tuple
from dataclasses import dataclass, field

import torch.nn as nn


_param_t = Union[int, Tuple[int, int]]
_module_t = Union["Conv_", "ReLU_", "BatchNorm_", "MaxPool_", "Linear_"]


@dataclass
class _Params:
    module: nn.Module = field(init=False)

    def to_tuple(self, x: _param_t) -> Tuple[int, int]:
        return (x, x) if not isinstance(x, tuple) else x


@dataclass
class Conv_(_Params):
    in_channels: int
    out_channels: int
    kernel_size: _param_t
    stride: _param_t = (1, 1)
    padding: _param_t = (0, 0)
    dilation: _param_t = (1, 1)
    groups: int = 1
    bias: bool = True
    padding_mode: str = "zeros"

    def __post_init__(self) -> None:
        super().__init__()
        self.kernel_size = self.to_tuple(self.kernel_size)
        self.stride = self.to_tuple(self.stride)
        self.padding = self.to_tuple(self.padding)
        self.dilation = self.to_tuple(self.dilation)

        self.module = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode,
        )


@dataclass
class ReLU_(_Params):
    inplace: bool = False

    def __post_init__(self) -> None:
        super().__init__()
        self.module = nn.ReLU(inplace=self.inplace)


@dataclass
class BatchNorm_(_Params):
    num_features: int

    def __post_init__(self) -> None:
        super().__init__()
        self.module = nn.BatchNorm2d(num_features=self.num_features)


@dataclass
class MaxPool_(_Params):
    kernel_size: _param_t
    stride: _param_t = (1, 1)
    padding: _param_t = (0, 0)
    dilation: _param_t = (1, 1)

    def __post_init__(self) -> None:
        super().__init__()
        self.kernel_size = self.to_tuple(self.kernel_size)
        self.stride = self.to_tuple(self.stride)
        self.padding = self.to_tuple(self.padding)
        self.dilation = self.to_tuple(self.dilation)

        self.module = nn.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


@dataclass
class Linear_(_Params):
    in_features: int
    out_features: int

    def __post_init__(self) -> None:
        super().__init__()
        self.module = nn.Linear(in_features=self.in_features, out_features=self.out_features)
