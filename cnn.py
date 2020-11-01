from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict
from typing import OrderedDict as OrderedDictType
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modules import net_utils as nutils


@dataclass(init=False, repr=False, eq=False, unsafe_hash=False)
class BaseNetUtility:
    features_dict: OrderedDictType[str, nn.Module] = field(init=False)
    classifier_dict: OrderedDictType[str, nn.Module] = field(init=False)

    features: nn.Sequential = field(init=False)
    classifier: nn.Sequential = field(init=False)

    def make_sequential(self) -> None:
        self.features = nn.Sequential(self.features_dict)
        self.classifier = nn.Sequential(self.classifier_dict)

    def calc_linear_in_features(
        self,
        # features: dict,
        input_size: Tuple[int, int],
    ) -> int:
        # out_channels of  last conv.
        prev_out_channel = [
            int(module.out_channels)  # type: ignore
            for layer, module in self.features_dict.items()
            if layer.find("conv") > -1
        ][-1]
        h, w = input_size

        def _calc_out_shape(target: int, module: nn.Module, idx: int) -> int:
            padding = module.padding[idx]  # type: ignore
            dilation = module.dilation[idx]  # type: ignore
            kernel_size = module.kernel_size[idx]  # type: ignore
            stride = module.stride[idx]  # type: ignore
            return int((target + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1

        for phase, x in self.features_dict.items():
            if phase.find("conv") > -1 or phase.find("pool") > -1:
                h = _calc_out_shape(target=h, module=x, idx=0)
                w = _calc_out_shape(target=w, module=x, idx=1)

        return h * w * prev_out_channel


class Net(nn.Module, BaseNetUtility):
    def __init__(
        self, input_size: Tuple[int, int] = (60, 60), classify_size: int = 3, in_channels: int = 3
    ) -> None:
        super(Net, self).__init__()

        # network definiiton
        self.features_dict = OrderedDict(
            [
                # ("conv1", nutils.Conv_(in_channels, 96, kernel_size=7, stride=1, padding=3).module),
                ("conv1", nutils.Conv_(in_channels, 96, kernel_size=7, stride=1, padding=1).module),
                ("relu1", nutils.ReLU_(inplace=True).module),
                ("bn1", nutils.BatchNorm_(96).module),
                ("pool1", nutils.MaxPool_(2, stride=2).module),
                # ("conv2", nutils.Conv_(96, 128, kernel_size=5, stride=1, padding=2).module),
                ("conv2", nutils.Conv_(96, 128, kernel_size=5, stride=1, padding=1).module),
                ("relu2", nutils.ReLU_(inplace=True).module),
                ("bn2", nutils.BatchNorm_(128).module),
                # ("pool1", nutils.MaxPool_(2, stride=2).module),
                ("pool2", nutils.MaxPool_(2, stride=2).module),
                ("conv3", nutils.Conv_(128, 256, kernel_size=3, stride=1, padding=1).module),
                # ("relu3", nutils.ReLU_(inplace=True).module),
                ("conv4", nutils.Conv_(256, 384, kernel_size=3, stride=1, padding=1).module),
                # ("relu4", nutils.ReLU_(inplace=True).module),
                ("conv5", nutils.Conv_(384, 256, kernel_size=3, stride=1, padding=1).module),
                # ("relu5", nutils.ReLU_(inplace=True).module),
                # ("pool2", nutils.MaxPool_(2, stride=2).module),
            ]
        )
        self.classifier_dict: Dict[str, nn.Module] = OrderedDict(
            [
                ("fc1", nutils.Linear_(self.calc_linear_in_features(input_size), 2048).module),
                ("relu1", nutils.ReLU_(inplace=True).module),
                ("fc2", nutils.Linear_(2048, 512).module),
                ("relu2", nutils.ReLU_(inplace=True).module),
                ("fc3", nutils.Linear_(512, classify_size).module),
            ]
        )

        self.make_sequential()

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LightNet(nn.Module, BaseNetUtility):
    def __init__(
        self, input_size: Tuple[int, int] = (60, 60), classify_size: int = 3, in_channels: int = 3
    ) -> None:
        super(LightNet, self).__init__()

        self.features_dict = OrderedDict(
            [
                ("conv1", nutils.Conv_(in_channels, 48, kernel_size=7, stride=1, padding=3).module),
                ("relu1", nutils.ReLU_(inplace=True).module),
                ("bn1", nutils.BatchNorm_(48).module),
                ("conv2", nutils.Conv_(48, 96, kernel_size=5, stride=1, padding=2).module),
                ("relu2", nutils.ReLU_(inplace=True).module),
                ("bn2", nutils.BatchNorm_(96).module),
                ("pool1", nutils.MaxPool_(2).module),
                ("conv3", nutils.Conv_(96, 128, kernel_size=3, stride=1, padding=1).module),
                ("relu3", nutils.ReLU_(inplace=True).module),
                ("conv4", nutils.Conv_(128, 256, kernel_size=3, stride=1, padding=1).module),
                ("relu4", nutils.ReLU_(inplace=True).module),
                ("conv5", nutils.Conv_(256, 128, kernel_size=3, stride=1, padding=1).module),
                ("relu5", nutils.ReLU_(inplace=True).module),
                ("pool2", nutils.MaxPool_(2).module),
            ]
        )
        self.classifier_dict = OrderedDict(
            [
                ("fc1", nutils.Linear_(self.calc_linear_in_features(input_size), 2048).module),
                ("relu1", nutils.ReLU_(inplace=True).module),
                ("fc2", nutils.Linear_(2048, 512).module),
                ("relu2", nutils.ReLU_(inplace=True).module),
                ("fc3", nutils.Linear_(512, classify_size).module),
            ]
        )
        self.make_sequential()

    def forward(self, x: Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
