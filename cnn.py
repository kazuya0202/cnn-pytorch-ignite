from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class _NetUtility:
    def calc_linear_in_features(
        self, features_dict: Dict[str, nn.Module], input_size: Tuple[int, int]
    ) -> int:
        # out_channels of  last conv.
        prev_out_channel = [
            int(module.out_channels)  # type: ignore
            for layer, module in features_dict.items()
            if layer.find("conv") > -1
        ][-1]
        h, w = input_size

        def _num2tuple(num) -> tuple:
            return num if isinstance(num, tuple) else (num, num)

        def _preprocess(module: nn.Module) -> None:
            module.padding = _num2tuple(module.padding)
            module.dilation = _num2tuple(module.dilation)
            module.kernel_size = _num2tuple(module.kernel_size)
            module.stride = _num2tuple(module.stride)

        def _calc_out_shape(target: int, module: nn.Module, idx: int) -> int:
            padding = module.padding[idx]  # type: ignore
            dilation = module.dilation[idx]  # type: ignore
            kernel_size = module.kernel_size[idx]  # type: ignore
            stride = module.stride[idx]  # type: ignore
            return int((target + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1

        for phase, x in features_dict.items():
            if phase.find("conv") > -1 or phase.find("pool") > -1:
                _preprocess(module=x)
                h = _calc_out_shape(target=h, module=x, idx=0)
                w = _calc_out_shape(target=w, module=x, idx=1)
        return h * w * prev_out_channel


class Net(nn.Module, _NetUtility):
    def __init__(
        self, input_size: Tuple[int, int] = (60, 60), classify_size: int = 3, in_channels: int = 3
    ) -> None:
        super(Net, self).__init__()
        features_dict: Dict[str, nn.Module]
        classifier_dict: Dict[str, nn.Module]

        self.features: nn.Sequential
        self.classifier: nn.Sequential

        # network definiiton
        features_dict = OrderedDict(
            [
                ("conv1", nn.Conv2d(in_channels, 96, kernel_size=7, stride=1, padding=1)),
                ("relu1", nn.ReLU(inplace=True)),
                ("bn1", nn.BatchNorm2d(96)),
                ("pool1", nn.MaxPool2d(2, stride=2)),
                ("conv2", nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=1)),
                ("relu2", nn.ReLU(inplace=True)),
                ("bn2", nn.BatchNorm2d(128)),
                ("pool2", nn.MaxPool2d(2, stride=2)),
                ("conv3", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
                ("conv4", nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)),
                ("conv5", nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)),
            ]
        )

        in_features = self.calc_linear_in_features(features_dict, input_size)
        classifier_dict = OrderedDict(
            [
                ("fc1", nn.Linear(in_features, 2048)),
                ("relu1", nn.ReLU(inplace=True)),
                ("fc2", nn.Linear(2048, 512)),
                ("relu2", nn.ReLU(inplace=True)),
                ("fc3", nn.Linear(512, classify_size)),
            ]
        )

        # make sequential
        self.features = nn.Sequential(features_dict)
        self.classifier = nn.Sequential(classifier_dict)

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LightNet(nn.Module, _NetUtility):
    def __init__(
        self, input_size: Tuple[int, int] = (60, 60), classify_size: int = 3, in_channels: int = 3
    ) -> None:
        super(LightNet, self).__init__()
        features_dict: Dict[str, nn.Module]
        classifier_dict: Dict[str, nn.Module]

        self.features: nn.Sequential
        self.classifier: nn.Sequential

        features_dict = OrderedDict(
            [
                ("conv1", nn.Conv2d(in_channels, 48, kernel_size=7, stride=1, padding=3)),
                ("relu1", nn.ReLU(inplace=True)),
                ("bn1", nn.BatchNorm2d(48)),
                ("conv2", nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=2)),
                ("relu2", nn.ReLU(inplace=True)),
                ("bn2", nn.BatchNorm2d(96)),
                ("pool1", nn.MaxPool2d(2)),
                ("conv3", nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)),
                ("relu3", nn.ReLU(inplace=True)),
                ("conv4", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
                ("relu4", nn.ReLU(inplace=True)),
                ("conv5", nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)),
                ("relu5", nn.ReLU(inplace=True)),
                ("pool2", nn.MaxPool2d(2)),
            ]
        )

        in_features = self.calc_linear_in_features(features_dict, input_size)
        classifier_dict = OrderedDict(
            [
                ("fc1", nn.Linear(in_features, 2048)),
                ("relu1", nn.ReLU(inplace=True)),
                ("fc2", nn.Linear(2048, 512)),
                ("relu2", nn.ReLU(inplace=True)),
                ("fc3", nn.Linear(512, classify_size)),
            ]
        )

        # make sequential
        self.features = nn.Sequential(features_dict)
        self.classifier = nn.Sequential(classifier_dict)

    def forward(self, x: Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
