from collections import OrderedDict
from typing import Iterator, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Net(nn.Module):
    def __init__(
        self, input_size: Tuple[int, int] = (60, 60), classify_size: int = 3, in_channels: int = 3
    ) -> None:
        super(Net, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(in_channels, 96, kernel_size=7, stride=1, padding=3)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("bn1", nn.BatchNorm2d(96)),
                    ("conv2", nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("bn2", nn.BatchNorm2d(128)),
                    ("pool1", nn.MaxPool2d(2)),
                    ("conv3", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("conv4", nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)),
                    ("relu4", nn.ReLU(inplace=True)),
                    ("conv5", nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)),
                    ("relu5", nn.ReLU(inplace=True)),
                    ("pool2", nn.MaxPool2d(2)),
                ]
            )
        )

        in_features = calc_linear_in_features(256, input_size, self.features.named_modules())

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_features, 2048)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("fc2", nn.Linear(2048, 512)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("fc3", nn.Linear(512, classify_size)),
                ]
            )
        )

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)

        # don't run `softmax()` because of softmax process in CrossEntropyLoss
        # x = F.softmax(x)
        return x


class LightNet(nn.Module):
    def __init__(
        self, input_size: Tuple[int, int] = (60, 60), classify_size: int = 3, in_channels: int = 3
    ) -> None:
        super(LightNet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
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
        )

        in_features = calc_linear_in_features(128, input_size, self.features.named_modules())

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_features, 2048)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("fc2", nn.Linear(2048, 512)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("fc3", nn.Linear(512, classify_size)),
                ]
            )
        )

    def forward(self, x: Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)

        # don't run `softmax()` because of softmax process in CrossEntropyLoss
        # x = F.softmax(x)

        return x


def num_flat_features(x: Tensor) -> int:
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def calc_linear_in_features(
    prev_out_channel: int,
    input_size: Tuple[int, int],
    named_modules: Iterator[Tuple[str, nn.Module]],
) -> int:
    coef = 1
    # calc coefficient.
    for phase, x in named_modules:
        if phase.find("pool") > -1:
            coef *= int(x.kernel_size)  # type: ignore

    return (input_size[0] // coef) * (input_size[1] // coef) * prev_out_channel
