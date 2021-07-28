# ----------------------------------------------------- #
# Automatically generated from yaml configuration file. #
# ----------------------------------------------------- #
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data: Any) -> Union[dict, "AttrDict"]:
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dict(data[key]) for key in data})


@dataclass
class Gradcam:
    enabled: bool = True
    layer: str = "conv5"


@dataclass
class Config:
    model_path: str = "<pth path>"
    gpu_enabled: bool = True
    img_root: str = "C:/okamura/_cnn/yokoyama_test_images"
    save_root: str = "result"
    height: int = 60
    width: int = 60
    channels: int = 3
    gradcam: Gradcam = Gradcam()

    def update(self, data: dict[str, Any]):
        data_dict_as_attr = AttrDict.from_nested_dict(data)
        self.__dict__.update(data_dict_as_attr)
