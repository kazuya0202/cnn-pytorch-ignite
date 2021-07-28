from datetime import datetime
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import Tensor
from torchvision import transforms

# from torchvision.utils import save_image

import cnn  # noqa
from modules import T
from texttable import Texttable


@dataclass
class Predict:
    label: int = -1
    name: str = "None"
    rate: float = 0.0
    path: T._path = ""

    is_pred: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_pred = self.label != -1


@dataclass
class TestConfig:
    model_path: T._path = r""
    gpu_enabled: bool = True
    img_root: str = r""
    save_root: str = r""

    height: int = 60
    width: int = 60
    channels: int = 3
    class_name: str = "Net"

    imgs: List[Path] = field(init=False, default_factory=list)
    class_: T._type_net = field(init=False)
    input_size: Tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.class_ = eval(f"cnn.{self.class_name}")
        self.input_size = (self.height, self.width)

        exts = ["jpg", "png", "jpeg", "bmp", "gif"]
        root = Path(self.img_root)
        for ext in exts:
            self.imgs.extend(root.glob(f"*.{ext}"))


@dataclass
class ValidModel:
    cfg: TestConfig
    transform: Any = None

    classes: Dict[int, str] = field(init=False)
    classify_size: int = field(init=False)
    # save_root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.gpu_enabled = self.gpu_enabled and torch.cuda.is_available()  # type: ignore
        self.device = torch.device("cuda" if self.gpu_enabled else "cpu")
        self.__load(self.cfg.model_path)

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.cfg.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        if self.cfg.save_root == "":
            self.cfg.save_root = "result"
        self.save_root = Path(self.cfg.save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

    def __load(self, path: T._path):
        check_existence(path)

        ckpt = torch.load(path)
        self.classes = ckpt["classes"]
        self.classify_size = len(self.classes)

        # network
        self.net = self.cfg.class_(
            input_size=self.cfg.input_size,
            classify_size=self.classify_size,
            in_channels=self.cfg.channels,
        )
        self.net.load_state_dict(ckpt["model_state_dict"])

        self.net = self.net.to(self.device)
        self.net.eval()

    def run(self) -> Iterator[Predict]:
        for path in self.cfg.imgs:
            yield self.valid(path)

    def valid(self, img_path: Path) -> Predict:
        if not img_path.exists():
            return Predict()

        with torch.no_grad():
            img = self.preprocess(img_path)

            x: Tensor = self.net(img)
            # save_image(x, self.save_root / img_path.name)
            # todo: save grad-cam image

            x_sm = F.softmax(x, -1)
            pred = torch.max(x_sm.data, 1)  # type: ignore

            label = int(pred[1].item())
            return Predict(
                label, name=self.classes[label], rate=float(pred[0].item()), path=img_path
            )

    def preprocess(self, path: T._path) -> Tensor:
        img_pil = Image.open(path).convert("RGB")
        img: Tensor = self.transform(img_pil)
        img = img.unsqueeze_(0).to(self.device)
        return img


def parse_arg() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-c", "--cfg", help="config file", default="user_config_for_test.yaml")

    return parser.parse_args()


def parse_yaml(path: str) -> TestConfig:
    check_existence(path)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return TestConfig(**data)


def check_existence(path: T._path) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' does not exist.")


if __name__ == "__main__":
    args = parse_arg()
    cfg = parse_yaml(args.cfg)

    save_root = Path(cfg.save_root)
    runtime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    # softmax_file = oepn("")

    # transform = ...
    transform = None
    model = ValidModel(cfg, transform)

    for x in model.run():
        if x.is_pred:
            print(f"{x.label} | {x.path}")
