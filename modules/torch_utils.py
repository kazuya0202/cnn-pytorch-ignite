import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms.transforms import Resize

from . import T, utils
from .global_config import GlobalConfig


@dataclass
class Data:
    r"""Nesessary datas of create dataset."""
    path: str  # image path
    label: int  # label of class
    name: str  # class name

    def items(self):
        r"""Returns items of Data class.

        Returns:
            tuple: Items of class.
        """
        return self.path, self.label, self.name


@dataclass
class MiniBatch:
    __data: T._batch_path_t

    batch: T._batch_t = field(init=False)
    path: Tensor = field(init=False)

    def __post_init__(self) -> None:
        data = self.__data
        self.batch = (data[0], data[1])
        self.path = data[2]


@dataclass
class CreateDataset(Dataset):
    r"""Creating dataset.

    Args:
        path (str): path of image directory.
        extensions (list): supported extensions.
        valid_size (Union[float, int]): valid image size.
        config_path (str, optional): export path of config. Defaults to 'config'.
    """
    gc: GlobalConfig

    all_list: Dict[str, List[Data]] = field(init=False)  # {'train': [], 'unknown': [], 'known': []}
    classes: Dict[int, str] = field(init=False)  # {label: 'class name', ...}

    # size of images
    all_size: int = field(init=False)
    train_size: int = field(init=False)
    unknown_size: int = field(init=False)
    known_size: int = field(init=False)

    def __post_init__(self):
        self.__initialize()

        # collect images for dataset
        if self.gc.dataset.is_pre_splited:
            self.__get_dataset_from_dir()
        else:
            self.__get_dataset()

        if self.gc.option.is_save_config:
            self._write_config()  # write config of model

        self.train_size = len(self.all_list["train"])
        self.unknown_size = len(self.all_list["unknown"])
        self.known_size = len(self.all_list["known"])

        self.all_size = self.train_size + self.unknown_size

    def __initialize(self) -> None:
        self.all_list = {"train": [], "unknown": [], "known": []}
        self.classes = {}

    def __get_dataset_from_dir(self) -> None:
        train_dirs = [x for x in Path(self.gc.dataset.train_dir).glob("*") if x.is_dir()]
        valid_dirs = [x for x in Path(self.gc.dataset.valid_dir).glob("*") if x.is_dir()]

        def glob_img(dir_: Path) -> List[Data]:
            return [
                Data(x.as_posix(), label_idx, dir_.name)
                for ext in self.gc.dataset.extensions
                for x in dir_.glob(f"*.{ext}")
                if x.is_file()
            ]

        for label_idx, (t_dir, v_dir) in enumerate(zip(train_dirs, valid_dirs)):
            # if t_dir.name != v_dir.name:  # get images that exist in both directories.
            #     continue
            train = glob_img(t_dir)
            valid = glob_img(v_dir)

            self.all_list["train"].extend(train)
            self.all_list["unknown"].extend(valid)
            self.all_list["known"].extend(random.sample(train, len(valid)))

            self.classes[label_idx] = t_dir.name

    def __get_dataset(self) -> None:
        r"""Get all datas from each directory."""
        path = Path(self.gc.path.dataset)

        # directories in [image_path]
        dirs = [x for x in path.glob("*") if x.is_dir()]

        # all extensions / all sub directories
        for label_idx, dir_ in enumerate(dirs):
            xs = [
                Data(x.as_posix(), label_idx, dir_.name)
                for ext in self.gc.dataset.extensions
                for x in dir_.glob(f"*.{ext}")
                if x.is_file()
            ]

            # adjust to limit size
            if self.gc.dataset.limit_size is not None:
                random.shuffle(xs)
                xs = xs[: self.gc.dataset.limit_size]

            # split dataset
            train, valid = train_test_split(xs, test_size=self.gc.dataset.valid_size, shuffle=True)

            self.all_list["train"].extend(train)
            self.all_list["unknown"].extend(valid)
            self.all_list["known"].extend(random.sample(train, len(valid)))

            self.classes[label_idx] = dir_.name

        # self.train_size = len(self.all_list["train"])
        # self.unknown_size = len(self.all_list["unknown"])
        # self.known_size = len(self.all_list["known"])

        # self.all_size = self.train_size + self.unknown_size

    def _write_config(self) -> None:
        r"""Writing configs."""

        cfg_dir = Path(self.gc.path.config)

        collect_list = [
            ("train_used_images.txt", "train"),
            ("unknown_used_images.txt", "unknown"),
            ("known_used_images.txt", "known"),
        ]
        for fname, target in collect_list:
            path = cfg_dir.joinpath(fname)
            with utils.LogFile(path, stdout=False) as f:
                for x in self.all_list[target]:
                    # f.writeline(str(Path(x.path).resolve()))
                    f.writeline(x.path)

    def get_dataloader(self, transform: Any = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if transform is None:
            transform = Compose(
                [
                    Resize(self.gc.network.input_size),
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        def create_dataloader(key: str, batch_size: int):
            dataset_ = CustomDataset(self.all_list[key], transform)
            return DataLoader(dataset_, batch_size, shuffle=self.gc.dataset.is_shuffle_per_epoch)

        # create dataloader
        train_loader = create_dataloader("train", batch_size=self.gc.network.batch)
        unknown_loader = create_dataloader("unknown", batch_size=1)
        known_loader = create_dataloader("known", batch_size=1)

        return train_loader, unknown_loader, known_loader


@dataclass
class CustomDataset(Dataset):
    r"""Custom dataset

    Args:
        target_list (List[Data]): dataset list.
        transform (Any, optional): transform of tensor. Defaults to None.
    """
    target_list: List[Data]
    transform: Any = None

    list_size: int = field(init=False)

    def __post_init__(self):
        self.list_size = len(self.target_list)

    def __getitem__(self, idx: int):
        r"""Returns image data, label, path

        Args:
            idx (int): index of list.

        Returns:
            tuple: image data, label, path
        """
        x = self.target_list[idx]
        path, label, _ = x.items()

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label, path

    def __len__(self):
        r"""Returns length.

        Returns:
            int: length.
        """
        return self.list_size


def clear_grads(net: T._net_t):
    for param in net.features.parameters():  # type: ignore
        param.requires_grad = False
