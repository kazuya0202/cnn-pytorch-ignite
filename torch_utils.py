import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms.transforms import Resize

import utils
from my_typings import T
from yaml_parser import GlobalConfig


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

    def __post_init__(self):
        self.all_list: Dict[str, List[Data]]  # {'train': [], 'unknown': [], 'known': []}
        self.classes: Dict[int, str]  # {label: 'class name', ...}

        # size of images
        self.all_size: int
        self.train_size: int
        self.unknown_size: int
        self.known_size: int

        self._get_dataset()  # train / unknown / known

        if self.gc.option.is_save_config:
            self._write_config()  # write config of model

    def _get_dataset(self) -> None:
        r"""Get all datas from each directory."""

        # init
        self.all_list = {"train": [], "unknown": [], "known": []}
        self.classes = {}

        path = Path(self.gc.path.dataset)

        # directories in [image_path]
        dirs = [d for d in path.glob("*") if d.is_dir()]

        # all extensions / all sub directories
        for label_idx, dir_ in enumerate(dirs):
            # xs = []
            # for ext in self.gc.dataset.extensions:
            #     tmp = [
            #         Data(x.as_posix(), label_idx, dir_.name)
            #         for x in dir_.glob(f"*.{ext}")
            #         if x.is_file()
            #     ]
            #     xs.extend(tmp)
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

        self.train_size = len(self.all_list["train"])
        self.unknown_size = len(self.all_list["unknown"])
        self.known_size = len(self.all_list["known"])

        self.all_size = self.train_size + self.unknown_size

    def _write_config(self) -> None:
        r"""Writing configs."""

        dir_ = Path(self.gc.path.config, self.gc.filename_base)
        dir_.mkdir(parents=True, exist_ok=True)

        collect_list = [
            ("train_used_images.txt", "train"),
            ("unknown_used_images.txt", "unknown"),
            ("known_used_images.txt", "known"),
        ]
        for fname, target in collect_list:
            path = dir_.joinpath(fname)
            with utils.LogFile(path, stdout=False) as f:
                for x in self.all_list[target]:
                    f.writeline(str(Path(x.path).resolve()))

    def create_dataloader(
        self, batch_size: int = 64, transform: Any = None, is_shuffle: bool = True
    ) -> Dict[str, DataLoader]:
        r"""Create DataLoader instance of `train`, `unknown`, `known` dataset.

        Args:
            batch_size (int, optional): batch size. Defaults to 64.
            transform (Any, optional): transform. Defaults to None.
            is_shuffle (bool, optional): shuffle. Defaults to True.

        Returns:
            Dict[str, DataLoader]: DataLoader.
        """

        # create dataset
        train_ = CustomDataset(self.all_list["train"], transform)
        unknown_ = CustomDataset(self.all_list["unknown"], transform)
        known_ = CustomDataset(self.all_list["known"], transform)

        train_data = DataLoader(train_, batch_size=batch_size, shuffle=is_shuffle)
        unknown_data = DataLoader(unknown_, batch_size=1, shuffle=is_shuffle)
        known_data = DataLoader(known_, batch_size=1, shuffle=is_shuffle)

        # return train_data, unknown_data, known_data
        return {"train": train_data, "unknown": unknown_data, "known": known_data}


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

        img = Image.open(path)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, label, path
        # return MiniBatch((img, label), path)

    def __len__(self):
        r"""Returns length.

        Returns:
            int: length.
        """
        return self.list_size


def get_dataloader(dataset: CreateDataset, input_size: tuple, mini_batch: int, is_shuffle: bool):
    transform = Compose(
        [Resize(input_size), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # loader = {'train': [...], 'unknown': [...], 'known': [...]}
    loader = dataset.create_dataloader(mini_batch, transform, is_shuffle)

    return loader["train"], loader["unknown"], loader["known"]
