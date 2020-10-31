import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from . import T, utils
from .global_config import GlobalConfig


@dataclass
class Data:
    r"""Data of dataset."""
    path: str  # image path
    label: int  # label of class
    name: str  # class name

    def items(self) -> Tuple[str, int, str]:
        r"""Returns items of Data class.

        Returns:
            Tuple[str, int, str]: Items of class.
        """
        return self.path, self.label, self.name


@dataclass
class MiniBatch:
    r"""Data of mini batch.

    Args:
        __data (T._batch_path_t): img, label and path.
    """
    __data: T._batch_path

    batch: T._batch = field(init=False)
    path: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.batch = (self.__data[0], self.__data[1])
        self.path = self.__data[2]


@dataclass
class CreateDataset(Dataset):
    r"""Creating dataset.

    Args:
        gc (GlobalConfig): variable of config.
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
        # init
        self.all_list = {"train": [], "unknown": [], "known": []}
        self.classes = {}

        # collect images for dataset
        if self.gc.dataset.is_pre_splited:
            self.__get_dataset_from_dir()
        else:
            self.__get_dataset()

        self.train_size = len(self.all_list["train"])
        self.unknown_size = len(self.all_list["unknown"])
        self.known_size = len(self.all_list["known"])

        self.all_size = self.train_size + self.unknown_size
        self.__check_data_size()

    def __check_data_size(self) -> None:
        if self.all_size == 0:
            raise ValueError(f"data size == 0, path of dataset is invalid.")

    def __glob_data(self, dir_: Path, label_idx: int, *, shuffle: bool = True) -> List[Data]:
        data_list = [
            Data(x.as_posix(), label_idx, dir_.name)
            for ext in self.gc.dataset.extensions
            for x in dir_.glob(f"*.{ext}")
            if x.is_file()
        ]
        if shuffle:
            random.shuffle(data_list)
        return data_list

    def __extend_dataset(self, train: List[Data], valid: List[Data]) -> None:
        self.all_list["train"].extend(train)
        self.all_list["unknown"].extend(valid)
        self.all_list["known"].extend(random.sample(train, len(valid)))

    def __get_dataset_from_dir(self) -> None:
        r"""Get dataset from each directory."""
        train_dirs = sorted([x for x in Path(self.gc.dataset.train_dir).glob("*") if x.is_dir()])
        valid_dirs = sorted([x for x in Path(self.gc.dataset.valid_dir).glob("*") if x.is_dir()])

        for label_idx, (t_dir, v_dir) in enumerate(zip(train_dirs, valid_dirs)):
            if t_dir.name != v_dir.name:  # get images that exist in both directories.
                print(f"Different class name: 'train/{t_dir.name}' | 'valid/{v_dir.name}'")
                continue
            train = self.__glob_data(t_dir, label_idx, shuffle=True)
            valid = self.__glob_data(v_dir, label_idx, shuffle=True)

            self.__extend_dataset(train, valid)
            self.classes[label_idx] = t_dir.name

    def __get_dataset(self) -> None:
        r"""Get dataset and split dataset(train/unknown) at random."""
        path = Path(self.gc.path.dataset)

        # directories in [image_path]
        dirs = sorted([x for x in path.glob("*") if x.is_dir()])

        # all extensions / all sub directories
        for label_idx, dir_ in enumerate(dirs):
            data_list = self.__glob_data(dir_, label_idx, shuffle=True)

            # adjust to limit size
            if self.gc.dataset.limit_size is not None:
                data_list = data_list[: self.gc.dataset.limit_size]

            # split dataset
            train, valid = train_test_split(data_list, test_size=self.gc.dataset.valid_size)

            self.__extend_dataset(train, valid)
            self.classes[label_idx] = dir_.name

    def write_config(self) -> None:
        r"""Writing configs."""
        collect_list = [
            ("train_used_images.txt", "train"),
            ("unknown_used_images.txt", "unknown"),
            ("known_used_images.txt", "known"),
        ]
        for fname, target in collect_list:
            path = self.gc.path.config.joinpath(fname)
            with utils.LogFile(path, stdout=False) as f:
                for x in self.all_list[target]:
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

        def create_dataloader(key: str, batch_size: int, cpus: int) -> DataLoader:
            dataset_ = CustomDataset(self.all_list[key], transform)
            return DataLoader(dataset_, batch_size, shuffle=self.gc.dataset.is_shuffle_per_epoch, num_workers=cpus, pin_memory=True)

        cpus = os.cpu_count()
        cpus = 2 if cpus else 0

        # create dataloader
        train_loader = create_dataloader("train", batch_size=self.gc.network.batch, cpus=cpus)
        unknown_loader = create_dataloader("unknown", batch_size=1, cpus=cpus)
        known_loader = create_dataloader("known", batch_size=1, cpus=cpus)

        return train_loader, unknown_loader, known_loader


@dataclass
class CustomDataset(Dataset):
    r"""Custom dataset.

    Args:
        target_list (List[Data]): dataset list.
        transform (Any, optional): transform of tensor. Defaults to None.
    """
    target_list: List[Data]
    transform: Optional[Any] = None

    list_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.list_size = len(self.target_list)

    def __getitem__(self, idx: int) -> Tuple[Any, int, str]:
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

    def __len__(self) -> int:
        r"""Returns length.

        Returns:
            int: length.
        """
        return self.list_size
