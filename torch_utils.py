import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms.transforms import Resize

# my packages
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
class CreateDataset(Dataset):
    r"""Creating dataset.

    Args:
        path (str): path of image directory.
        extensions (list): supported extensions.
        test_size (Union[float, int]): test image size.
        config_path (str, optional): export path of config. Defaults to 'config'.
    """
    GCONF: GlobalConfig

    def __post_init__(self):
        # {'train': [], 'unknown': [], 'known': []}
        self.all_list: Dict[str, List[Data]]

        # {label: 'class name' ...}
        self.classes: Dict[int, str]

        # size of images
        self.all_size: int  # train_size + unknown_size
        self.train_size: int
        self.unknown_size: int
        self.known_size: int

        # ----------

        self._get_all_datas()  # train / unknown / known

        if self.GCONF.option.is_save_config:
            self._write_config()  # write config of model

    def _get_all_datas(self) -> None:
        r"""Get all datas from each directory."""

        # init
        self.all_list = {"train": [], "unknown": [], "known": []}
        self.classes = {}

        path = Path(self.GCONF.path.dataset)

        # directories in [image_path]
        dirs = [d for d in path.glob("*") if d.is_dir()]

        # all extensions / all sub directories
        for label_idx, _dir in enumerate(dirs):
            xs = []

            # for ext in self.extensions:
            for ext in self.GCONF.dataset.extensions:
                tmp = [
                    Data(x.as_posix(), label_idx, _dir.name)
                    for x in _dir.glob(f"*.{ext}")
                    if x.is_file()
                ]
                xs.extend(tmp)

            # adjust to limit size
            # if self.limit_size is not None:
            if self.GCONF.dataset.limit_size is not None:
                random.shuffle(xs)
                xs = xs[: self.GCONF.dataset.limit_size]

            # split dataset
            train, test = train_test_split(xs, test_size=self.GCONF.dataset.test_size, shuffle=True)

            self.all_list["train"].extend(train)
            self.all_list["unknown"].extend(test)
            self.all_list["known"].extend(random.sample(train, len(test)))

            self.classes[label_idx] = _dir.name

        self.train_size = len(self.all_list["train"])
        self.unknown_size = len(self.all_list["unknown"])
        self.known_size = len(self.all_list["known"])

        self.all_size = self.train_size + self.unknown_size

    def _write_config(self) -> None:
        r"""Writing configs."""

        _dir = Path(self.GCONF.path.config)
        _dir.mkdir(parents=True, exist_ok=True)

        def _inner_execute(add_path: str, target: str):
            pass
            # path = _dir.joinpath(add_path)
            # with ul.LogFile(path, std_debug_ok=False, _clear=True) as f:
            #     for x in self.all_list[target]:
            #         p = Path(x.path).resolve()  # convert to absolute path
            #         f.writeline(p.as_posix())

        _inner_execute("train_used_images.txt", "train")  # train
        _inner_execute("unknown_used_images.txt", "unknown")  # unknown
        _inner_execute("known_used_images.txt", "known")  # known

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

    # def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
    def __getitem__(self, idx: int):
        r"""Returns image data, label, path

        Args:
            idx (int): index of list.

        Returns:
            tuple: image data, label, path
        """
        x = self.target_list[idx]
        path, label, name = x.items()

        img = Image.open(path)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        # return img, label, path
        return img, label

    def __len__(self):
        r"""Returns length.

        Returns:
            int: length.
        """
        return self.list_size


def get_data_loader(dataset: CreateDataset, input_size: tuple, mini_batch: int, is_shuffle: bool):
    # transform
    transform = Compose(
        [Resize(input_size), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # loader = {'train': [...], 'unknown': [...], 'known': [...]}
    loader = dataset.create_dataloader(mini_batch, transform, is_shuffle)

    return loader["train"], loader["unknown"], loader["known"]
