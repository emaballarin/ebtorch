#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ──────────────────────────────────────────────────────────────────────────────
import os.path
from collections.abc import Callable
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from PIL import Image
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset


class TinyImagenet(VisionDataset):
    """TinyImagenet Dataset.
    Args:
        root (string): Root directory of the dataset where the data is stored.
        split (string): One of {"train", "val"}.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    split_dl_info: Dict[str, Tuple[str, str, str]] = {
        "train": (
            "https://bucket.ballarin.cc/data/tiny_imagenet_200/tiny-imagenet-200-npz/train.npz",
            "train.npz",
            "db414016436353892fdf00cb30b9ee57",
        ),
        "val": (
            "https://bucket.ballarin.cc/data/tiny_imagenet_200/tiny-imagenet-200-npz/val.npz",
            "val.npz",
            "7762694b6217fec8ba1a7be3c20ef218",
        ),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split: str = verify_str_arg(
            split, "split", tuple(self.split_dl_info.keys())
        )
        self.url: str = self.split_dl_info[split][0]
        self.filename: str = self.split_dl_info[split][1]
        self.file_md5: str = self.split_dl_info[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it (again)"
            )

        # reading(loading) npz file as array
        loaded_npz = np.load(os.path.join(self.root, self.filename))
        self.data = loaded_npz["image"]
        self.targets: list = loaded_npz["label"].tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = Image.fromarray(self.data[index]), int(self.targets[index])

        img = self.transform(img) if self.transform is not None else img
        target = (
            self.target_transform(target)
            if self.target_transform is not None
            else target
        )

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_dl_info[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_dl_info[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
