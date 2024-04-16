#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
import os
from typing import Optional
from typing import Tuple

import torch
from medmnist import OCTMNIST
from medmnist import PathMNIST
from medmnist import TissueMNIST
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import DatasetFolder
from torchvision.datasets import FashionMNIST
from torchvision.datasets import ImageFolder
from torchvision.datasets import KMNIST
from torchvision.datasets import MNIST
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

__all__ = [
    "mnist_dataloader_dispatcher",
    "fashionmnist_dataloader_dispatcher",
    "kmnist_dataloader_dispatcher",
    "cifarten_dataloader_dispatcher",
    "cifarhundred_dataloader_dispatcher",
    "imagenette_dataloader_dispatcher",
    "pathmnist_dataloader_dispatcher",
    "octmnist_dataloader_dispatcher",
    "tissuemnist_dataloader_dispatcher",
]

data_root_literal: str = "../datasets/"
cuda_args_true: dict = {"pin_memory": True}


def _determine_train_test_args_common(dataset_name: str, is_train: bool) -> dict:
    if dataset_name in ("pathmnist", "octmnist", "tissuemnist"):
        if is_train:
            return {"split": "train"}
        else:
            return {"split": "test"}
    else:
        if is_train:
            return {"train": True}
        else:
            return {"train": False}


def _dataloader_dispatcher(
    dataset: str,
    data_root: str = data_root_literal,
    batch_size_train: Optional[int] = None,
    batch_size_test: Optional[int] = None,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if dataset == "mnist":
        dataset_fx = MNIST
        if batch_size_train is None:
            batch_size_train: int = 256
        if batch_size_test is None:
            batch_size_test: int = 512

    elif dataset == "fashionmnist":
        dataset_fx = FashionMNIST
        if batch_size_train is None:
            batch_size_train: int = 256
        if batch_size_test is None:
            batch_size_test: int = 512

    elif dataset == "kmnist":
        dataset_fx = KMNIST
        if batch_size_train is None:
            batch_size_train: int = 256
        if batch_size_test is None:
            batch_size_test: int = 512

    elif dataset == "cifar10":
        dataset_fx = CIFAR10
        if batch_size_train is None:
            batch_size_train: int = 256
        if batch_size_test is None:
            batch_size_test: int = 512

    elif dataset == "cifar100":
        dataset_fx = CIFAR100
        if batch_size_train is None:
            batch_size_train: int = 256
        if batch_size_test is None:
            batch_size_test: int = 512

    elif dataset == "pathmnist":
        dataset_fx = PathMNIST
        if batch_size_train is None:
            batch_size_train: int = 256
        if batch_size_test is None:
            batch_size_test: int = 512

    elif dataset == "octmnist":
        dataset_fx = OCTMNIST
        if batch_size_train is None:
            batch_size_train: int = 256
        if batch_size_test is None:
            batch_size_test: int = 512

    elif dataset == "tissuemnist":
        dataset_fx = TissueMNIST
        if batch_size_train is None:
            batch_size_train: int = 256
        if batch_size_test is None:
            batch_size_test: int = 512

    else:
        raise ValueError("Dataset not supported... yet!")

    os.makedirs(name=data_root, exist_ok=True)

    transforms = Compose([ToTensor()])

    # Address dictionary mutability as default argument
    if dataset_kwargs is None:
        dataset_kwargs: dict = {}
    if dataloader_kwargs is None:
        dataloader_kwargs: dict = {}

    trainset = dataset_fx(
        root=data_root,
        **_determine_train_test_args_common(dataset, is_train=True),
        transform=transforms,
        download=True,
        **dataset_kwargs,
    )
    testset = dataset_fx(
        root=data_root,
        **_determine_train_test_args_common(dataset, is_train=False),
        transform=transforms,
        download=True,
        **dataset_kwargs,
    )

    cuda_args: dict = {}
    if cuda_accel:
        cuda_args: dict = cuda_args_true

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=batch_size_train,
        shuffle=(not unshuffle_train),
        **cuda_args,
        **dataloader_kwargs,
    )
    testloader = DataLoader(
        dataset=testset,
        batch_size=batch_size_test,
        shuffle=shuffle_test,
        **cuda_args,
        **dataloader_kwargs,
    )
    test_on_train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size_test,
        shuffle=shuffle_test,
        **cuda_args,
        **dataloader_kwargs,
    )

    return trainloader, testloader, test_on_train_loader


def mnist_dataloader_dispatcher(
    data_root: str = data_root_literal,
    batch_size_train: Optional[int] = None,
    batch_size_test: Optional[int] = None,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return _dataloader_dispatcher(
        dataset="mnist",
        data_root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        cuda_accel=cuda_accel,
        unshuffle_train=unshuffle_train,
        shuffle_test=shuffle_test,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )


def fashionmnist_dataloader_dispatcher(
    data_root: str = data_root_literal,
    batch_size_train: Optional[int] = None,
    batch_size_test: Optional[int] = None,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return _dataloader_dispatcher(
        dataset="fashionmnist",
        data_root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        cuda_accel=cuda_accel,
        unshuffle_train=unshuffle_train,
        shuffle_test=shuffle_test,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )


def kmnist_dataloader_dispatcher(
    data_root: str = data_root_literal,
    batch_size_train: Optional[int] = None,
    batch_size_test: Optional[int] = None,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return _dataloader_dispatcher(
        dataset="kmnist",
        data_root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        cuda_accel=cuda_accel,
        unshuffle_train=unshuffle_train,
        shuffle_test=shuffle_test,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )


def cifarten_dataloader_dispatcher(
    data_root: str = data_root_literal,
    batch_size_train: Optional[int] = None,
    batch_size_test: Optional[int] = None,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return _dataloader_dispatcher(
        dataset="cifar10",
        data_root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        cuda_accel=cuda_accel,
        unshuffle_train=unshuffle_train,
        shuffle_test=shuffle_test,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )


def cifarhundred_dataloader_dispatcher(
    data_root: str = data_root_literal,
    batch_size_train: Optional[int] = None,
    batch_size_test: Optional[int] = None,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return _dataloader_dispatcher(
        dataset="cifar100",
        data_root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        cuda_accel=cuda_accel,
        unshuffle_train=unshuffle_train,
        shuffle_test=shuffle_test,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )


def pathmnist_dataloader_dispatcher(
    data_root: str = data_root_literal,
    batch_size_train: Optional[int] = None,
    batch_size_test: Optional[int] = None,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return _dataloader_dispatcher(
        dataset="pathmnist",
        data_root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        cuda_accel=cuda_accel,
        unshuffle_train=unshuffle_train,
        shuffle_test=shuffle_test,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )


def octmnist_dataloader_dispatcher(
    data_root: str = data_root_literal,
    batch_size_train: Optional[int] = None,
    batch_size_test: Optional[int] = None,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return _dataloader_dispatcher(
        dataset="octmnist",
        data_root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        cuda_accel=cuda_accel,
        unshuffle_train=unshuffle_train,
        shuffle_test=shuffle_test,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )


def tissuemnist_dataloader_dispatcher(
    data_root: str = data_root_literal,
    batch_size_train: Optional[int] = None,
    batch_size_test: Optional[int] = None,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return _dataloader_dispatcher(
        dataset="tissuemnist",
        data_root=data_root,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        cuda_accel=cuda_accel,
        unshuffle_train=unshuffle_train,
        shuffle_test=shuffle_test,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )


def imagenette_dataloader_dispatcher(
    data_root: str = data_root_literal,
    batch_size_train: int = 64,
    batch_size_test: int = 128,
    cuda_accel: bool = False,
    unshuffle_train: bool = False,
    shuffle_test: bool = False,
    dataset_kwargs: Optional[dict] = None,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if dataset_kwargs is None:
        dataset_kwargs: dict = {}

    train_ds: DatasetFolder = ImageFolder(
        root=data_root + "imagenette2-320/train",
        transform=Compose(
            [
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        ),
        **dataset_kwargs,
    )

    test_ds: DatasetFolder = ImageFolder(
        root=data_root + "imagenette2-320/val",
        transform=Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
            ]
        ),
        **dataset_kwargs,
    )

    if dataloader_kwargs is None:
        dataloader_kwargs: dict = {}

    cuda_kwargs: dict = {}
    if cuda_accel:
        cuda_kwargs: dict = cuda_args_true

    train_dl: DataLoader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size_train,
        shuffle=(not unshuffle_train),
        **cuda_kwargs,
        **dataloader_kwargs,
    )

    test_dl: DataLoader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size_test,
        shuffle=shuffle_test,
        **cuda_kwargs,
        **dataloader_kwargs,
    )

    tot_dl: DataLoader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size_test,
        shuffle=shuffle_test,
        **cuda_kwargs,
        **dataloader_kwargs,
    )

    return train_dl, test_dl, tot_dl


def _dataloader_mean_std(dataloader: DataLoader, printout: bool = False):
    mean, std, nsamples = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            data, _ = batch
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nsamples += batch_samples
    mean /= nsamples
    std /= nsamples
    if printout:
        print("Mean: ", mean)
        print("Std: ", std)
    return mean, std
