#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#
# Copyright (c) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ──────────────────────────────────────────────────────────────────────────────
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from typing import Tuple

import torch as th
from torch import nn as thnn

import ebtorch.nn as ebthnn

# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "data_prep_dispatcher_1ch",
    "data_prep_dispatcher_3ch",
]

# ──────────────────────────────────────────────────────────────────────────────


def data_prep_dispatcher_1ch(
    device, post_flatten: bool = True, inverse: bool = False, dataset: str = "mnist"
) -> thnn.Module:
    if dataset == "mnist":
        mean: float = 0.1307
        std: float = 0.3081
    elif dataset == "fashionmnist":
        mean: float = 0.2860
        std: float = 0.3530
    elif dataset == "kmnist":
        mean: float = 0.1918
        std: float = 0.3483
    elif dataset == "tissuemnist":
        mean: float = 0.102
        std: float = 0.08
    elif dataset == "octmnist":
        mean: float = 0.1889
        std: float = 0.1694
    else:
        raise ValueError("Invalid dataset.")
    if post_flatten:
        post_function: thnn.Module = thnn.Flatten()
    else:
        post_function: thnn.Module = thnn.Identity()
    data_prep: thnn.Module = thnn.Sequential(
        ebthnn.FieldTransform(
            pre_sum=(not inverse) * (-mean),
            mult_div=std,
            div_not_mul=not inverse,
            post_sum=inverse * mean,
        ),
        deepcopy(post_function),
    ).to(device)
    return data_prep


def data_prep_dispatcher_3ch(
    device, post_flatten: bool = True, inverse: bool = False, dataset: str = "cifarten"
) -> thnn.Module:
    if dataset == "cifarten":
        means: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
        stds: Tuple[float, float, float] = (0.2471, 0.2435, 0.2616)
    elif dataset == "cifarhundred":
        means: Tuple[float, float, float] = (0.5071, 0.4865, 0.4409)
        stds: Tuple[float, float, float] = (0.2673, 0.2564, 0.2762)
    elif dataset == "imagenet":
        means: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        stds: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    elif dataset == "pathmnist":
        means: Tuple[float, float, float] = (0.7405, 0.533, 0.7058)
        stds: Tuple[float, float, float] = (0.0723, 0.1038, 0.0731)
    elif dataset == "svhn":
        means: Tuple[float, float, float] = (0.5, 0.5, 0.5)
        stds: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    elif dataset == "tinyimagenet":
        means: Tuple[float, float, float] = (0.4802, 0.4481, 0.3975)
        stds: Tuple[float, float, float] = (0.2302, 0.2265, 0.2262)
    else:
        raise ValueError("Invalid dataset.")
    if post_flatten:
        post_function: thnn.Module = thnn.Flatten()
    else:
        post_function: thnn.Module = thnn.Identity()
    data_prep: thnn.Module = thnn.Sequential(
        ebthnn.FieldTransform(
            pre_sum=(not inverse)
            * th.tensor([[[-means[0]]], [[-means[1]]], [[-means[2]]]]).to(device),
            mult_div=th.tensor([[[stds[0]]], [[stds[1]]], [[stds[2]]]]).to(device),
            div_not_mul=not inverse,
            post_sum=inverse
            * th.tensor([[[means[0]]], [[means[1]]], [[means[2]]]]).to(device),
        ),
        deepcopy(post_function),
    ).to(device)
    return data_prep
