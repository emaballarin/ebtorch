#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#
#  Copyright (c) 2020-2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ──────────────────────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: MIT
#
# ──────────────────────────────────────────────────────────────────────────────
from .cutmixup import FastCollateMixup
from .cutmixup import Mixup
from .datasets import cifarhundred_dataloader_dispatcher
from .datasets import cifarten_dataloader_dispatcher
from .datasets import fashionmnist_dataloader_dispatcher
from .datasets import imagenette_dataloader_dispatcher
from .datasets import kmnist_dataloader_dispatcher
from .datasets import mnist_dataloader_dispatcher
from .datasets import octmnist_dataloader_dispatcher
from .datasets import pathmnist_dataloader_dispatcher
from .datasets import tinyimagenet_dataloader_dispatcher
from .datasets import tissuemnist_dataloader_dispatcher
from .prep import data_prep_dispatcher_1ch
from .prep import data_prep_dispatcher_3ch

# ──────────────────────────────────────────────────────────────────────────────

del cutmixup
del datasets
del prep
del tinyimagenet
