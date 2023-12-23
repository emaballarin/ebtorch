#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
# All Rights Reserved. Unless otherwise explicitly stated.
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
# ==============================================================================
#
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn

__all__ = [
    "extract_conv_filters",
    "show_filters",
]


def extract_conv_filters(layer: nn.Module) -> torch.Tensor:
    if isinstance(layer, nn.Conv1d):
        conv_dim: int = 1
    elif isinstance(layer, nn.Conv2d):
        conv_dim: int = 2
    else:
        raise ValueError("Layer must be a 1D or 2D convolutional layer!")

    weights = layer.weight.clone().detach()

    if conv_dim == 1:
        weights = weights.reshape(weights.shape[0], -1, weights.shape[-1])
    else:  # conv_dim == 2:
        weights = weights.reshape(
            weights.shape[0], -1, weights.shape[-2], weights.shape[-1]
        )

    return weights


def show_filters(filters: torch.Tensor) -> None:
    plt.axis("off")
    plt.imshow(
        torchvision.utils.make_grid(filters, padding=1, normalize=True)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
