#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# Copyright 2020 Deepmind Technologies Limited.
# Copyright 2023 Cui Jiequan (崔洁全). Minor edits.
# Copyright 2023 Emanuele Ballarin <emanuele@ballarin.cc>. Minor edits.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ──────────────────────────────────────────────────────────────────────────────
"""WideResNet and PreActResNet implementations in PyTorch. From DeepMind's original."""
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "WideResNet",
    "PreActResNet",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "SVHN_MEAN",
    "SVHN_STD",
    "TINY_MEAN",
    "TINY_STD",
]

# ──────────────────────────────────────────────────────────────────────────────

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN: Tuple[float, float, float] = (0.5071, 0.4865, 0.4409)
CIFAR100_STD: Tuple[float, float, float] = (0.2673, 0.2564, 0.2762)
SVHN_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
SVHN_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)
TINY_MEAN: Tuple[float, float, float] = (0.4802, 0.4481, 0.3975)
TINY_STD: Tuple[float, float, float] = (0.2302, 0.2265, 0.2262)

# ──────────────────────────────────────────────────────────────────────────────


def _auto_pad_to_same(argself: nn.Module, argx: torch.Tensor) -> torch.Tensor:
    """Auto-pad input `argx` to obtain the same effect as `padding='same'`."""
    if argself.padding > 0:
        return F.pad(argx, (argself.padding,) * 4)
    else:
        return argx


def _auto_cuda_mean_handler(argself: nn.Module, argx: torch.Tensor) -> torch.Tensor:
    """Automatically handle mean / cuda_mean duplication and synchronization."""
    if argx.is_cuda:
        if argself.mean_cuda is None:
            argself.mean_cuda = argself.mean.cuda()
            argself.std_cuda = argself.std.cuda()
        return (argx - argself.mean_cuda) / argself.std_cuda
    else:
        return (argx - argself.mean) / argself.std


# ──────────────────────────────────────────────────────────────────────────────


class _Block(nn.Module):
    """WideResNet Block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        activation_fn: nn.Module = nn.SiLU,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.batchnorm_0: nn.Module = nn.BatchNorm2d(in_planes, momentum=bn_momentum)
        self.relu_0: nn.Module = activation_fn()
        self.conv_0: nn.Module = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False
        )
        self.batchnorm_1: nn.Module = nn.BatchNorm2d(out_planes, momentum=bn_momentum)
        self.relu_1: nn.Module = activation_fn()
        self.conv_1: nn.Module = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.has_shortcut: bool = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut: Optional[nn.Module] = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut: Optional[nn.Module] = None
        self._stride: int = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_shortcut:
            x: torch.Tensor = self.relu_0(self.batchnorm_0(x))
            v: torch.Tensor = x
        else:
            out: torch.Tensor = self.relu_0(self.batchnorm_0(x))
            v: torch.Tensor = out

        if self._stride == 1:
            v = F.pad(v, pad=(1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, pad=(0, 1, 0, 1))
        else:
            raise ValueError("Unsupported `stride != 1` or `stride != 2`.")

        out: torch.Tensor = self.conv_0(v)
        out: torch.Tensor = self.relu_1(self.batchnorm_1(out))
        out: torch.Tensor = self.conv_1(out)
        out: torch.Tensor = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _BlockGroup(nn.Module):
    """WideResNet block group."""

    def __init__(
        self,
        num_blocks: int,
        in_planes: int,
        out_planes: int,
        stride: int,
        activation_fn: nn.Module = nn.SiLU,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        block: nn.ModuleList = nn.ModuleList()
        for i in range(num_blocks):
            block.append(
                _Block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    activation_fn=activation_fn,
                    bn_momentum=bn_momentum,
                )
            )
        self.block: nn.Module = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _PreActBlock(nn.Module):
    """Pre-activation ResNet Block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        activation_fn: nn.Module = nn.SiLU,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self._stride: int = stride
        self.batchnorm_0: nn.Module = nn.BatchNorm2d(in_planes, momentum=bn_momentum)
        self.relu_0: nn.Module = activation_fn()
        self.conv_2d_1: nn.Module = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False
        )
        self.batchnorm_1: nn.Module = nn.BatchNorm2d(out_planes, momentum=bn_momentum)
        self.relu_1: nn.Module = activation_fn()
        self.conv_2d_2: nn.Module = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.has_shortcut: bool = stride != 1 or in_planes != out_planes
        if self.has_shortcut:
            self.shortcut: nn.Module = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=0,
                bias=False,
            )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self._stride == 1:
            x: torch.Tensor = F.pad(x, pad=(1, 1, 1, 1))
        elif self._stride == 2:
            x: torch.Tensor = F.pad(x, pad=(0, 1, 0, 1))
        else:
            raise ValueError("Unsupported `stride != 1` or `stride != 2`.")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.relu_0(self.batchnorm_0(x))
        shortcut: torch.Tensor = self.shortcut(self._pad(x)) if self.has_shortcut else x
        out: torch.Tensor = self.conv_2d_1(self._pad(out))
        out: torch.Tensor = self.conv_2d_2(self.relu_1(self.batchnorm_1(out)))
        return out + shortcut


# ──────────────────────────────────────────────────────────────────────────────


class WideResNet(nn.Module):
    """WideResNet."""

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        width: int = 10,
        activation_fn: nn.Module = nn.SiLU,
        mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
        std: Union[Tuple[float, ...], float] = CIFAR10_STD,
        padding: int = 0,
        num_input_channels: int = 3,
        bn_momentum: float = 0.1,
        autopool: bool = False,
    ) -> None:
        super().__init__()
        self.mean: torch.Tensor = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std: torch.Tensor = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda: Optional[torch.Tensor] = None
        self.std_cuda: Optional[torch.Tensor] = None
        self.padding: int = padding
        num_channels: List[int] = [
            16,
            16 * width,
            32 * width,
            64 * width,
        ]

        if (depth - 4) % 6 != 0:
            raise ValueError("Unsupported `depth != 6 * n + 4` for integer `n`.")

        num_blocks: int = (depth - 4) // 6
        self.init_conv: nn.Module = nn.Conv2d(
            num_input_channels,
            num_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.layer: nn.Module = nn.Sequential(
            _BlockGroup(
                num_blocks,
                num_channels[0],
                num_channels[1],
                stride=1,
                activation_fn=activation_fn,
                bn_momentum=bn_momentum,
            ),
            _BlockGroup(
                num_blocks,
                num_channels[1],
                num_channels[2],
                stride=2,
                activation_fn=activation_fn,
                bn_momentum=bn_momentum,
            ),
            _BlockGroup(
                num_blocks,
                num_channels[2],
                num_channels[3],
                stride=2,
                activation_fn=activation_fn,
                bn_momentum=bn_momentum,
            ),
        )
        self.batchnorm: nn.Module = nn.BatchNorm2d(
            num_channels[3], momentum=bn_momentum
        )
        self.relu: nn.Module = activation_fn()
        self.logits: nn.Module = nn.Linear(num_channels[3], num_classes)
        self.num_channels: int = num_channels[3]
        self.pooling: nn.Module = (
            nn.AdaptiveAvgPool2d((1, 1)) if autopool else nn.AvgPool2d(8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = _auto_pad_to_same(self, x)
        out: torch.Tensor = _auto_cuda_mean_handler(self, x)
        out: torch.Tensor = self.init_conv(out)
        out: torch.Tensor = self.layer(out)
        out: torch.Tensor = self.relu(self.batchnorm(out))
        out: torch.Tensor = self.pooling(out)
        out: torch.Tensor = out.view(-1, self.num_channels)
        return self.logits(out)


class PreActResNet(nn.Module):
    """Pre-activation ResNet."""

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 18,
        width: int = 0,  # Used to make the constructor consistent.
        activation_fn: nn.Module = nn.SiLU,
        mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
        std: Union[Tuple[float, ...], float] = CIFAR10_STD,
        padding: int = 0,
        num_input_channels: int = 3,
        bn_momentum: float = 0.1,
        autopool: bool = False,
    ) -> None:
        super().__init__()
        if width != 0:
            raise ValueError("Unsupported `width = 0`.")
        self.mean: torch.Tensor = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std: torch.Tensor = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda: Optional[torch.Tensor] = None
        self.std_cuda: Optional[torch.Tensor] = None
        self.padding: int = padding
        self.conv_2d: nn.Module = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        if depth == 18:
            num_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2)
        elif depth == 34:
            num_blocks: Tuple[int, int, int, int] = (3, 4, 6, 3)
        else:
            raise ValueError("Unsupported `depth != 18` or `depth != 34`.")
        self.layer_0: nn.Module = self._make_layer(
            in_planes=64,
            out_planes=64,
            num_blocks=num_blocks[0],
            stride=1,
            activation_fn=activation_fn,
            bn_momentum=bn_momentum,
        )
        self.layer_1: nn.Module = self._make_layer(
            in_planes=64,
            out_planes=128,
            num_blocks=num_blocks[1],
            stride=2,
            activation_fn=activation_fn,
            bn_momentum=bn_momentum,
        )
        self.layer_2: nn.Module = self._make_layer(
            in_planes=128,
            out_planes=256,
            num_blocks=num_blocks[2],
            stride=2,
            activation_fn=activation_fn,
            bn_momentum=bn_momentum,
        )
        self.layer_3: nn.Module = self._make_layer(
            in_planes=256,
            out_planes=512,
            num_blocks=num_blocks[3],
            stride=2,
            activation_fn=activation_fn,
            bn_momentum=bn_momentum,
        )
        self.batchnorm: nn.Module = nn.BatchNorm2d(
            num_features=512, momentum=bn_momentum
        )
        self.relu: nn.Module = activation_fn()
        self.logits: nn.Module = nn.Linear(in_features=512, out_features=num_classes)
        self.pooling: nn.Module = (
            nn.AdaptiveAvgPool2d((1, 1)) if autopool else nn.AvgPool2d(4)
        )

    def _make_layer(  # Do not make static.
        self,
        in_planes: int,
        out_planes: int,
        num_blocks: int,
        stride: int,
        activation_fn: nn.Module,
        bn_momentum: float = 0.1,
    ) -> nn.Module:
        layers: nn.ModuleList = nn.ModuleList()
        for i, stride in enumerate([stride] + [1] * (num_blocks - 1)):
            layers.append(
                _PreActBlock(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    stride,
                    activation_fn,
                    bn_momentum=bn_momentum,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = _auto_pad_to_same(self, x)
        out: torch.Tensor = _auto_cuda_mean_handler(self, x)
        out: torch.Tensor = self.conv_2d(out)
        out: torch.Tensor = self.layer_0(out)
        out: torch.Tensor = self.layer_1(out)
        out: torch.Tensor = self.layer_2(out)
        out: torch.Tensor = self.layer_3(out)
        out: torch.Tensor = self.relu(self.batchnorm(out))
        out: torch.Tensor = self.pooling(out)
        out: torch.Tensor = out.view(out.size(0), -1)
        return self.logits(out)
