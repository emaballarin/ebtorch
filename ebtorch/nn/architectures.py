#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
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
# ==============================================================================
# SPDX-License-Identifier: Apache-2.0

# IMPORTS
import copy
from typing import List, Union, Optional
from torch import nn, Tensor


# CLASSES

# Fully-Connected Block, New version
# Joint work with Davide Roznowicz (https://github.com/DavideRoznowicz)
class FCBlock(nn.Module):
    def __init__(
        self,
        in_sizes: Union[List[int], tuple],
        out_size: int,
        bias: Optional[Union[List[bool], tuple, bool]] = None,
        activation_fx: Optional[Union[nn.ModuleList, nn.Module]] = None,
        dropout: Optional[Union[List[Union[float, bool]], float, bool, tuple]] = None,
        batchnorm: Optional[Union[List[bool], bool, tuple]] = None,
    ) -> None:
        super().__init__()

        self.activation_fx = nn.ModuleList()

        error_uneven_size: str = (
            "The length of lists of arguments must be the same across them."
        )
        error_illegal_dropout: str = "The 'dropout' argument must be either False, a float, or an iterable of floats and/or False."

        # Default cases
        if bias is None:
            bias = [True] * len(in_sizes)
        if dropout is None:
            dropout = [False] * len(in_sizes)
        if batchnorm is None:
            batchnorm = [True] * (len(in_sizes) - 1) + [False]
        if activation_fx is None:
            for _ in range(len(in_sizes) - 1):
                self.activation_fx.append(nn.ReLU())

        # Ergonomics
        if isinstance(bias, bool):
            bias = [bias] * len(in_sizes)
        if isinstance(dropout, bool):
            if not dropout:
                dropout = [False] * len(in_sizes)
            else:
                raise ValueError(error_illegal_dropout)
        elif isinstance(dropout, float) or (
            isinstance(dropout, int) and (dropout in (0, 1))
        ):
            dropout = [dropout] * len(in_sizes)
        elif not isinstance(dropout, list):
            raise ValueError(error_illegal_dropout)

        if isinstance(batchnorm, bool):
            batchnorm = [batchnorm] * len(in_sizes)

        if isinstance(activation_fx, nn.Module) and not isinstance(
            activation_fx, nn.ModuleList
        ):
            for _ in range(len(in_sizes)):
                self.activation_fx.append(copy.deepcopy(activation_fx))

        # Sanitize
        if (
            not len(in_sizes)
            == len(bias)
            == len(self.activation_fx)
            == len(dropout)
            == len(batchnorm)
        ):
            raise ValueError(error_uneven_size)

        # Start with an empty module list
        self.module_battery = nn.ModuleList(modules=None)

        for layer_idx in range(len(in_sizes) - 1):
            self.module_battery.append(
                nn.Linear(
                    in_features=in_sizes[layer_idx],
                    out_features=in_sizes[layer_idx + 1],
                    bias=bias[layer_idx],
                )
            )
            self.module_battery.append(copy.deepcopy(self.activation_fx[layer_idx]))

            if batchnorm[layer_idx]:
                self.module_battery.append(
                    nn.BatchNorm1d(num_features=in_sizes[layer_idx + 1])
                )

            if isinstance(dropout[layer_idx], bool) and dropout[layer_idx]:
                raise ValueError(error_illegal_dropout)
            elif not isinstance(dropout[layer_idx], bool):
                self.module_battery.append(nn.Dropout(p=dropout[layer_idx]))

        self.module_battery.append(
            nn.Linear(in_features=in_sizes[-1], out_features=out_size, bias=bias[-1])
        )
        self.module_battery.append(copy.deepcopy(self.activation_fx[-1]))
        if batchnorm[-1]:
            self.module_battery.append(nn.BatchNorm1d(num_features=out_size))
        if isinstance(dropout[-1], bool) and dropout[-1]:
            raise ValueError(error_illegal_dropout)
        elif not isinstance(dropout[-1], bool):
            self.module_battery.append(nn.Dropout(p=dropout[-1]))

    def forward(self, x: Tensor) -> Tensor:
        for module_idx in range(len(self.module_battery)):
            x = self.module_battery[module_idx](x)
        return x


# Fully-Connected Block, Legacy version
class FCBlockLegacy(nn.Module):
    def __init__(
        self,
        fin: int,
        hsizes: List[int],
        fout: int,
        hactiv,
        oactiv,
        bias: Union[bool, List[bool]] = True,
    ) -> None:
        super().__init__()
        allsizes: List[int] = [fin] + hsizes + [fout]

        # Biases for the linears below
        if not isinstance(bias, list):
            bias = [bias] * (len(allsizes) - 1)
        else:
            if not len(bias) == len(allsizes) - 1:
                raise RuntimeError(
                    "If 'bias' is a list, it must have as many elements as #linears"
                )

        self.linears: nn.ModuleList = nn.ModuleList(
            [
                nn.Linear(allsizes[i], allsizes[i + 1], bias=bias[i])
                for i in range(0, len(allsizes) - 1)
            ]
        )
        self.hactiv = hactiv
        self.oactiv = oactiv

        # Address the "hactiv: list" case
        if (
            hactiv is not None
            and isinstance(hactiv, list)
            and not len(hactiv) == len(self.linears) - 1
        ):
            raise RuntimeError(
                "If 'hactiv' is a list, it must have as many elements as (#linears - 1)"
            )

    def forward(self, x: Tensor) -> Tensor:
        idx: int
        linear: nn.Module
        for idx, linear in enumerate(self.linears):
            x: Tensor = linear(x)
            if self.hactiv is not None and idx < len(self.linears) - 1:
                if not isinstance(self.hactiv, list):
                    x: Tensor = self.hactiv(x)
                else:
                    x: Tensor = self.hactiv[idx](x)
        if self.oactiv is not None:
            x: Tensor = self.oactiv(x)
        return x


# Causal Convolutional Layer, 1D
# (cfr.: https://github.com/pytorch/pytorch/issues/1333#issuecomment-400338207)
# TODO: possibly tidy-up the code!


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        result = super().forward(x)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result
