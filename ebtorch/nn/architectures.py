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
from typing import List, Union
from torch import nn, Tensor


# CLASSES

# Fully-Connected Block
class FCBlock(nn.Module):
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
