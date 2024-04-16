#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
#                      [maintainance, adaptation, extension]
#
# ==============================================================================
#
# Copyright (c) 2021-* Ashis Paul. All Rights Reserved.
#                      [orig. paper: https://www.mdpi.com/2227-7390/10/3/337;
#                       orig. code: https://github.com/ashis0013/SinLU]
#
# ==============================================================================
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
from typing import List
from typing import TypeVar

import torch
import torch.nn as nn
from torch import Tensor

SinLUModule = TypeVar("SinLUModule", bound="SinLU")

__all__: List[str] = [
    "SinLU",
]


class SinLU(nn.Module):
    def __init__(self: SinLUModule) -> None:
        super(SinLU, self).__init__()
        self.a: Tensor = nn.Parameter(torch.ones(1))
        self.b: Tensor = nn.Parameter(torch.ones(1))

    def forward(self: SinLUModule, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * (x + self.a * torch.sin(self.b * x))
