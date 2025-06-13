#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2025 Emanuele Ballarin <emanuele@ballarin.cc>
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
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

__all__: List[str] = [
    "SinLU",
]


class SinLU(nn.Module):
    def __init__(
        self,
        a: Union[int, float] = 1,
        b: Union[int, float] = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs: Dict[
            str, Union[Optional[torch.device], Optional[torch.dtype]]
        ] = {"device": device, "dtype": dtype}
        super().__init__()
        self.a: Tensor = nn.Parameter(torch.ones(1, **factory_kwargs) * a)
        self.b: Tensor = nn.Parameter(torch.ones(1, **factory_kwargs) * b)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * (x + self.a * torch.sin(self.b * x))
