#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2021-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
#                      [maintainance, adaptation, extension]
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
from torch import nn
from torch import Tensor

from .functional import serlu as fserlu


__all__ = ["SERLU"]


class SERLU(nn.Module):
    """
    Applies the SERLU function element-wise,
    defined after [Zhang & Li, 2018]
    """

    def __init__(self, lambd: float = 1.07862, alph: float = 2.90427) -> None:
        super().__init__()
        self._lambd: float
        self._alph: float
        self._lambd, self._alph = lambd, alph

    def forward(self, x_input: Tensor) -> Tensor:
        return fserlu(x_input, lambd=self._lambd, alph=self._alph)
