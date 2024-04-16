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
#
# Copyright (c) 2019-* Diganta Misra. All Rights Reserved. MIT Licensed.
#                      [orig. paper: https://www.bmvc2020-conference.com/assets/papers/0928.pdf;
#                       orig. code: https://github.com/digantamisra98/Mish ;
#                       license text: https://github.com/digantamisra98/Mish/blob/master/LICENSE]
#
# ==============================================================================
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0
# IMPORTS
from typing import Union

from torch import nn
from torch import Tensor

from .functional import field_transform as ffield_transform


__all__ = ["FieldTransform"]


class FieldTransform(nn.Module):
    def __init__(
        self,
        pre_sum: Union[float, Tensor] = 0.0,
        mult_div: Union[float, Tensor] = 1.0,
        post_sum: Union[float, Tensor] = 0.0,
        div_not_mul: bool = False,
    ):
        super().__init__()
        self.pre_sum: Union[float, Tensor] = pre_sum
        self.mult_div: Union[float, Tensor] = mult_div
        self.post_sum: Union[float, Tensor] = post_sum
        self.div_not_mul: bool = div_not_mul

    def forward(self, x_input: Tensor) -> Tensor:
        return ffield_transform(
            x_input=x_input,
            pre_sum=self.pre_sum,
            mult_div=self.mult_div,
            post_sum=self.post_sum,
            div_not_mul=self.div_not_mul,
        )
