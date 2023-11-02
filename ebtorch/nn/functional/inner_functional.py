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
import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "field_transform",
    "mish",
    "serlu",
    "smelu",
    "serf",
]


# FUNCTIONS


def field_transform(
    x_input: Tensor,
    pre_sum: float = 0.0,
    mult_div: float = 1.0,
    post_sum: float = 0.0,
    div_not_mul: bool = False,
) -> Tensor:
    if div_not_mul:
        return torch.add(
            input=torch.div(
                input=torch.add(input=x_input, other=pre_sum), other=mult_div
            ),
            other=post_sum,
        )
    else:
        return torch.add(
            input=torch.mul(
                input=torch.add(input=x_input, other=pre_sum), other=mult_div
            ),
            other=post_sum,
        )


@torch.jit.script
def mish(x_input: Tensor) -> Tensor:
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    return F.mish(x_input)


@torch.jit.script
def serlu(x_input: Tensor, lambd: float = 1.07862, alph: float = 2.90427) -> Tensor:
    """
    Applies the SERLU function element-wise,
    defined after [Zhang & Li, 2018]
    """
    return torch.where(  # type: ignore
        x_input >= 0.0,
        torch.mul(x_input, lambd),
        torch.mul(torch.mul(x_input, torch.exp(x_input)), lambd * alph),
    )


@torch.jit.script
def smelu(x_input: Tensor, beta: float = 2.0) -> Tensor:
    """
    Applies the SmeLU function element-wise,
    defined after [Shamir & Ling, 2022]
    """
    assert beta >= 0
    return torch.where(  # type: ignore
        torch.abs(x_input) <= beta,
        torch.div(torch.pow(torch.add(x_input, beta), 2), 4.0 * beta),
        F.relu(x_input),
    )


@torch.jit.script
def serf(x: Tensor) -> Tensor:
    """Applies the Scaled ERror Function, element-wise."""
    return torch.erf(x / math.sqrt(2.0))  # type: ignore
