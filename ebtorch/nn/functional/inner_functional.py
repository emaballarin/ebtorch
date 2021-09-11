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

import torch  # lgtm [py/import-and-import-from]
from torch import Tensor
from torch import abs as th_abs
from torch import sign as th_sign
from torch import heaviside as th_heaviside
from torch.overrides import has_torch_function_unary, handle_torch_function
import torch.nn.functional as F


# FUNCTIONS


@torch.jit.script
def mish(x_input: Tensor) -> Tensor:
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    if has_torch_function_unary(x_input):
        return handle_torch_function(mish, (x_input,), x_input)
    return x_input * torch.tanh(F.softplus(x_input))


@torch.jit.script
def mishpulse(x_input: Tensor) -> Tensor:
    """
    Applies the mishpulse function element-wise:
    mishpulse(x) = -sign(x) * mish(-abs(x) + 0.6361099463262276) + step(x)
    """
    if has_torch_function_unary(x_input):
        return handle_torch_function(mish, (x_input,), x_input)

    return -th_sign(x_input) * mish(
        -th_abs(x_input) + 0.6361099463262276
    ) + th_heaviside(x_input, values=torch.tensor([0.0]))


@torch.jit.script
def mishpulse_symmy(x_input: Tensor) -> Tensor:
    """
    Applies the mishpulse function, adapted to be y-symmetric, element-wise:
    mishpulse_symmy(x) = -sign(x) * (mish(-abs(x) + 1.127332431855187) - 1)
    """
    if has_torch_function_unary(x_input):
        return handle_torch_function(mish, (x_input,), x_input)

    return -th_sign(x_input) * (mish(-th_abs(x_input) + 1.127332431855187) - 1.0)
