#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#
# Copyright 2023-* Emanuele Ballarin <emanuele@ballarin.cc>
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
# ──────────────────────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: Apache-2.0
#
# ──────────────────────────────────────────────────────────────────────────────
# Imports
from math import pow as mpow
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────

# Auxiliary Functions


def _bool_one_zero(boolean: bool) -> int:
    return int(boolean)


def _bool_one_minusone(boolean: bool) -> int:
    return 1 - 2 * int(not boolean)


# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    "multilasso",
    "multiridge",
    "beta_gaussian_kldiv",
]
# ──────────────────────────────────────────────────────────────────────────────


def multilasso(
    params: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    p_lasso: float = 1.0,
    p_ridge: float = 2.0,
    lam: float = 0.1,
    alp: float = 1,
    reg_oneminus: bool = False,
    adimensionalize: bool = False,
) -> Tensor:
    # Handle params multi-instance
    if isinstance(params, Tensor):
        params: List[Tensor] = [params]
    elif isinstance(params, tuple):
        params: List[Tensor] = list(params)

    # Preprocess params and decouple lists
    if reg_oneminus:
        params: List[Tensor] = [
            _bool_one_zero(reg_oneminus) + _bool_one_minusone(not reg_oneminus) * param
            for param in params
        ]
    params: List[Tensor] = [param.flatten() for param in params]

    # Compute Lasso penalty
    lpen: Tensor = torch.cat(params).norm(p=p_lasso)

    # Compute Group Lasso penalty
    gpen: Tensor = torch.tensor(
        [
            (param.norm(p=p_ridge) * mpow(param.numel(), (1 - 1 / p_ridge)))
            for param in params
        ],
        device=lpen.device,
    ).norm(p=p_lasso)

    # Eventually adimensionalize
    if adimensionalize:
        divnorm: float = torch.pow(
            torch.tensor([param.numel() for param in params], device=lpen.device).sum(),
            1 / p_lasso,
        ).item()
    else:
        divnorm: int = 1

    # Return Sparse Group Lasso penalty
    return lam * (alp * lpen + (1 - alp) * gpen) / divnorm


# ──────────────────────────────────────────────────────────────────────────────


def multiridge(
    params: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    p_ridge: float = 2.0,
    lam: float = 0.1,
    adimensionalize: bool = False,
) -> Tensor:
    # Handle params multi-instance
    if isinstance(params, Tensor):
        params: List[Tensor] = [params]
    elif isinstance(params, tuple):
        params: List[Tensor] = list(params)

    # Preprocess params
    params: List[Tensor] = [param.flatten() for param in params]

    # Compute penalty
    rpen: Tensor = torch.cat(params).norm(p=p_ridge)

    # Eventually adimensionalize
    if adimensionalize:
        divnorm: float = torch.pow(
            torch.tensor([param.numel() for param in params], device=rpen.device).sum(),
            1 / p_ridge,
        ).item()
    else:
        divnorm: int = 1

    # Return Ridge penalty
    return lam * rpen / divnorm


# ──────────────────────────────────────────────────────────────────────────────


@torch.jit.script
def beta_gaussian_kldiv(mu: Tensor, sigma: Tensor, beta: float = 0.5) -> Tensor:
    kldiv = (torch.pow(mu, 2) + torch.exp(sigma) - sigma - 1).sum()
    return beta * kldiv
