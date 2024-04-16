#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#
#  Copyright (c) 2020-2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ──────────────────────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: MIT
#
# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
from torch import nn
from torch import Tensor

from .functional import smelu as fsmelu


__all__ = ["SmeLU"]


class SmeLU(nn.Module):
    """
    Applies the SmeLU function element-wise,
    defined after [Shamir & Ling, 2022]
    """

    def __init__(self, beta: float = 2.0) -> None:
        super().__init__()
        self._beta: float = beta

    def forward(self, x_input: Tensor) -> Tensor:
        return fsmelu(x_input, beta=self._beta)
