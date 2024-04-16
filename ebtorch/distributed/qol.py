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
from typing import Union

from torch import distributed as dist
from torch import Tensor

__all__ = ["reduce_accumulate_keepalive"]


def reduce_accumulate_keepalive(
    reduction_tensor: Tensor, accumulator: Union[int, float]
):
    dist.barrier()
    dist.all_reduce(reduction_tensor, op=dist.ReduceOp.SUM)
    dist.barrier()
    accumulator += reduction_tensor.item()
    reduction_tensor.zero_()
    return accumulator
