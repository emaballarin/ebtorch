#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#
# Copyright (c) 2017-* The PyTorch Contributors. All Rights Reserved.
#                      BSD-3 licensed.
#                      [orig. code: https://github.com/pytorch/examples/blob/master/imagenet/main.py]
#
# ──────────────────────────────────────────────────────────────────────────────
#
#  Copyright (c) 2020-2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ──────────────────────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
#
# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS:
from typing import Union

__all__ = ["AverageMeter"]

# TYPE HINTS:
realnum = Union[float, int]


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name: str = name
        self.fmt = fmt
        self.val: realnum = 0
        self.avg: realnum = 0
        self.sum: realnum = 0
        self.count: int = 0

    def reset(self):
        self.val: realnum = 0
        self.avg: realnum = 0
        self.sum: realnum = 0
        self.count: int = 0

    def update(self, val: realnum, n: int = 1):
        self.val: realnum = val
        self.sum += val * n
        self.count += n
        self.avg: realnum = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
