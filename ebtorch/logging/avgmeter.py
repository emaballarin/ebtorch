#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2017-* The PyTorch Contributors. All Rights Reserved.
#                      BSD-3 licensed.
#                      [orig. code: https://github.com/pytorch/examples/blob/master/imagenet/main.py]
#
# ==============================================================================
#
# Copyright (c) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
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
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: Apache-2.0
# IMPORTS:
from typing import Union

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
