#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright (c) 2020-* Prem Seetharaman, Gordon Wichern, Bryan Pardo,
#                      Jonathan Le Roux. All Rights Reserved. MIT Licensed.
#                      [orig. work: https://arxiv.org/abs/2007.14469;
#                       orig. code: https://github.com/pseeth/autoclip ;
#                       license text: https://github.com/pseeth/autoclip/blob/master/LICENSE ]
#
# ==============================================================================
#
# Copyright (c) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
#                      [maintainance, adaptation, extension]
#
# ==============================================================================
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Optional, List
import torch.nn.utils as thutils
import numpy as np

realnum = Union[float, int]


def _get_grad_norm(model) -> realnum:
    total_norm: realnum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm: realnum = total_norm ** (1.0 / 2)
    return total_norm


class AutoClipper:
    __constants__: List[str] = ["queue_size"]
    queue_size: Optional[int]

    def __init__(self, queue_size: Optional[int] = None) -> None:
        self.queue_size: Optional[int] = int(queue_size)
        self.queue_list: List[float] = []

    def grad_consider(self, model) -> None:
        if self.queue_size is not None and len(self.queue_list) >= self.queue_size:
            _ = self.queue_list.pop()
        self.queue_list.insert(0, _get_grad_norm(model))

    def grad_autoclip_(self, model, clip_percentile) -> None:
        self.grad_consider(model)
        thutils.clip_grad_norm_(
            model.parameters(), np.percentile(self.queue_list, clip_percentile)
        )

    def reset(self) -> None:
        self.queue_list = []

    def resize(self, queue_size: Optional[int] = None):
        self.queue_size = queue_size
