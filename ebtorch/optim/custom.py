#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
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
# ==============================================================================
#
# SPDX-License-Identifier: Apache-2.0
#
from collections.abc import Iterable
from typing import Tuple
from typing import Union

import torch as th
import torch.optim
from torch import Tensor

from .lookahead import Lookahead
from .radam import RAdam

# ==============================================================================


def ralah_optim(
    parameters: Union[Iterable[Tensor], Iterable[dict]],
    radam_lr: float = 1e-3,
    la_steps: int = 5,
    la_alpha: float = 0.8,
    radam_betas: Tuple[float, float] = (0.9, 0.999),
    radam_eps: float = 1e-8,
    radam_wd: float = 0.0,
    radam_degenerate_to_sgd: bool = True,
    la_pullback_momentum: str = "none",
):
    """RAdam + Lookahead optimizer"""
    return Lookahead(
        RAdam(
            params=parameters,
            lr=radam_lr,
            betas=radam_betas,
            eps=radam_eps,
            weight_decay=radam_wd,
            degenerated_to_sgd=radam_degenerate_to_sgd,
        ),
        la_steps=la_steps,
        la_alpha=la_alpha,
        pullback_momentum=la_pullback_momentum,
    )


# ==============================================================================


def wfneal(
    optim: torch.optim.Optimizer,
    lr: float,
    epochs: int,
    magic_fraction: float = 0.56,
    verbose: bool = False,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    # Durations
    steady_epochs: int = int((epochs - 2) * magic_fraction)
    anneal_epochs: int = epochs - 2 - steady_epochs

    # Seethrough early stopping
    stes_epoch: int = epochs - max(20, int(anneal_epochs / 4)) - 1

    # Prepare optim
    for grp in optim.param_groups:
        grp["lr"] = lr

    # Schedulers
    warmup_scheduler = th.optim.lr_scheduler.LinearLR(
        optimizer=optim,
        start_factor=0.5,
        end_factor=1.0,
        total_iters=2,
        last_epoch=-1,
        verbose=verbose,
    )
    steady_scheduler = th.optim.lr_scheduler.ConstantLR(
        optimizer=optim,
        factor=1.0,
        total_iters=steady_epochs,
        last_epoch=-1,
        verbose=verbose,
    )
    anneal_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=anneal_epochs,
        eta_min=lr * 1e-4,
        last_epoch=-1,
        verbose=verbose,
    )

    # Prepare scheduler
    sched = th.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_scheduler, steady_scheduler, anneal_scheduler],
        milestones=[2, 2 + steady_epochs],
        last_epoch=-1,
        verbose=verbose,
    )

    # Return
    return optim, sched, stes_epoch
