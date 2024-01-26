#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ──────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2020-* Hugging Face, Inc.
#                      Ross Wightman
#                      All Rights Reserved.
#                      [orig. code: https://github.com/huggingface/pytorch-image-models ;
#                      license text: https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE]
#
# Copyright (c) 2023-* Emanuele Ballarin <emanuele@ballarin.cc> (minor edits)
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ──────────────────────────────────────────────────────────────────────────────
#
import abc
import math
import warnings
from typing import Any
from typing import Dict
from typing import Optional

import torch

__all__ = ["CosineLRScheduler"]


class _Scheduler(abc.ABC):
    """Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    The schedulers built on this should try to remain as stateless as possible (for simplicity).

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.

    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        param_group_field: str,
        t_in_epochs: bool = True,
        noise_range_t=None,
        noise_type="normal",
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=None,
        initialize: bool = True,
    ) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(
                        f"{param_group_field} missing from param_groups[{i}]"
                    )
                group.setdefault(
                    self._initial_param_group_field, group[param_group_field]
                )
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(
                        f"{self._initial_param_group_field} missing from param_groups[{i}]"
                    )
        self.base_values = [
            group[self._initial_param_group_field]
            for group in self.optimizer.param_groups
        ]
        self.metric = None  # any point to having this for all?
        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    @abc.abstractmethod
    def _get_lr(self, t: int) -> float:
        pass

    def _get_values(self, t: int, on_epoch: bool = True) -> Optional[float]:
        proceed = (on_epoch and self.t_in_epochs) or (
            not on_epoch and not self.t_in_epochs
        )
        if not proceed:
            return None
        return self._get_lr(t)

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self._get_values(epoch, on_epoch=True)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self._get_values(num_updates, on_epoch=False)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            if "lr_scale" in param_group:
                param_group[self.param_group_field] = value * param_group["lr_scale"]
            else:
                param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self._is_apply_noise(t):
            noise = self._calculate_noise(t)
            lrs = [v + v * noise for v in lrs]
        return lrs

    def _is_apply_noise(self, t) -> bool:
        """Return True if scheduler in noise range."""
        apply_noise = False
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
        return apply_noise

    def _calculate_noise(self, t) -> float:
        g = torch.Generator()
        g.manual_seed(self.noise_seed + t)
        if self.noise_type == "normal":
            while True:
                # resample if noise out of percent limit; it's brute force but shouldn't spin much
                noise = torch.randn(1, generator=g).item()
                if abs(noise) < self.noise_pct:
                    return noise
        else:
            noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
        return noise


class CosineLRScheduler(_Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """

    def __init__(
        self,  # NOSONAR
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min: float = 0.0,
        cycle_mul: float = 1.0,
        cycle_decay: float = 1.0,
        cycle_limit: int = 1,
        warmup_t=0,
        warmup_lr_init=0,
        warmup_prefix=False,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        k_decay=1.0,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        if t_initial <= 0:
            raise ValueError(f"Expected positive `t_initial`, got {t_initial}")
        if lr_min < 0:
            raise ValueError(f"Expected positive `lr_min`, got {lr_min}")
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            warnings.warn(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1."
            )
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(
                    math.log(
                        1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul
                    )
                )
                t_i = self.cycle_mul**i * self.t_initial
                t_curr = (
                    t
                    - (1 - self.cycle_mul**i) / (1 - self.cycle_mul) * self.t_initial
                )
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay**i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min
                    + 0.5
                    * (lr_max - self.lr_min)
                    * (1 + math.cos(math.pi * t_curr**k / t_i**k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(
                math.floor(
                    -self.t_initial
                    * (self.cycle_mul**cycles - 1)
                    / (1 - self.cycle_mul)
                )
            )
