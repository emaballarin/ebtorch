#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2019 Google Research (main, core work; original source)
#           2019 Liangchen Luo   (main, core work; original source)
#           2023 Nikolay Novik   (minor edits; refactoring)
#           2023 Emanuele Ballarin <emanuele@ballarin.cc> (minor edits; maintainance)
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
# ==============================================================================
import math
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


__all__ = ("AdaBound",)

msg_error_sparse_grad: str = (
    "AdaBound does not support sparse gradients, please consider SparseAdam instead"
)


class AdaBound(Optimizer):
    r"""Implements the AdaBound(W) algorithm (with optional Decoupled Weight Decay).

    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of
    Learning Rate`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate (default: 1e-3)
        betas (Tuple[float, float]): coefficients used for computing running averages of gradient and its square
              (default: (0.9, 0.999))
        final_lr (float): final (SGD) learning rate (default: 0.1)
        gamma (float): convergence speed of the bound functions (default: 1e-3)
        eps (float): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        amsbound (bool): whether to use the AMSBound variant of this algorithm
        decouple_wd (bool): whether to decouple the weight decay from the gradient optimization step

    Example:
        >>> from ebtorch import optim
        >>> optimizer = optim.AdaBound(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1902.09843
    """

    def __init__(
        self,
        params: Union[Iterable[Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        final_lr: float = 0.1,
        gamma: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsbound: bool = False,
        decouple_wd: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if final_lr < 0.0:
            raise ValueError(f"Invalid final learning rate: {final_lr}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super(AdaBound, self).__init__(params, defaults)
        self.base_lrs = [group["lr"] for group in self.param_groups]
        self.decouple_wd = decouple_wd

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsbound", False)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(msg_error_sparse_grad)
                amsbound = group["amsbound"]

                state = self.state[p]

                # Dict[str, Any] initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsbound:
                        # Maintains max of all exp. moving avg. of
                        # sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsbound:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if (not self.decouple_wd) and group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running
                    # avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround
                # to apply lr decay
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower_bound = final_lr * (1 - 1 / (group["gamma"] * state["step"] + 1))
                upper_bound = final_lr * (1 + 1 / (group["gamma"] * state["step"]))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                if (not self.decouple_wd) or group["weight_decay"] == 0:
                    p.data.add_(-step_size)
                else:
                    decayed_weights = torch.mul(p.data, group["weight_decay"])
                    p.data.add_(-step_size)
                    p.data.sub_(decayed_weights)
        return loss
