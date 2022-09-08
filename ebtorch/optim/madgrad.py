#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2021-* Facebook, Inc. and its affiliates, Aaron Defazio,
#                      Samy Jelassi. All Rights Reserved. MIT Licensed.
#                      [orig. work: https://arxiv.org/pdf/2101.11075.pdf;
#                       orig. code: https://github.com/facebookresearch/madgrad;
#                       license text: https://github.com/facebookresearch/madgrad/blob/master/LICENSE]
#
# ==============================================================================
#
# Copyright (c) 2021-* Nestor Demeure. All Rights Reserved.
#                      [orig. code: https://github.com/nestordemeure/flaxOptimizers/blob/main/flaxOptimizers/madgrad.py]
#
# Copyright (c) 2021-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved. MIT + Apache 2.0 licensed.
#                      [maintainance, adaptation, extension;
#                       orig. code: https://github.com/emaballarin/madgrad;
#                       orig. license text: https://github.com/emaballarin/madgrad/blob/master/LICENSE]
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
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Any
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

import torch.optim

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class MADGRAD(torch.optim.Optimizer):
    """
    MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
    Optimization.

    .. _MADGRAD: https://arxiv.org/abs/2101.11075

    MADGRAD is a general purpose optimizer that can be used in place of SGD or
    Adam may converge faster and generalize better. Currently GPU-only.
    Typically, the same learning rate schedule that is used for SGD or Adam may
    be used. The overall learning rate is not comparable to either method and
    should be determined by a hyper-parameter sweep.

    MADGRAD requires less weight decay than other methods, often as little as
    zero. Momentum values used for SGD or Adam's beta1 should work here also.

    On sparse problems both weight_decay and momentum should be set to 0.

    With respect to the original implementation [2], AdamW-style weight decay [3],
    momentum-aware AdamW-style weight decay [4], and linearized learning rate [5]
    are also available. Such additions and tweaks have been suggested by
    Nestor Demeure [6] and closely follow his JAX implementation [7].

    Default behaviour (``awwd=False, mawd=False, linlr=False``) reproduces that of
    the original implementation [2], whereas ``awwd=True, mawd=True, linlr=True``
    reproduce that of Nestor Demeure's [6] JAX implementation [7].

    .. [2]: https://github.com/facebookresearch/madgrad
    .. [3]: https://github.com/facebookresearch/madgrad/issues/1#issue-846152670
    .. [4]: https://github.com/facebookresearch/madgrad/issues/1#issuecomment-811696746
    .. [5]: https://github.com/facebookresearch/madgrad/issues/1#issuecomment-811696746
    .. [6]: https://github.com/nestordemeure
    .. [7]: https://github.com/nestordemeure/flaxOptimizers/blob/main/flaxOptimizers/madgrad.py

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in  the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-6).
        awwd (bool):
            Apply stepweight (AdamW-like) weight decay (default: False).
        mawd (bool):
            Make stepweight (AdamW-like) weight decay momentum-aware (default: False).
        linlr (bool):
            Linearize the application of learning rate (default: False).
    """

    def __init__(
        self,
        params: _params_t,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eps: float = 1e-6,
        awwd=False,
        mawd=False,
        linlr=False,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps {eps} must be non-negative")
        if mawd and not awwd:
            raise RuntimeError("MAWD requires AWWD, but AWWD is not enabled")

        defaults = dict(
            lr=lr,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
            awwd=awwd,
            mawd=mawd,
            linlr=linlr,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("awwd", False)
            group.setdefault("mawd", False)
            group.setdefault("linlr", False)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under
        # optimizer sharding
        if "k" not in self.state:
            self.state["k"] = torch.tensor([0], dtype=torch.long)
        k = self.state["k"].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1 - momentum

            # Apply LR linearization or not
            if not group["linlr"]:
                lamb = lr * math.pow(k + 1, 0.5)
            else:
                lamb = math.pow(lr, 1.5) * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                    state["s"] = torch.zeros_like(p.data).detach()
                    if momentum != 0:
                        state["x0"] = torch.clone(p.data).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError(
                        "momentum != 0 is not compatible with sparse gradients"
                    )

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )

                    # Apply stepweight (AdamW-like) decay or not: `not` case
                    if not group["awwd"]:
                        grad.add_(p.data, alpha=decay)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()  # skipcq: PYL-W0212

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_vals = (
                        grad_sum_sq_masked._values()  # skipcq: PYL-W0212
                        .pow(1 / 3)
                        .add_(eps)  # skipcq: PYL-W0212
                    )  # skipcq: PYL-W0212
                    x0_masked_vals = p_masked._values().addcdiv(  # skipcq: PYL-W0212
                        s_masked._values(),  # skipcq: PYL-W0212
                        rms_masked_vals,
                        value=1,  # skipcq: PYL-W0212
                    )

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = (
                        grad_sum_sq_masked._values()  # skipcq: PYL-W0212
                        .pow_(1 / 3)  # skipcq: PYL-W0212
                        .add_(eps)  # skipcq: PYL-W0212
                    )

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)  # skipcq: PYL-W0212

                    # update masked copy of p
                    p_kp1_masked_vals = x0_masked_vals.addcdiv(
                        s_masked._values(),  # skipcq: PYL-W0212
                        rms_masked_vals,
                        value=-1,
                    )
                    # Copy updated masked p to dense p using an add operation
                    p_masked._values().add_(  # skipcq: PYL-W0212
                        p_kp1_masked_vals, alpha=-1
                    )
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    # Update s
                    s.data.add_(grad, alpha=lamb)

                    # Apply stepweight (AdamW-like) decay or not: `yes` case; part 1
                    if group["awwd"]:
                        if group["mawd"]:
                            wd_diff = torch.mul(input=p.data, other=-ck * lr * decay)
                        else:
                            wd_diff = torch.mul(input=p.data, other=-lr * decay)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

                    # Apply stepweight (AdamW-like) decay or not: `yes` case; part 2
                    if group["awwd"]:
                        p.data.add_(wd_diff)

        self.state["k"] += 1
        return loss
