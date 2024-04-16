#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2020-* NAVER Corp.
#                      Byeongho Heo, Sanghyuk Chun, Seong Joon Oh, Dongyoon Han,
#                      Sangdoo Yun, Gyuwan Kim, Youngjung Uh, Jung-Woo Ha
#                      All Rights Reserved. MIT Licensed.
#                      [orig. work: https://arxiv.org/abs/2006.08217;
#                      orig. code: https://github.com/clovaai/AdamP ;
#                      license text: https://github.com/clovaai/AdamP/blob/master/LICENSE]
#
# ──────────────────────────────────────────────────────────────────────────────
# Copyright (c) 2020-* Hugging Face, Inc.
#                      Ross Wightman
#                      All Rights Reserved.
#                      [orig. code: https://github.com/huggingface/pytorch-image-models ;
#                      license text: https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE]
#
# Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc> (minor edits)
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
import math

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import required

__all__ = ["SGDP", "AdamP"]


def _channel_view(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1)


def _layer_view(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(1, -1)


def _projection(p, grad, perturb, delta: float, wd_ratio: float, eps: float):
    wd = 1.0
    expand_size = (-1,) + (1,) * (len(p.shape) - 1)
    for view_func in [_channel_view, _layer_view]:
        param_view = view_func(p)
        grad_view = view_func(grad)
        cosine_sim = F.cosine_similarity(grad_view, param_view, dim=1, eps=eps).abs_()

        # FIXME: this is a problem for PyTorch XLA
        if cosine_sim.max() < delta / math.sqrt(param_view.size(1)):
            p_n = p / param_view.norm(p=2, dim=1).add_(eps).reshape(expand_size)
            perturb -= p_n * view_func(p_n * perturb).sum(dim=1).reshape(expand_size)
            wd = wd_ratio
            return perturb, wd

    return perturb, wd


class SGDP(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        eps=1e-8,
        delta=0.1,
        wd_ratio=0.1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            delta=delta,
            wd_ratio=wd_ratio,
        )
        super(SGDP, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # NOSONAR
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)

                # SGD
                buf = state["momentum"]
                buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)
                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                # Projection
                wd_ratio = 1.0
                if len(p.shape) > 1:
                    d_p, wd_ratio = _projection(
                        p, grad, d_p, group["delta"], group["wd_ratio"], group["eps"]
                    )

                # Weight decay
                if weight_decay != 0:
                    p.mul_(
                        1.0
                        - group["lr"]
                        * group["weight_decay"]
                        * wd_ratio
                        / (1 - momentum)
                    )

                # Step
                p.add_(d_p, alpha=-group["lr"])

        return loss


class AdamP(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        delta=0.1,
        wd_ratio=0.1,
        nesterov=False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
        )
        super(AdamP, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # NOSONAR
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                beta1, beta2 = group["betas"]
                nesterov = group["nesterov"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                # Adam
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1.0
                if len(p.shape) > 1:
                    perturb, wd_ratio = _projection(
                        p,
                        grad,
                        perturb,
                        group["delta"],
                        group["wd_ratio"],
                        group["eps"],
                    )

                # Weight decay
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"] * wd_ratio)

                # Step
                p.add_(perturb, alpha=-step_size)

        return loss
