#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2020-* Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen,
#                      Xiaodong Liu, Jianfeng Gao, Jiawei Han. All Rights Reserved.
#                      [orig. work: https://arxiv.org/abs/1908.03265;
#                       orig. code: https://github.com/LiyuanLucasLiu/RAdam]
#
# Copyright (c) 2020-* @wenhui-prudencemed (GitHub user). All Rights Reserved.
#                      [orig. code: https://github.com/wenhui-prudencemed/RAdam]
#
# Copyright (c) 2020-2025 Emanuele Ballarin <emanuele@ballarin.cc>
#                         All Rights Reserved.
#                         [maintainance, adaptation, extension]
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
import math

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["RAdam"]


class RAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state["step"] += 1
                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = n_sma

                    # more conservative since it's an approximated value
                    if n_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if n_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                        )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                        )
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group["lr"])
                    p.data.copy_(p_data_fp32)

        return loss
