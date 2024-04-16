#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2024   Jiangtao Zhang and Shunyu Liu and Jie Song and
#                      Tongtian Zhu and Zhengqi Xu and Mingli Song
#                      [original work: https://arxiv.org/abs/2306.07684]
#                      All Rights Reserved. Apache-2.0 Licensed.
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
# Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc> [minor edits]
#                      All Rights Reserved. MIT Licensed.
# ==============================================================================
import copy
from typing import List
from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import required  # type: ignore

__all__ = ("Lookaround",)


def _sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    i: int
    for i, param in enumerate(params):
        d_p: Tensor = d_p_list[i] if not maximize else -d_p_list[i]
        if weight_decay != 0:
            d_p: Tensor = d_p.add(param, alpha=weight_decay)
        if momentum != 0:
            buf: Optional[Tensor] = momentum_buffer_list[i]
            if buf is None:
                buf: Tensor = torch.clone(d_p).detach()
                momentum_buffer_list[i]: Tensor = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            if nesterov:
                d_p: Tensor = d_p.add(buf, alpha=momentum)
            else:
                d_p: Tensor = buf
        param.add_(d_p, alpha=-lr)


def _parcheck(lr, required, momentum, weight_decay, nesterov, dampening):
    if lr is not required and lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if momentum < 0.0:
        raise ValueError(f"Invalid momentum value: {momentum}")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    if nesterov and (momentum <= 0 or dampening != 0):
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")


class Lookaround(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        head_num=3,
        frequence=1,
        weight_decay=0,
        nesterov=False,
        maximize=False,
    ):
        _parcheck(lr, required, momentum, weight_decay, nesterov, dampening)
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            head_num=head_num,
            frequence=frequence,
        )

        self.base_w = 0
        self.accu_w = None
        self.net_head = []
        self.step_n = 0
        super(Lookaround, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Lookaround, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self.step_n == 0:
                for i in range(group["head_num"]):
                    self.net_head.append(copy.deepcopy(group["params"]))

            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]
            lr = group["lr"]

            head = self.step_n % group["head_num"]

            m_str = "momentum_buffer_" + str(head)

            r = (self.step_n % (group["frequence"] * group["head_num"])) % group[
                "head_num"
            ]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if m_str not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state[m_str])
            _sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
            )
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                self.state[p][m_str] = momentum_buffer
            for i, p in enumerate(group["params"]):
                self.net_head[r][i].data = group["params"][i].data
            if (self.step_n % (group["frequence"] * group["head_num"])) + 1 == (
                group["frequence"] * group["head_num"]
            ):
                for i, p in enumerate(group["params"]):
                    self.net_head[0][i][:] = (
                        1 / group["head_num"] * self.net_head[0][i][:]
                    )
                    for j in range(1, group["head_num"]):
                        self.net_head[0][i][:] = (
                            self.net_head[0][i][:]
                            + 1 / group["head_num"] * self.net_head[j][i][:]
                        )
                    for j in range(1, group["head_num"]):
                        self.net_head[j][i][:] = self.net_head[0][i][:]
            for i, p in enumerate(group["params"]):
                group["params"][i].data = self.net_head[(r + 1) % group["head_num"]][
                    i
                ].data

        self.step_n += 1
        return loss
