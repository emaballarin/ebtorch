#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ──────────────────────────────────────────────────────────────────────────────
from typing import List
from typing import Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

# ──────────────────────────────────────────────────────────────────────────────

__all__ = ["SophiaG"]

# ──────────────────────────────────────────────────────────────────────────────


class SophiaG(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=1e-1,
        *,
        maximize: bool = False,
        capturable: bool = False,
    ):
        # Validate parameters
        if lr < 0:
            raise ValueError(f"Learning rate must be positive. Got: {lr}.")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Beta[0] must be in [0, 1). Got: {betas[0]}.")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Beta[1] must be in [0, 1). Got: {betas[1]}.")
        if rho < 0:
            raise ValueError(f"Rho must be non-negative. Got: {rho}.")
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative. Got: {weight_decay}.")
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # noinspection DuplicatedCode
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError("SophiaG does not support sparse gradients.")
                grads.append(p.grad)
                # noinspection DuplicatedCode
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])
                hessian.append(state["hessian"])

                if self.defaults["capturable"]:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(
                params_with_grad,
                grads,
                exp_avgs,
                hessian,
                state_steps,
                bs=bs,
                beta1=beta1,
                beta2=beta2,
                rho=group["rho"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                capturable=group["capturable"],
            )

        return loss


def sophiag(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    capturable: bool = False,
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("`state_steps` must contain a list of singleton tensors.")

    func = _single_tensor_sophiag

    func(
        params,
        grads,
        exp_avgs,
        hessian,
        state_steps,
        bs=bs,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        capturable=capturable,
    )


def _single_tensor_sophiag(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    *,
    bs: Union[int, Tensor],
    beta1: float,
    beta2: float,
    rho: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    maximize: bool,
    capturable: bool,
):
    _ = beta2  # unused, for uniformity in signature
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        # Capturable: asserts
        assert not capturable or (
            all(isinstance(iiv, Tensor) for iiv in (bs, lr))
            and all(icv.is_cuda for icv in (bs, param, step_t))
        )

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        # noinspection PyUnresolvedReferences
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        # Compute the (negative) step size
        step_size_neg = lr.neg() if capturable else -lr

        ratio = rho * bs * hess + 1e-15
        torch.div(exp_avg, ratio, out=ratio)
        torch.clamp_(ratio, -1, 1)
        param.add_(ratio, alpha=step_size_neg)
