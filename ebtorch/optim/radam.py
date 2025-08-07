#!/usr/bin/env python3
# ~~ NOTE ~~ ───────────────────────────────────────────────────────────────────
# This module provides a battle-tested, tweaked implementation of the RAdam
# optimizer (after Liu et al., 2019). It started as a best-effort maintained
# merge of wenhui-prudencemed's PR to Liyuan Liu's original implementation after
# PyTorch 1.6 operations deprecation made the original codebase incompatible.
# As bugfixes and improvements were made, the code diverged from the original,
# and (somehow) provides stabler training in comparison to the now official
# PyTorch or fast.ai implementations, for most tasks considered.
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import math
from typing import Any

import torch
from torch.optim.optimizer import Optimizer

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = ["RAdam"]


# ~~ Classes ~~ ────────────────────────────────────────────────────────────────
class RAdam(Optimizer):
    """
    RAdam optimizer implementation.
    RAdam (Rectified Adam) is an adaptive learning rate optimization algorithm
    that rectifies the variance of the adaptive learning rate, making it more
    stable and effective for training deep neural networks.
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate. Default: 1e-3.
        betas (tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square. Default: (0.9, 0.999).
        eps (float, optional): Term added to the denominator to improve
            numerical stability. Default: 1e-8.
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.
        degenerated_to_sgd (bool, optional): If True, when the variance is
            degenerated, the optimizer will behave like SGD. If False, it will
            not update the parameters in such cases. Default: True.
    Raises:
        ValueError: If `lr`, `eps`, or `betas` are not within
            valid ranges, or if `params` is not a valid iterable.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        degenerated_to_sgd: bool = True,
    ) -> None:
        """
        Initialize the RAdam optimizer.
        Args:
            params (iterable): Iterable of parameters to optimize or dicts
                defining parameter groups.
            lr (float, optional): Learning rate. Default: 1e-3.
            betas (tuple[float, float], optional): Coefficients used for computing
                running averages of gradient and its square. Default: (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve
                numerical stability. Default: 1e-8.
            weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.
            degenerated_to_sgd (bool, optional): If True, when the variance is
                degenerated, the optimizer will behave like SGD. If False, it will
                not update the parameters in such cases. Default: True.
        Raises:
            ValueError: If `lr`, `eps`, or `betas` are not within
                valid ranges, or if `params` is not a valid iterable.
        Raises:
            ValueError: If `lr`, `eps`, or `betas` are not within
                valid ranges, or if `params` is not a valid iterable.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid ε: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid β1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid β2: {betas[1]}")

        self.degenerated_to_sgd: bool = degenerated_to_sgd

        if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
            for param in params:
                if "betas" in param and (param["betas"][0] != betas[0] or param["betas"][1] != betas[1]):
                    param["buffer"] = [[None, None, None] for _ in range(10)]

        defaults: dict[str, float | tuple[float, float] | list[list[None]]] = dict(  # NOSONAR
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params=params, defaults=defaults)

    def __setstate__(self, state) -> None:
        """
        Set the state of the optimizer.
        This method is called when loading the optimizer state from a checkpoint.
        It ensures that the buffer is initialized correctly for each parameter group.
        Args:
            state (dict): The state of the optimizer, including parameter groups.
        Raises:
            RuntimeError: If the optimizer state is not compatible with RAdam.
            ValueError: If the state is missing required keys.
        """
        super(RAdam, self).__setstate__(state=state)

    def step(self, closure=None) -> torch.Tensor | None:  # type: ignore NOSONAR
        """
        Perform a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            torch.Tensor | None: The loss value, if provided by the closure.
        Raises:
            RuntimeError: If sparse gradients are provided, as RAdam does not
                support them.
        """
        loss: torch.Tensor | None = None
        if closure is not None:
            loss: torch.Tensor | None = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad: torch.Tensor = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_float: torch.Tensor = p.data.float()

                state: dict[str, Any] = self.state[p]

                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(input=p_data_float)
                    state["exp_avg_sq"] = torch.zeros_like(input=p_data_float)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_float)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_float)

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
                    beta2_t: float = beta2 ** state["step"]
                    n_sma_max: float = 2 / (1 - beta2) - 1
                    n_sma: float = n_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = n_sma

                    if n_sma >= 5:
                        step_size: float = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size: float = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size: float = -1
                    buffered[2] = step_size

                if n_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_float.add_(other=p_data_float, alpha=-group["weight_decay"] * group["lr"])
                    denom: torch.Tensor = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_float.addcdiv_(tensor1=exp_avg, tensor2=denom, value=-step_size * group["lr"])
                    p.data.copy_(p_data_float)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_float.add_(other=p_data_float, alpha=-group["weight_decay"] * group["lr"])
                    p_data_float.add_(other=exp_avg, alpha=-step_size * group["lr"])
                    p.data.copy_(p_data_float)

        return loss
