#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ~~ Credits ~~ ────────────────────────────────────────────────────────────────
# Original implementation: https://github.com/iShohei220/adopt.git
# Derivative implementation: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adopt.py
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from collections.abc import Callable
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.optim.optimizer import _disable_dynamo_if_unsupported
from torch.optim.optimizer import _get_scalar_dtype
from torch.optim.optimizer import _get_value
from torch.optim.optimizer import _use_grad_for_differentiable
from torch.optim.optimizer import _view_as_real
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import ParamsT

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: List[str] = ["ADOPT"]

# ~~ Error Messages ~~ ─────────────────────────────────────────────────────────
_nocapture_err: str = (
    "`lr` as a `Tensor` is not supported for `capturable=False` and `foreach=True`"
)


# ~~ ADOPT Optimizer ~~ ────────────────────────────────────────────────────────
class ADOPT(Optimizer):
    """
    ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate: https://arxiv.org/abs/2411.02853
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        clip_exp: Optional[float] = 0.25,
        weight_decay: float = 0.0,
        decouple: bool = False,
        *,
        caution: bool = False,
        foreach: Optional[bool] = False,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(_nocapture_err)
            if lr.numel() != 1:
                raise ValueError("`lr` as `Tensor` must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid `lr`: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid `eps`: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid `betas[0]`: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid `betas[1]`: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid `weight_decay`: {weight_decay}")

        defaults: dict = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clip_exp=clip_exp,
            decouple=decouple,
            caution=caution,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("clip_exp", None)
            group.setdefault("caution", False)
            for p in group["params"]:
                p_state: list = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):  # type: ignore
                    step_val: float = float(p_state["step"])  # type: ignore
                    p_state["step"] = (  # type: ignore
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(),
                            device=p.device,
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(  # NOSONAR
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
    ) -> bool:
        has_complex: bool = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("ADOPT does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
            # Lazy state initialization
            if len(state) == 0:
                # note(crcrpar): [special device hosting for step]
                # Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(), device=p.grad.device)
                    if group["capturable"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError(
                    "`requires_grad` is not supported for `step` in `differentiable=True` mode"
                )

            # Foreach without capturable does not support a tensor lr
            if (
                group["foreach"]
                and torch.is_tensor(group["lr"])
                and not group["capturable"]
            ):
                raise RuntimeError(_nocapture_err)

            state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex: bool = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            adopt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                clip_exp=group["clip_exp"],
                decouple=group["decouple"],
                eps=group["eps"],
                caution=group["caution"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


# noinspection PyProtectedMember
def _single_tensor_adopt(  # NOSONAR
    params: List[Tensor],  # NOSONAR
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    clip_exp: Optional[float],
    decouple: bool,
    eps: float,
    caution: bool,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    assert grad_scale is None and found_inf is None

    _ = has_complex  # unused

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad: Tensor = grads[i] if not maximize else -grads[i]
        exp_avg: Tensor = exp_avgs[i]
        exp_avg_sq: Tensor = exp_avg_sqs[i]
        step_t: Tensor = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if capturable and not torch.compiler.is_compiling():
            from torch.optim.optimizer import _get_capturable_supported_devices

            capturable_supported_devices: List[str] = (
                _get_capturable_supported_devices()
            )
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on\
                supported devices: {capturable_supported_devices}."

        # update step
        step_t += 1

        if torch.is_complex(param):
            grad: Tensor = torch.view_as_real(grad)
            if exp_avg is not None:
                exp_avg: Tensor = torch.view_as_real(exp_avg)
            if exp_avg_sq is not None:
                exp_avg_sq: Tensor = torch.view_as_real(exp_avg_sq)
            param: Tensor = torch.view_as_real(param)

        if weight_decay != 0 and not decouple:
            grad: Tensor = grad.add(param, alpha=weight_decay)

        step = step_t if capturable or differentiable else _get_value(step_t)
        if step == 1:
            exp_avg_sq.addcmul_(grad, grad.conj())
            continue

        if weight_decay != 0 and decouple:
            param.add_(param, alpha=-lr * weight_decay)

        denom: Tensor = torch.clamp(exp_avg_sq.sqrt(), eps)
        normed_grad: Tensor = grad.div(denom)

        if clip_exp is not None:
            clip_val = (step - 1) ** clip_exp
            normed_grad.clamp_(-clip_val, clip_val)

        exp_avg.lerp_(normed_grad, 1 - beta1)

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            mask: Tensor = (exp_avg * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            exp_avg: Tensor = exp_avg * mask

        param.add_(exp_avg, alpha=-lr)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)


# noinspection PyProtectedMember
def _multi_tensor_adopt(  # NOSONAR
    params: List[Tensor],  # NOSONAR
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    clip_exp: Optional[float],
    decouple: bool,
    eps: float,
    caution: bool,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
) -> None:
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(_nocapture_err)

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if capturable and not torch.compiler.is_compiling():
        from torch.optim.optimizer import _get_capturable_supported_devices

        capturable_supported_devices: List[str] = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        # Handle complex parameters
        if has_complex:
            _view_as_real(
                device_params, device_grads, device_exp_avgs, device_exp_avg_sqs
            )

        if maximize:
            device_grads: List[Tensor] = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        if weight_decay != 0 and not decouple:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads: List[Tensor] = torch._foreach_add(
                    device_grads, device_params, alpha=weight_decay
                )

        if device_state_steps[0] == 1:
            torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads)
            continue

        if weight_decay != 0 and decouple:
            torch._foreach_add_(device_params, device_params, alpha=-lr * weight_decay)

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        torch._foreach_maximum_(exp_avg_sq_sqrt, eps)

        normed_grad = torch._foreach_div(device_grads, exp_avg_sq_sqrt)

        if clip_exp is not None:
            clip_val = (device_state_steps[0] - 1) ** clip_exp
            torch._foreach_maximum_(normed_grad, -clip_val)
            torch._foreach_minimum_(normed_grad, clip_val)

        torch._foreach_lerp_(device_exp_avgs, normed_grad, 1 - beta1)

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            masks = torch._foreach_mul(device_exp_avgs, device_grads)
            masks = [(m > 0).to(g.dtype) for m, g in zip(masks, device_grads)]
            mask_scale: list = [m.mean() for m in masks]
            torch._foreach_maximum_(mask_scale, 1e-3)
            torch._foreach_div_(masks, mask_scale)
            device_exp_avgs: List[Tensor] = torch._foreach_mul(device_exp_avgs, masks)

        torch._foreach_add_(device_params, device_exp_avgs, alpha=-lr)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs, device_grads, device_grads, value=1 - beta2
        )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adopt)
def adopt(  # NOSONAR
    params: List[Tensor],  # NOSONAR
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    clip_exp: Optional[float],
    decouple: bool,
    eps: float,
    caution: bool,
    maximize: bool,
) -> None:
    r"""Functional API that performs ADOPT algorithm computation."""
    if foreach is None:
        foreach: bool = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    # noinspection PyProtectedMember
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func: Callable = _multi_tensor_adopt
    else:
        func: Callable = _single_tensor_adopt

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        clip_exp=clip_exp,
        decouple=decouple,
        eps=eps,
        caution=caution,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
