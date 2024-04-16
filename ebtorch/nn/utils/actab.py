#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List
from typing import Tuple

import torch

# ──────────────────────────────────────────────────────────────────────────────

__all__: List[str] = ["act_auto_broadcast", "broadcast_in_dim"]

# ──────────────────────────────────────────────────────────────────────────────


def broadcast_in_dim(
    xin: torch.Tensor, target_shape: torch.Size, broadcast_dims: Tuple[int, ...]
) -> torch.Tensor:
    s: List[int] = list(target_shape)
    for broadcast_dim in broadcast_dims:
        s[broadcast_dim] = -1
    for idx, dim in enumerate(s):
        if dim != -1:
            xin: torch.Tensor = xin.unsqueeze(idx)
    return xin.expand(target_shape)


# noinspection PyProtectedMember
def act_auto_broadcast(
    xin: torch.Tensor, xpar: torch.Tensor, act_name: str, xpar_name: str
) -> torch.Tensor:
    if xpar.numel() != 1:
        torch._check(
            xin.ndim > 0,
            lambda: f"A zero-dimensional input tensor is not allowed unless `{xpar_name}` contains exactly one element.",
        )
        channel_size: int = xin.shape[1] if xin.ndim >= 2 else 1
        torch._check(
            xpar.numel() == channel_size,
            lambda: f"Mismatch of parameters number and input channel size. Found parameters number ="
            f" {xpar.numel()} and channel size = {channel_size}.",
        )
    torch._check(
        xpar.ndim in (0, 1),
        lambda: f"{act_name}: Expected `{xpar_name}` to be a scalar or 1D tensor, but got: "
        f"ndim = {xpar.ndim}",
    )

    if xin.ndim == 0:
        xpar_ret: torch.Tensor = xpar[0] if xpar.ndim == 1 else xpar
    else:
        xpar_ret: torch.Tensor = broadcast_in_dim(
            xpar,
            xin.shape,
            () if xpar.ndim == 0 else (0 if xin.ndim == 1 else 1,),  # NOSONAR
        )

    return xpar_ret
