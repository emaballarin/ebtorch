#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from collections.abc import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy
import torch
from numpy.typing import NDArray
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils import data as torch_data
from torch_lr_finder import LRFinder
from tqdm.auto import trange

from .onlyutils import suppress_std

# ──────────────────────────────────────────────────────────────────────────────
__all__ = ["find_lr"]
# ──────────────────────────────────────────────────────────────────────────────


def _find_lr(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: Union[Callable[[Tensor, Tensor], Tensor], nn.Module],
    train_dl: torch_data.DataLoader,
    start_lr: float = 1e-8,
    end_lr: float = 100.0,
    num_iter: int = 100,
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[float]:

    lr_finder: LRFinder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_dl, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
    _ = lr_finder.plot()

    lrs: NDArray = numpy.array(lr_finder.history["lr"])
    lss: NDArray = numpy.array(lr_finder.history["loss"])

    try:
        min_gidx: Optional[numpy.int64] = (numpy.gradient(numpy.array(lss))).argmin()
    except ValueError:
        min_gidx: Optional[numpy.int64] = None

    retlr: Optional[float] = lrs[min_gidx] if min_gidx is not None else None

    lr_finder.reset()
    return retlr


def find_lr(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: Union[Callable[[Tensor, Tensor], Tensor], nn.Module],
    train_dl: torch_data.DataLoader,
    start_lr: float = 1e-8,
    end_lr: float = 100.0,
    num_iter: int = 100,
    num_rep: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
    noprint: bool = False,
) -> float:
    rep_lrs: List[float] = []

    with suppress_std("all" if (not verbose or noprint) else "none"):
        for _ in trange(num_rep):
            lr = _find_lr(
                model,
                optimizer,
                criterion,
                train_dl,
                start_lr,
                end_lr,
                num_iter,
                device,
            )
            rep_lrs.append(lr)

    lrpick: float = numpy.median(rep_lrs).item()

    if not noprint:
        print(f"Picking LR: {lrpick:.2E}")

    return lrpick
