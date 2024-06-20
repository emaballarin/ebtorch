#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from collections.abc import Callable
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

# ──────────────────────────────────────────────────────────────────────────────
__all__ = ["find_lr"]
# ──────────────────────────────────────────────────────────────────────────────


def find_lr(
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
