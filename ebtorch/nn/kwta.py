#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2019-* Chang Xiao, Peilin Zhong, Changxi Zheng
#                      (Columbia University). All Rights Reserved.
#                      [orig. work: https://arxiv.org/abs/1905.10510;
#                       orig. code: https://github.com/a554b554/kWTA-Activation]
#
# Copyright (c) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
#                      [maintainance, adaptation, extension]
#
# ==============================================================================
# For type-annotation
from collections.abc import Callable
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

__all__ = ["KWTA1d", "KWTA2d", "BrokenReLU"]

# Custom type-annotation types
realnum = Union[float, int]


# Trivial reduction functions
def onlyratiok(ratiok: realnum, k: int) -> int:
    _: int = k  # discarded
    return int(ratiok)


def onlyk(ratiok: realnum, k: int) -> int:
    _: realnum = ratiok  # discarded
    return k


def intmax(ratiok: realnum, k: int) -> int:
    return int(max(ratiok, k))


# Trivial 'absolute transformation' functions
def noabs(x: Tensor) -> Tensor:
    return x


def doabs(x: Tensor) -> Tensor:
    return torch.abs(x)


# Flippable comparison operator
def flipcmp(lhs: Any, rhs: Any, largest: bool = True) -> Any:
    if largest:
        return lhs >= rhs
    else:
        return rhs >= lhs


# Error messages
red_not_none: str = str("'reduction' applies only if both 'ratio' and 'k' are defined!")
red_none_but: str = str(
    "'reduction' must be defined if both 'ratio' and 'k' are defined!"
)


class KWTA1d(Module):
    __constants__: List[str] = ["largest"]
    largest: bool

    def __init__(
        self,
        largest: bool = True,
        absolute: bool = False,
        ratio: Optional[realnum] = None,
        k: Optional[int] = None,
        reduction: Callable[[realnum, realnum], realnum] = None,
    ) -> None:
        super(KWTA1d, self).__init__()

        self.largest: bool = largest

        # Build the 'absolute transformation' function
        if not absolute:
            self.abstransf: Callable[[Tensor], Tensor] = noabs
        else:
            self.abstransf: Callable[[Tensor], Tensor] = doabs

        # Define usage logic and value-check arguments
        if ratio is None and k is None:
            # Follow default behaviour (50% but not < 1)
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnum = 0.5
            self.k: int = 1
            self.reduction: Callable[[realnum, realnum], realnum] = intmax

        elif ratio is not None and k is None:
            # Use user-defined 'ratio' and check 'reduction' is None
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnum = ratio
            self.k: int = 0
            self.reduction: Callable[[realnum, realnum], realnum] = onlyratiok

        elif ratio is None and k is not None:
            # Use user-defined 'k' and check 'reduction' is None
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnum = 0
            self.k: int = k
            self.reduction: Callable[[realnum, realnum], realnum] = onlyk

        else:
            # Use user-defined 'ratio', 'k' and 'reduction' (must be not None)
            if reduction is None:
                raise ValueError(red_none_but)
            self.ratio: realnum = ratio
            self.k: int = k
            self.reduction: Callable[[realnum, realnum], realnum] = reduction

    def forward(self, x: Tensor) -> Tensor:
        k: int = int(self.reduction(self.ratio * x.shape[1], self.k))
        transfx: Tensor = self.abstransf(x)
        topval = transfx.topk(k, dim=1, largest=self.largest)[0][:, -1]
        topval = topval.expand(transfx.shape[1], transfx.shape[0]).permute(1, 0)
        comp = (flipcmp(transfx, topval, self.largest)).to(x)  # acceptable
        return comp * x

    def extra_repr(self) -> str:
        return f"largest={self.largest}, abstransf={self.abstransf}, ratio={self.ratio}, k={self.k}, reduction={self.reduction}"


class KWTA2d(Module):
    __constants__: List[str] = ["largest"]
    largest: bool

    def __init__(
        self,
        largest: bool = True,
        absolute: bool = False,
        ratio: Optional[realnum] = None,
        k: Optional[int] = None,
        reduction: Callable[[realnum, realnum], realnum] = None,
        xchan: bool = False,
    ) -> None:
        super(KWTA2d, self).__init__()

        self.largest: bool = largest

        # Build the 'absolute transformation' function
        if not absolute:
            self.abstransf: Callable[[Tensor], Tensor] = noabs
        else:
            self.abstransf: Callable[[Tensor], Tensor] = doabs

        # Define usage logic and value-check arguments
        if ratio is None and k is None:
            # Follow default behaviour (50% but not < 1)
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnum = 0.5
            self.k: int = 1
            self.reduction: Callable[[realnum, realnum], realnum] = intmax

        elif ratio is not None and k is None:
            # Use user-defined 'ratio' and check 'reduction' is None
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnum = ratio
            self.k: int = 0
            self.reduction: Callable[[realnum, realnum], realnum] = onlyratiok

        elif ratio is None and k is not None:
            # Use user-defined 'k' and check 'reduction' is None
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnum = 0
            self.k: int = k
            self.reduction: Callable[[realnum, realnum], realnum] = onlyk

        else:
            # Use user-defined 'ratio', 'k' and 'reduction' (must be not None)
            if reduction is None:
                raise ValueError(red_none_but)
            self.ratio: realnum = ratio
            self.k: int = k
            self.reduction: Callable[[realnum, realnum], realnum] = reduction

        if not xchan:
            self.actual_forward: Callable[
                [Tensor],
                Tensor,
            ] = self.channelwise_forward
        else:
            self.actual_forward: Callable[
                [Tensor],
                Tensor,
            ] = self.crosschannel_forward

    # Define methods corresponding to channelwise or crosschannel forward
    def channelwise_forward(self, x: Tensor) -> Tensor:
        size: realnum = x.shape[2] * x.shape[3]
        k: int = int(self.reduction(self.ratio * size, self.k))
        transfx: Tensor = self.abstransf(x)
        tmpx = transfx.view(transfx.shape[0], transfx.shape[1], -1)
        topval = tmpx.topk(k, dim=2, largest=self.largest)[0][:, :, -1]
        topval = topval.expand(
            transfx.shape[2], transfx.shape[3], transfx.shape[0], transfx.shape[1]
        ).permute(2, 3, 0, 1)
        comp = (flipcmp(transfx, topval, self.largest)).to(x)  # acceptable
        return comp * x

    def crosschannel_forward(self, x: Tensor) -> Tensor:
        size: realnum = x.shape[1] * x.shape[2] * x.shape[3]
        k: int = int(self.reduction(self.ratio * size, self.k))
        transfx: Tensor = self.abstransf(x)
        tmpx = transfx.view(transfx.shape[0], -1)
        topval = tmpx.topk(k, dim=1, largest=self.largest)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(transfx)
        comp = (flipcmp(transfx, topval, self.largest)).to(x)  # acceptable
        return comp * x

    def forward(self, x: Tensor) -> Tensor:
        return self.actual_forward(x)

    def extra_repr(self) -> str:
        return f"largest={self.largest}, abstransf={self.abstransf}, ratio={self.ratio}, k={self.k}, reduction={self.reduction}, actual_forward={self.actual_forward}"


class BrokenReLU(Module):
    __constants__: List[str] = ["plateau", "inplace"]
    plateau: float
    inplace: bool

    def __init__(self, plateau: float = -5.0, inplace: bool = False) -> None:
        super(BrokenReLU, self).__init__()
        self.plateau: float = plateau
        self.inplace: bool = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.threshold(x, 0.0, self.plateau, self.inplace)

    def extra_repr(self) -> str:
        inplace_str: str = ", inplace=True" if self.inplace else ""
        return f"plateau={self.plateau}{inplace_str}"
