# ==============================================================================
#
# Copyright (c) 2019-* Chang Xiao, Peilin Zhong, Changxi Zheng
#                      (Columbia University). All Rights Reserved.
#                      [orig. work: https://arxiv.org/abs/1905.10510;
#                      orig. code: https://github.com/a554b554/kWTA-Activation]
#
# Copyright (c) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
#                      [maintainance, adaptation, extension]
#
# ==============================================================================


# For type-annotation
from typing import Optional, Callable, Union, Any

# For neural network layers and tensor functions
import torch
from torch import nn


# Custom type-annotation types
realnumeric = Union[float, int]


# Trivial reduction functions
def onlyratiok(ratiok: realnumeric, k: int) -> int:
    _: int = k  # discarded
    return int(ratiok)


def onlyk(ratiok: realnumeric, k: int) -> int:
    _: realnumeric = ratiok  # discarded
    return k


def intmax(ratiok: realnumeric, k: int) -> int:
    return int(max(ratiok, k))


# Trivial 'absolute transformation' functions
def noabs(x: torch.Tensor) -> torch.Tensor:
    return x


def doabs(x: torch.Tensor) -> torch.Tensor:
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


class KWTA1d(nn.Module):
    def __init__(
        self,
        largest: bool = True,
        absolute: bool = False,
        ratio: Optional[realnumeric] = None,
        k: Optional[int] = None,
        reduction: Callable[[realnumeric, realnumeric], realnumeric] = None,
    ) -> None:
        super(KWTA1d, self).__init__()

        self.largest: bool = largest

        # Build the 'absolute transformation' function
        if not absolute:
            self.abstransf: Callable[[torch.Tensor], torch.Tensor] = noabs
        else:
            self.abstransf: Callable[[torch.Tensor], torch.Tensor] = doabs

        # Define usage logic and value-check arguments
        if ratio is None and k is None:
            # Follow default behaviour (50% but not < 1)
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnumeric = 0.5
            self.k: int = 1
            self.reduction: Callable[[realnumeric, realnumeric], realnumeric] = intmax

        elif ratio is not None and k is None:
            # Use user-defined 'ratio' and check 'reduction' is None
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnumeric = ratio
            self.k: int = 0
            self.reduction: Callable[
                [realnumeric, realnumeric], realnumeric
            ] = onlyratiok

        elif ratio is None and k is not None:
            # Use user-defined 'k' and check 'reduction' is None
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnumeric = 0
            self.k: int = k
            self.reduction: Callable[[realnumeric, realnumeric], realnumeric] = onlyk

        else:
            # Use user-defined 'ratio', 'k' and 'reduction' (must be not None)
            if reduction is None:
                raise ValueError(red_none_but)
            self.ratio: realnumeric = ratio
            self.k: int = k
            self.reduction: Callable[
                [realnumeric, realnumeric], realnumeric
            ] = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k: int = int(self.reduction(self.ratio * x.shape[1], self.k))
        transfx: torch.Tensor = self.abstransf(x)
        topval = transfx.topk(k, dim=1, largest=self.largest)[0][:, -1]
        topval = topval.expand(transfx.shape[1], transfx.shape[0]).permute(1, 0)
        comp = (flipcmp(transfx, topval, self.largest)).to(x)  # acceptable
        return comp * x


class KWTA2d(nn.Module):
    def __init__(
        self,
        largest: bool = True,
        absolute: bool = False,
        ratio: Optional[realnumeric] = None,
        k: Optional[int] = None,
        reduction: Callable[[realnumeric, realnumeric], realnumeric] = None,
        xchan: bool = False,
    ) -> None:
        super(KWTA2d, self).__init__()

        self.largest: bool = largest

        # Build the 'absolute transformation' function
        if not absolute:
            self.abstransf: Callable[[torch.Tensor], torch.Tensor] = noabs
        else:
            self.abstransf: Callable[[torch.Tensor], torch.Tensor] = doabs

        # Define usage logic and value-check arguments
        if ratio is None and k is None:
            # Follow default behaviour (50% but not < 1)
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnumeric = 0.5
            self.k: int = 1
            self.reduction: Callable[[realnumeric, realnumeric], realnumeric] = intmax

        elif ratio is not None and k is None:
            # Use user-defined 'ratio' and check 'reduction' is None
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnumeric = ratio
            self.k: int = 0
            self.reduction: Callable[
                [realnumeric, realnumeric], realnumeric
            ] = onlyratiok

        elif ratio is None and k is not None:
            # Use user-defined 'k' and check 'reduction' is None
            if reduction is not None:
                raise ValueError(red_not_none)
            self.ratio: realnumeric = 0
            self.k: int = k
            self.reduction: Callable[[realnumeric, realnumeric], realnumeric] = onlyk

        else:
            # Use user-defined 'ratio', 'k' and 'reduction' (must be not None)
            if reduction is None:
                raise ValueError(red_none_but)
            self.ratio: realnumeric = ratio
            self.k: int = k
            self.reduction: Callable[
                [realnumeric, realnumeric], realnumeric
            ] = reduction

        if not xchan:
            self.actual_forward: Callable[
                [torch.Tensor],
                torch.Tensor,
            ] = self.channelwise_forward
        else:
            self.actual_forward: Callable[
                [torch.Tensor],
                torch.Tensor,
            ] = self.crosschannel_forward

    # Define methods corresponding to channelwise or crosschannel forward
    def channelwise_forward(self, x: torch.Tensor) -> torch.Tensor:
        size: realnumeric = x.shape[2] * x.shape[3]
        k: int = int(self.reduction(self.ratio * size, self.k))
        transfx: torch.Tensor = self.abstransf(x)
        tmpx = transfx.view(transfx.shape[0], transfx.shape[1], -1)
        topval = tmpx.topk(k, dim=2, largest=self.largest)[0][:, :, -1]
        topval = topval.expand(
            transfx.shape[2], transfx.shape[3], transfx.shape[0], transfx.shape[1]
        ).permute(2, 3, 0, 1)
        comp = (flipcmp(transfx, topval, self.largest)).to(x)  # acceptable
        return comp * x

    def crosschannel_forward(self, x: torch.Tensor) -> torch.Tensor:
        size: realnumeric = x.shape[1] * x.shape[2] * x.shape[3]
        k: int = int(self.reduction(self.ratio * size, self.k))
        transfx: torch.Tensor = self.abstransf(x)
        tmpx = transfx.view(transfx.shape[0], -1)
        topval = tmpx.topk(k, dim=1, largest=self.largest)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(transfx)
        comp = (flipcmp(transfx, topval, self.largest)).to(x)  # acceptable
        return comp * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actual_forward(x)


class BrokenReLU(nn.Module):
    # 'Broken ReLU' activation function
    def __init__(self, plateau: float = -5.0, inplace: bool = False) -> None:
        super(BrokenReLU, self).__init__()
        self.brokenrelu = nn.Threshold(0.0, plateau, inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.brokenrelu(x)
