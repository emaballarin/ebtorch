#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections.abc import Callable
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch import nn
from torch import Tensor

from .functional import tensor_replicate

TTupList = Union[Tuple[Tensor, ...], List[Tensor]]

__all__ = [
    "ScatterGatherModule",
    "scatter_even_split",
    "scatter_replicate",
    "gather_cat",
    "gather_sum",
    "module_rep_deploy",
]


class ScatterGatherModule(nn.Module):
    def __init__(
        self,
        scatter_fx: Callable[[Tensor], TTupList],
        scattered_modules: nn.ModuleList,
        gather_fx: Callable[[TTupList], Tensor],
    ):
        super().__init__()
        self.scatter_fx: Callable[[Tensor], TTupList] = scatter_fx
        self.scattered_modules: nn.ModuleList = scattered_modules
        self.gather_fx: Callable[[TTupList], Tensor] = gather_fx

    def forward(self, x: Tensor) -> Tensor:
        scattered: TTupList = self.scatter_fx(x)
        scattered: Tensor = torch.stack(scattered, dim=0)
        scattered: Tensor = torch.vmap(
            lambda mm, xx: mm(xx), in_dims=(0, 0), out_dims=0, randomness="different"
        )(self.scattered_modules, scattered)
        scattered: TTupList = scattered.unbind(dim=0)
        return self.gather_fx(scattered)


def scatter_even_split(x: Tensor, nsplits: int) -> TTupList:
    return x.split(x.shape[1] // nsplits, dim=1)


def scatter_replicate(x: Tensor, nsplits: int) -> TTupList:
    return tensor_replicate(x, nsplits, dim=0).unbind(dim=0)


def gather_cat(x: TTupList) -> Tensor:
    return torch.cat(x, dim=1)


def gather_sum(x: TTupList) -> Tensor:
    return torch.sum(torch.stack(x, dim=0), dim=0)


def module_rep_deploy(
    module_deployer: Callable[[], nn.Module], nrep: int
) -> nn.ModuleList:
    return nn.ModuleList([module_deployer() for _ in range(nrep)])
