#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List

from torch import nn
from torch import Tensor

from .functional import telu as ftelu

__all__: List[str] = ["TeLU"]


class TeLU(nn.Module):
    """TeLU Function."""

    def __init__(self) -> None:
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x: Tensor) -> Tensor:
        return ftelu(x)
