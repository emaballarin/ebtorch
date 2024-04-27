#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Generator
from typing import List
from typing import TypeVar
from typing import Union

from torch import nn
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "matched_apply",
    "tensor_module_matched_apply",
]
# ──────────────────────────────────────────────────────────────────────────────

FInputType = TypeVar("FInputType")
FOutputType = TypeVar("FOutputType")

# ──────────────────────────────────────────────────────────────────────────────


def matched_apply(
    x: Collection[FInputType],
    f: Union[Collection[Callable[[FInputType], FOutputType]], nn.ModuleList],
    to_list: bool = False,
) -> Union[Generator[FOutputType, None, None], List[FOutputType]]:
    if len(x) != len(f):
        raise ValueError("Length of `x` and `f` must be the same.")
    gen: Generator[FOutputType, None, None] = (fi(xi) for fi, xi in zip(f, x))
    return list(gen) if to_list else gen


# ──────────────────────────────────────────────────────────────────────────────


def tensor_module_matched_apply(x: List[Tensor], f: nn.ModuleList) -> List[Tensor]:
    return matched_apply(x=x, f=f, to_list=True)
