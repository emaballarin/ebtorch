#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
import math
from typing import List
from typing import Union

import torch as th
from torch import Tensor

from ...typing import numlike

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = ["div_by_factorial", "auto_comb", "log_muldiv"]


# ──────────────────────────────────────────────────────────────────────────────
def div_by_factorial(x: Tensor, n: int, max_exact_n: int = 18) -> Tensor:
    """
    Compute the division of `x` by `n!` using Stirling's approximation for large `n`.

    Parameters
    ----------
    x : Tensor
        The dividend tensor.
    n : int
        The factorial to divide by.
    max_exact_n : int, optional
        The maximum `n` for which the exact value of `n!` is computed, by default 18.

    Returns
    -------
    Tensor
        The result of the division.
    """
    if n < 0:
        raise ValueError("`n` must be a non-negative integer.")

    if n <= max_exact_n:
        return x / math.factorial(n)

    log_factorial_n: float = n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)
    return th.sign(x) * th.exp(th.log(th.abs(x)) - log_factorial_n)


def auto_comb(n: int, k: int, max_exact_n: int = 66) -> Union[int, float]:
    """
    Compute the binomial coefficient `n choose k` using Stirling's approximation for large `n`.

    Parameters
    ----------
    n : int
        The total number of items.
    k : int
        The number of items to choose.
    max_exact_n : int, optional
        The maximum `n` for which the exact value of `n!` is computed, by default 66.

    Returns
    -------
    Union[int, float]
        The binomial coefficient.
    """

    if n <= max_exact_n:
        return math.comb(n, k)
    else:
        lognf: float = n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)
        logkf = k * math.log(k) - k + 0.5 * math.log(2 * math.pi * k) if k > 1 else 0
        lognkf = (
            (n - k) * math.log(n - k) - (n - k) + 0.5 * math.log(2 * math.pi * (n - k))
            if (n - k) > 1
            else 0
        )
        return math.exp(lognf - logkf - lognkf)


# ──────────────────────────────────────────────────────────────────────────────


def log_muldiv(x: numlike, y: numlike, div_not_mul: bool = False) -> numlike:
    """
    Compute product/division of `x` and `y`, in log-space for numerical stability.

    Parameters
    ----------
    x : numlike
        The first element.
    y : numlike
        The second element.
    div_not_mul : bool, optional
        Whether to perform division instead of multiplication, by default False.

    Returns
    -------
    numlike
        The result of the multiplication/division.
    """
    if xit := isinstance(x, Tensor):
        xlog = th.log
    else:
        xlog = math.log

    if yit := isinstance(y, Tensor):
        ylog = th.log
    else:
        ylog = math.log

    if xit or yit:
        exp = th.exp
    else:
        exp = math.exp

    if div_not_mul:
        return exp(xlog(x) - ylog(y))
    return exp(xlog(x) + ylog(y))
