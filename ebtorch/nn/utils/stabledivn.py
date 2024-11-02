#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from ...typing import numlike

# ──────────────────────────────────────────────────────────────────────────────


def stablediv(
    num: numlike, den: numlike, eps: numlike, stabilize_both: bool = False
) -> numlike:
    """Numerically stable division of two numbers.

    Args:
        num (numlike): Numerator.
        den (numlike): Denominator.
        eps (numlike): Numerical stability factor.
        stabilize_both (bool, optional): Whether to stabilize both terms. Defaults to False.
    """
    return (num + eps * stabilize_both) / (den + eps)
