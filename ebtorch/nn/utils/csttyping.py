#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# Imports
from collections.abc import Callable
from typing import List
from typing import Union

import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = ["realnum", "strdev", "numlike", "actvt"]
# ──────────────────────────────────────────────────────────────────────────────
realnum = Union[int, float]
strdev = Union[str, torch.device]
numlike = Union[realnum, Tensor]
actvt = Union[torch.nn.Module, Callable[[Tensor], Tensor]]
# ──────────────────────────────────────────────────────────────────────────────
