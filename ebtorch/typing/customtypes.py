#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# Imports
from collections.abc import Callable
from typing import List
from typing import Union

import numpy as np
import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = ["realnum", "strdev", "numlike", "tensorlike", "actvt"]
# ──────────────────────────────────────────────────────────────────────────────
realnum = Union[int, float]
strdev = Union[str, torch.device]
tensorlike = Union[Tensor, np.ndarray]
numlike = Union[realnum, tensorlike]
actvt = Union[torch.nn.Module, Callable[[Tensor], Tensor]]
# ──────────────────────────────────────────────────────────────────────────────
