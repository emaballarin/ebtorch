#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# Imports
from typing import List
from typing import Union

import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = ["realnum", "strdev", "numlike"]
# ──────────────────────────────────────────────────────────────────────────────
realnum = Union[int, float]
strdev = Union[str, torch.device]
numlike = Union[realnum, Tensor]
# ──────────────────────────────────────────────────────────────────────────────
