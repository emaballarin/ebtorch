#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
# All Rights Reserved. Unless otherwise explicitly stated.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
#
# SPDX-License-Identifier: Apache-2.0

# Imports (wildcard)
from .utils import *  # lgtm [py/polluting-import]
from .functional import mish

# Imports (specific)
from .kwta import KWTA1d, KWTA2d, BrokenReLU
from .architectures import FCBlockLegacy, FCBlock, CausalConv1d
from .lmu import LMUCell
from .mish import Mish, mishlayer_init
from .adaptiveio import VariationalDropout, AdaptiveInput, AdaptiveLogSoftmaxWithLoss

# Deletions (from .)
del kwta
del architectures
del lmu
del mish
del adaptiveio

# Deletions (from .functional)
# del mish  # (already done by chance!)

# Deletions (from .utils)
del AutoClipper
del store_repr_fx
del store_repr_hook
del store_repr_autohook
del argser_f
