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
from .logging import *  # lgtm [py/polluting-import]
from .nn import *  # lgtm [py/polluting-import]
from .optim import *  # lgtm [py/polluting-import]

# Deletions (from .nn)
del KWTA1d
del KWTA2d
del BrokenReLU
del FCBlockLegacy
del FCBlock
del CausalConv1d
del SGRUHCell
del LMUCell
del VariationalDropout
del AdaptiveInput
del AdaptiveLogSoftmaxWithLoss
del FieldTransform
del Mish
del MishPulse
del MishPulseSymmY
del mishlayer_init
del field_transform
del mishpulse
del mishpulse_symmy
del GaussianReparameterizerSampler
del SERLU
del SmeLU
del ReshapeLayer
del FlatChannelize2DLayer
del ProbePrintLayer

# Deletions (from .optim)
del Lookahead
del RAdam
del PlainRAdam
del Adan
del WarmAdamW
del Ranger
del SAM
del HCGD
del HCAdam
del MADGRAD

# Deletions (from .logging)
del LogCSV
del AverageMeter
