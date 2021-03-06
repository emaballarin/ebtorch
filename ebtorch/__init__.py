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

# IMPORTS:

from .nn import *  # lgtm [py/polluting-import]
from .optim import *  # lgtm [py/polluting-import]
from .logging import *  # lgtm [py/polluting-import]


# DELETIONS:

# (from .nn)
del KWTA1d
del KWTA2d
del BrokenReLU
del FCBlock
del LMUCell

# (from .optim)
del Lookahead
del RAdam
del PlainRAdam
del WarmAdamW
del Ranger
del utils
del SAM
del HCGD
del HCAdam
del MADGRAD

# (from .utils)
del LogCSV
del AverageMeter

# (from .mish)
del Mish
del mishlayer_init
