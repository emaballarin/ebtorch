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
# Imports (specific)
from .adan import Adan
from .hcgd import HCAdam
from .hcgd import HCGD
from .lookahead import Lookahead
from .madgrad import MADGRAD
from .radam import PlainRAdam
from .radam import RAdam
from .radam import WarmAdamW
from .ranger import Ranger
from .sam import SAM

# Deletions (from .)
del lookahead
del radam
del adan
del ranger
del sam
del hcgd
del madgrad
