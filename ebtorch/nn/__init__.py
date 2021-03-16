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

# Imports
from .kwta import KWTA1d, KWTA2d, BrokenReLU
from .architectures import FCBlock
from .utils import *  # lgtm [py/polluting-import]

# Deletions
del kwta
del AutoClipper
del architectures
