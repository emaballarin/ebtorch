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
#
# Imports
from .adverutils import AdverApply
from .adverutils import TA2ATAdapter
from .autoclip import AutoClipper
from .onlyutils import argser_f
from .onlyutils import argsink
from .onlyutils import download_gdrive
from .onlyutils import emplace_kv
from .reprutils import gather_model_repr
from .reprutils import model_reqgrad
from .reprutils import model_reqgrad_
from .reprutils import store_repr_autohook
from .reprutils import store_repr_fx
from .reprutils import store_repr_hook

# Deletions (from .)
del adverutils
del autoclip
del onlyutils
del reprutils
