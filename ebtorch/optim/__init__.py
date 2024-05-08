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
from .adabound import AdaBound
from .adahessian import AdaHessian
from .adan import Adan
from .custom import epochwise_onecycle
from .custom import make_beta_scheduler
from .custom import onecycle_lincos
from .custom import onecycle_linlin
from .custom import onecycle_linlin_updown
from .custom import ralah_optim
from .custom import tricyc1c
from .custom import warmed_up_linneal
from .custom import wfneal
from .lamb import Lamb
from .lion import Lion
from .lookahead import Lookahead
from .lookaround import Lookaround
from .poptims import AdamP
from .poptims import SGDP
from .radam import RAdam
from .sam import SAM
from .schedcos import CosineLRScheduler

# Deletions (from .)
del adabound
del adahessian
del adan
del custom
del lamb
del lion
del lookahead
del lookaround
del poptims
del radam
del sam
del schedcos
