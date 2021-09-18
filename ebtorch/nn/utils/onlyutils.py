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
from typing import Union
from functools import partial as fpartial


# Functions
def argser_f(f, arglist: Union[list, tuple, dict]):
    error_listerror = "Function arguments must be either an args tuple or a kwargs dictionary, or both in this order inside a list."
    if isinstance(arglist, list):
        if len(arglist) == 0:
            return fpartial(f)
        elif len(arglist) > 2:
            raise ValueError(error_listerror)
        elif len(arglist) == 2:
            return fpartial(f, *arglist[0], **arglist[1])
        else:
            if isinstance(arglist[0], tuple):
                return fpartial(f, *arglist[0])
            if isinstance(arglist[0], dict):
                return fpartial(f, **arglist[0])
    elif isinstance(arglist, tuple):
        return fpartial(f, *arglist)
    elif isinstance(arglist, dict):
        return fpartial(f, **arglist)
    else:
        raise ValueError(error_listerror)


def emplace_kv(dictionary: dict, k, v) -> dict:
    """
    Returns input dict with added k:v pair, overwriting if k already exists
    """
    return {**dictionary, k: v}
