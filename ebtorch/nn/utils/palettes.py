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
from typing import List

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.rcsetup import cycler

# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    "petroff_2021_color",
    "tableau10_color",
    "petroff_2021_cycler",
    "tableau10_cycler",
    "petroff_2021_cmap",
    "tableau10_cmap",
    "set_petroff_2021_colors",
    "set_tableau10_colors",
]
# ──────────────────────────────────────────────────────────────────────────────

# After: M. A. Petroff, "Accessible Color Sequences for Data Visualization", 2021
# (https://arxiv.org/abs/2107.02270;
# https://github.com/matplotlib/matplotlib/issues/9460#issuecomment-875185352)
petroff_2021_color: List[str] = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]

# After: M. Stone, "How we designed the new color palettes in Tableau 10", 2016
# (https://www.tableau.com/blog/colors-upgrade-tableau-10-56782)
tableau10_color: List[str] = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]

# ──────────────────────────────────────────────────────────────────────────────

petroff_2021_cycler = cycler(color=petroff_2021_color)
tableau10_cycler = cycler(color=tableau10_color)

# ──────────────────────────────────────────────────────────────────────────────
petroff_2021_cmap = LinearSegmentedColormap.from_list(
    name="petroff_2021", colors=petroff_2021_color, N=len(petroff_2021_color)
)

tableau10_cmap = LinearSegmentedColormap.from_list(
    name="tableau10", colors=tableau10_color, N=len(tableau10_color)
)
# ──────────────────────────────────────────────────────────────────────────────


def set_petroff_2021_colors() -> None:
    mpl.rcParams["axes.prop_cycle"] = petroff_2021_cycler


def set_tableau10_colors() -> None:
    mpl.rcParams["axes.prop_cycle"] = tableau10_cycler
