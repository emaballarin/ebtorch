#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#
#  Copyright (c) 2020-2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ──────────────────────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: MIT
#
# ──────────────────────────────────────────────────────────────────────────────
# Imports
from .actab import act_auto_broadcast
from .actab import broadcast_in_dim
from .adverutils import AdverApply
from .adverutils import TA2ATAdapter
from .autoclip import AutoClipper
from .evalutils import eval_model_on_test
from .filtermanip import extract_conv_filters
from .filtermanip import show_filters
from .lrfinder import find_lr
from .mapply import matched_apply
from .mapply import tensor_module_matched_apply
from .onlyutils import argser_f
from .onlyutils import argsink
from .onlyutils import download_gdrive
from .onlyutils import emplace_kv
from .onlyutils import fxfx2module
from .onlyutils import no_op
from .onlyutils import subset_state_dict
from .onlyutils import suppress_std
from .palettes import petroff_2021_cmap
from .palettes import petroff_2021_color
from .palettes import petroff_2021_cycler
from .palettes import set_petroff_2021_colors
from .palettes import set_tableau10_colors
from .palettes import tableau10_cmap
from .palettes import tableau10_color
from .palettes import tableau10_cycler
from .patches import patchify_2d
from .patches import patchify_batch
from .patches import patchify_dataset
from .reprutils import gather_model_repr
from .reprutils import model_reqgrad
from .reprutils import model_reqgrad_
from .reprutils import repr_fx_flat_adapter
from .reprutils import repr_sizes_flat_adapter
from .reprutils import store_repr_autohook
from .reprutils import store_repr_fx
from .reprutils import store_repr_hook

# Deletions (from .)
del adverutils
del actab
del autoclip
del evalutils
del lrfinder
del onlyutils
del patches
del reprutils
del filtermanip
del palettes
del mapply
