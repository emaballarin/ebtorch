#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2021-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
#                      [maintainance, adaptation, extension]
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
# Copyright (c) 2019-* Diganta Misra. All Rights Reserved. MIT Licensed.
#                      [orig. paper: https://www.bmvc2020-conference.com/assets/papers/0928.pdf;
#                       orig. code: https://github.com/digantamisra98/Mish ;
#                       license text: https://github.com/digantamisra98/Mish/blob/master/LICENSE]
#
# ==============================================================================
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0
# IMPORTS
# Type hints
from math import sqrt as math_sqrt
from typing import Union

import torch
from torch import nn

__all__ = ["Mish", "mishlayer_init"]

Mish = nn.Mish

# Adapted from Federico Andres Lois' mish_init.py GitHub Gist
# (cfr.: https://gist.github.com/redknightlois/b5d36fd2ae306cb8b3484c1e3bcce253)


def mishlayer_init(mlayer, variance: Union[float, int] = 1.0):
    """
    Initialize the weights and biases of a Layer according to the
    "Variance-based initialization method for Mish-activation layers"
    by Federico Andres Lois.

    BatchNorm and other layers endowed with running statistics are automatically
    skipped (as they should), but particular care must be put in ensuring
    that other non-Mish-activated (or non-weights-and-biases) layers are skipped
    too.


    Parameters
    ----------
    mlayer : Layer
        The Layer that will be initialized.
    variance : float, optional
        Dispersion parameter for the weights. Default: 1.0
    """

    def _calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.dim()

        if dimensions < 2:  # 1D Tensor
            return 1, 1

        if dimensions == 2:  # Linear Layer
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1

            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()

            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def _initialize_weights(tensor, _variance, _filters=1):
        fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        gain = _variance / math_sqrt(fan_in * _filters)

        with torch.no_grad():
            torch.nn.init.normal_(tensor)
            return tensor.data * gain

    def _initialize_bias(tensor, _variance):
        with torch.no_grad():
            torch.nn.init.normal_(tensor)
            return tensor.data * _variance

    if mlayer is None:
        return

    if hasattr(mlayer, "weight") and mlayer.weight is not None:
        # Explicitly skip layers with running statistics (e.g. BatchNorm)
        if hasattr(mlayer, "running_mean"):
            return

        filters = 1

        # Treat Layers with "channels" as Convolutional Layers
        if hasattr(mlayer, "in_channels"):
            filters = mlayer.in_channels

        mlayer.weight.data = _initialize_weights(
            mlayer.weight, _variance=variance, _filters=filters
        )

    if hasattr(mlayer, "bias") and mlayer.bias is not None:
        mlayer.bias.data = _initialize_bias(mlayer.bias, _variance=variance)
