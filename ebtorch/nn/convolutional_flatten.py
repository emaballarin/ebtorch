#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
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
# SPDX-License-Identifier: Apache-2.0
import math

from torch import nn


class ConvolutionalFlattenLayer(nn.Module):
    """
    A better alternative to flattening for spatially-coded data!
    """

    def __init__(
        self,
        height: int,
        width: int,
        detail_size: int,
        channels_in: int,
        bias: bool = True,
        actually_flatten: bool = True,
    ):
        super(ConvolutionalFlattenLayer, self).__init__()
        channels_out: int = math.ceil(
            (channels_in * height * width)
            / ((height - detail_size) * (width - detail_size))
        )

        self.actually_flatten = actually_flatten
        self.conv = nn.Conv2d(
            in_channels=channels_in,
            out_channels=channels_out,
            kernel_size=detail_size,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        if self.actually_flatten:
            return self.conv(x).flatten(start_dim=1)
        else:
            return self.conv(x)
