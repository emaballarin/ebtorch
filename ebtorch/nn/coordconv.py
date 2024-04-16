#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc>
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
# Copyright (c) 2018-* Chao Wen (walsvid). All Rights Reserved. MIT Licensed.
#                      [orig. paper: https://arxiv.org/abs/1807.03247;
#                       orig. code: https://github.com/uber-research/CoordConv;
#                       deriv. code: https://github.com/walsvid/CoordConv;
#                       license text: https://github.com/walsvid/CoordConv/blob/master/LICENSE]
#
# ==============================================================================
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0
from typing import List
from typing import TypeVar
from typing import Union

import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from torch.nn.common_types import _size_1_t as th_size_1_t
from torch.nn.common_types import _size_2_t as th_size_2_t
from torch.nn.common_types import _size_3_t as th_size_3_t

ACM = TypeVar("ACM", bound="AddCoords")
CCM1D = TypeVar("CCM1D", bound="CoordConv1d")
CCM2D = TypeVar("CCM2D", bound="CoordConv2d")
CCM3D = TypeVar("CCM3D", bound="CoordConv3d")

__all__: List[str] = [
    "CoordConv1d",
    "CoordConv2d",
    "CoordConv3d",
]

_error_rank_in_123 = "CoordConv is supported only for `rank` in (1, 2, 3)."


class AddCoords(nn.Module):
    """
    AddCoords Layer, as implemented in the original CoordConv paper and code
    """

    def __init__(self: ACM, rank: int, with_r: bool = False) -> None:
        # __init__ arguments validation
        if rank not in (1, 2, 3):
            raise ValueError(_error_rank_in_123 + f" Got {rank}.")

        # Actual initialization
        super(AddCoords, self).__init__()
        self.rank: int = rank
        self.with_r: bool = with_r

    def forward(self: ACM, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param input_tensor: shape (N, C_in, H, [W], [D])
        :return:
        """
        if self.rank == 1:
            batch_size_shape, _, dim_x = input_tensor.shape
            xx_range: torch.Tensor = torch.arange(
                dim_x, dtype=torch.int32, device=input_tensor.device
            )
            xx_channel: torch.Tensor = (xx_range[None, None, :]).to(input_tensor.device)

            xx_channel: torch.Tensor = xx_channel.float() / (dim_x - 1)
            xx_channel: torch.Tensor = xx_channel * torch.tensor(2).to(
                input_tensor.device
            ) - torch.tensor(1).to(input_tensor.device)
            xx_channel: torch.Tensor = xx_channel.repeat(batch_size_shape, 1, 1)

            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr: torch.Tensor = torch.sqrt(
                    torch.pow(
                        xx_channel - torch.tensor(0.5).to(input_tensor.device),
                        torch.tensor(2).to(input_tensor.device),
                    )
                )
                out: torch.Tensor = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, _, dim_y, dim_x = input_tensor.shape
            xx_ones: torch.Tensor = torch.ones(
                [1, 1, 1, dim_x], dtype=torch.int32, device=input_tensor.device
            )
            yy_ones: torch.Tensor = torch.ones(
                [1, 1, 1, dim_y], dtype=torch.int32, device=input_tensor.device
            )

            xx_range: torch.Tensor = torch.arange(
                dim_y, dtype=torch.int32, device=input_tensor.device
            )
            yy_range: torch.Tensor = torch.arange(
                dim_x, dtype=torch.int32, device=input_tensor.device
            )
            xx_range: torch.Tensor = xx_range[None, None, :, None].to(
                input_tensor.device
            )
            yy_range: torch.Tensor = yy_range[None, None, :, None].to(
                input_tensor.device
            )

            xx_channel: torch.Tensor = torch.matmul(xx_range, xx_ones)
            yy_channel: torch.Tensor = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel: torch.Tensor = yy_channel.permute(0, 1, 3, 2)

            xx_channel: torch.Tensor = xx_channel.float() / (dim_y - 1)
            yy_channel: torch.Tensor = yy_channel.float() / (dim_x - 1)

            xx_channel: torch.Tensor = xx_channel * torch.tensor(2).to(
                input_tensor.device
            ) - torch.tensor(1).to(input_tensor.device)
            yy_channel: torch.Tensor = yy_channel * torch.tensor(2).to(
                input_tensor.device
            ) - torch.tensor(1).to(input_tensor.device)

            xx_channel: torch.Tensor = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel: torch.Tensor = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            out: torch.Tensor = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr: torch.Tensor = torch.sqrt(
                    torch.pow(xx_channel - torch.tensor(0.5).to(input_tensor.device), 2)
                    + torch.pow(
                        yy_channel - torch.tensor(0.5).to(input_tensor.device), 2
                    )
                )
                out: torch.Tensor = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, _, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones: torch.Tensor = torch.ones(
                [1, 1, 1, 1, dim_x], dtype=torch.int32, device=input_tensor.device
            )
            yy_ones: torch.Tensor = torch.ones(
                [1, 1, 1, 1, dim_y], dtype=torch.int32, device=input_tensor.device
            )
            zz_ones: torch.Tensor = torch.ones(
                [1, 1, 1, 1, dim_z], dtype=torch.int32, device=input_tensor.device
            )

            xy_range: torch.Tensor = torch.arange(
                dim_y, dtype=torch.int32, device=input_tensor.device
            )
            yz_range: torch.Tensor = torch.arange(
                dim_z, dtype=torch.int32, device=input_tensor.device
            )
            zx_range: torch.Tensor = torch.arange(
                dim_x, dtype=torch.int32, device=input_tensor.device
            )

            xy_range: torch.Tensor = xy_range[None, None, None, :, None].to(
                input_tensor.device
            )
            yz_range: torch.Tensor = yz_range[None, None, None, :, None].to(
                input_tensor.device
            )
            zx_range: torch.Tensor = zx_range[None, None, None, :, None].to(
                input_tensor.device
            )

            xy_channel: torch.Tensor = torch.matmul(xy_range, xx_ones)
            xx_channel: torch.Tensor = torch.cat(
                [
                    xy_channel + torch.tensor(i).to(input_tensor.device)
                    for i in range(dim_z)
                ],
                dim=2,
            )
            xx_channel: torch.Tensor = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            yz_channel: torch.Tensor = torch.matmul(yz_range, yy_ones)
            yz_channel: torch.Tensor = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel: torch.Tensor = torch.cat(
                [
                    yz_channel + torch.tensor(i).to(input_tensor.device)
                    for i in range(dim_x)
                ],
                dim=4,
            )
            yy_channel: torch.Tensor = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            zx_channel: torch.Tensor = torch.matmul(zx_range, zz_ones)
            zx_channel: torch.Tensor = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel: torch.Tensor = torch.cat(
                [
                    zx_channel + torch.tensor(i).to(input_tensor.device)
                    for i in range(dim_y)
                ],
                dim=3,
            )
            zz_channel: torch.Tensor = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            out: torch.Tensor = torch.cat(
                [input_tensor, xx_channel, yy_channel, zz_channel], dim=1
            )

            if self.with_r:
                rr: torch.Tensor = torch.sqrt(
                    torch.pow(xx_channel - torch.tensor(0.5).to(input_tensor.device), 2)
                    + torch.pow(
                        yy_channel - torch.tensor(0.5).to(input_tensor.device), 2
                    )
                    + torch.pow(
                        zz_channel - torch.tensor(0.5).to(input_tensor.device), 2
                    )
                )
                out: torch.Tensor = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError(_error_rank_in_123 + f" Got {self.rank}.")

        return out


class CoordConv1d(conv.Conv1d):
    def __init__(
        self: CCM1D,
        in_channels: int,
        out_channels: int,
        kernel_size: th_size_1_t,
        stride: th_size_1_t = 1,
        padding: Union[str, th_size_1_t] = 0,
        dilation: th_size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super(CoordConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.rank: int = 1
        self.addcoords: ACM = AddCoords(self.rank, with_r)
        self.conv: nn.Module = nn.Conv1d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def forward(self: CCM1D, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        input_tensor_shape: (N, C_in, H)
        output_tensor_shape: (N, C_out, H_out）
        :return: CoordConv1d result in 1D
        """
        out: torch.Tensor = self.addcoords(input_tensor)
        out: torch.Tensor = self.conv(out)

        return out


class CoordConv2d(conv.Conv2d):
    def __init__(
        self: CCM2D,
        in_channels: int,
        out_channels: int,
        kernel_size: th_size_2_t,
        stride: th_size_2_t = 1,
        padding: Union[str, th_size_2_t] = 0,
        dilation: th_size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super(CoordConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.rank: int = 2
        self.addcoords: ACM = AddCoords(self.rank, with_r)
        self.conv: nn.Module = nn.Conv2d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def forward(self: CCM2D, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        input_tensor_shape: (N, C_in, H, W)
        output_tensor_shape: (N, C_out, H_out, W_out）
        :return: CoordConv2d result in 2D
        """
        out: torch.Tensor = self.addcoords(input_tensor)
        out: torch.Tensor = self.conv(out)

        return out


class CoordConv3d(conv.Conv3d):
    def __init__(
        self: CCM3D,
        in_channels: int,
        out_channels: int,
        kernel_size: th_size_3_t,
        stride: th_size_3_t = 1,
        padding: Union[str, th_size_3_t] = 0,
        dilation: th_size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super(CoordConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.rank: int = 3
        self.addcoords: ACM = AddCoords(self.rank, with_r)
        self.conv: nn.Module = nn.Conv3d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def forward(self: CCM3D, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        input_tensor_shape: (N, C_in, H, W, D)
        output_tensor_shape: N, C_out, H_out, W_out, D_out）
        :return: CoordConv2d result in 3D
        """
        out: torch.Tensor = self.addcoords(input_tensor)
        out: torch.Tensor = self.conv(out)

        return out
