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
# IMPORTS
import copy
from typing import List
from typing import Optional
from typing import Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

# CUSTOM TYPES
realnum = Union[float, int]


# Loss functions
@torch.jit.script
def pixelwise_bce_sum(lhs: Tensor, rhs: Tensor) -> Tensor:
    return F.binary_cross_entropy(lhs, rhs, reduction="sum")


@torch.jit.script
def pixelwise_bce_mean(lhs: Tensor, rhs: Tensor) -> Tensor:
    return F.binary_cross_entropy(lhs, rhs, reduction="mean")


@torch.jit.script
def beta_reco_bce(
    input_reco: Tensor,
    input_orig: Tensor,
    mu: Tensor,
    sigma: Tensor,
    beta: float = 0.5,
):
    kldiv = (beta * (torch.pow(mu, 2) + torch.exp(sigma) - sigma - 1)).sum()
    pwbce = pixelwise_bce_sum(input_reco, input_orig)
    return pwbce + kldiv


# Fully-Connected Block, New version
# Joint work with Davide Roznowicz (https://github.com/DavideRoznowicz)
class FCBlock(nn.Module):
    def __init__(
        self,
        in_sizes: Union[List[int], tuple],
        out_size: int,
        bias: Optional[Union[List[bool], tuple, bool]] = None,
        activation_fx: Optional[Union[List, nn.ModuleList, nn.Module]] = None,
        dropout: Optional[Union[List[Union[float, bool]], float, bool, tuple]] = None,
        batchnorm: Optional[Union[List[bool], bool, tuple]] = None,
    ) -> None:
        super().__init__()

        self.activation_fx = nn.ModuleList()

        error_uneven_size: str = (
            "The length of lists of arguments must be the same across them."
        )
        error_illegal_dropout: str = "The 'dropout' argument must be either False, a float, or an iterable of floats and/or False."

        # Default cases
        if bias is None:
            bias = [True] * len(in_sizes)
        if dropout is None:
            dropout = [False] * len(in_sizes)
        if batchnorm is None:
            batchnorm = [True] * (len(in_sizes) - 1) + [False]
        if activation_fx is None:
            for _ in range(len(in_sizes) - 1):
                self.activation_fx.append(nn.ReLU())
            self.activation_fx.append(nn.Identity())

        # Ergonomics
        if isinstance(bias, bool):
            bias = [bias] * len(in_sizes)
        if isinstance(dropout, bool):
            if not dropout:
                dropout = [False] * len(in_sizes)
            else:
                raise ValueError(error_illegal_dropout)
        elif isinstance(dropout, float) or (
            isinstance(dropout, int) and (dropout in (0, 1))
        ):
            dropout = [dropout] * len(in_sizes)
        elif not isinstance(dropout, list):
            raise ValueError(error_illegal_dropout)

        if isinstance(batchnorm, bool):
            batchnorm = [batchnorm] * len(in_sizes)

        if isinstance(activation_fx, list):
            self.activation_fx = nn.ModuleList(copy.deepcopy(activation_fx))
        elif isinstance(activation_fx, nn.Module) and not isinstance(
            activation_fx, nn.ModuleList
        ):
            for _ in enumerate(in_sizes):
                self.activation_fx.append(copy.deepcopy(activation_fx))
        elif isinstance(activation_fx, nn.ModuleList):
            self.activation_fx = copy.deepcopy(activation_fx)

        # Sanitize
        if (
            not len(in_sizes)
            == len(bias)
            == len(self.activation_fx)
            == len(dropout)
            == len(batchnorm)
        ):
            raise ValueError(error_uneven_size)

        # Start with an empty module list
        self.module_battery = nn.ModuleList(modules=None)

        for layer_idx in range(len(in_sizes) - 1):
            self.module_battery.append(
                nn.Linear(
                    in_features=in_sizes[layer_idx],
                    out_features=in_sizes[layer_idx + 1],
                    bias=bias[layer_idx],
                )
            )
            self.module_battery.append(copy.deepcopy(self.activation_fx[layer_idx]))

            if batchnorm[layer_idx]:
                self.module_battery.append(
                    nn.BatchNorm1d(num_features=in_sizes[layer_idx + 1])
                )

            if isinstance(dropout[layer_idx], bool) and dropout[layer_idx]:
                raise ValueError(error_illegal_dropout)
            if not isinstance(dropout[layer_idx], bool):
                self.module_battery.append(nn.Dropout(p=dropout[layer_idx]))

        self.module_battery.append(
            nn.Linear(in_features=in_sizes[-1], out_features=out_size, bias=bias[-1])
        )
        self.module_battery.append(copy.deepcopy(self.activation_fx[-1]))
        if batchnorm[-1]:
            self.module_battery.append(nn.BatchNorm1d(num_features=out_size))
        if isinstance(dropout[-1], bool) and dropout[-1]:
            raise ValueError(error_illegal_dropout)
        if not isinstance(dropout[-1], bool):
            self.module_battery.append(nn.Dropout(p=dropout[-1]))

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for module_idx in enumerate(self.module_battery):
            module_idx = module_idx[0]
            x = self.module_battery[module_idx](x)
        return x


# Fully-Connected Block, Legacy version
class FCBlockLegacy(nn.Module):
    def __init__(
        self,
        fin: int,
        hsizes: List[int],
        fout: int,
        hactiv,
        oactiv,
        bias: Union[bool, List[bool]] = True,
    ) -> None:
        super().__init__()
        allsizes: List[int] = [fin] + hsizes + [fout]

        # Biases for the linears below
        if not isinstance(bias, list):
            bias = [bias] * (len(allsizes) - 1)
        else:
            if len(bias) != len(allsizes) - 1:
                raise RuntimeError(
                    "If 'bias' is a list, it must have as many elements as #linears"
                )

        self.linears: nn.ModuleList = nn.ModuleList(
            [
                nn.Linear(allsizes[i], allsizes[i + 1], bias=bias[i])
                for i in range(0, len(allsizes) - 1)
            ]
        )
        self.hactiv = hactiv
        self.oactiv = oactiv

        # Address the "hactiv: list" case
        if (
            hactiv is not None
            and isinstance(hactiv, list)
            and (len(hactiv) != len(self.linears) - 1)
        ):
            raise RuntimeError(
                "If 'hactiv' is a list, it must have as many elements as (#linears - 1)"
            )

    def forward(self, x: Tensor) -> Tensor:
        idx: int
        linear: nn.Module
        for idx, linear in enumerate(self.linears):
            x: Tensor = linear(x)
            if self.hactiv is not None and idx < len(self.linears) - 1:
                if not isinstance(self.hactiv, list):
                    x: Tensor = self.hactiv(x)
                else:
                    x: Tensor = self.hactiv[idx](x)
        if self.oactiv is not None:
            x: Tensor = self.oactiv(x)
        return x


# Causal Convolutional Layer, 1D
# (cfr.: https://github.com/pytorch/pytorch/issues/1333#issuecomment-400338207)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        result = super().forward(x)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


# Reparameterizer / Sampler for (C)VAEs & co.


# Do not make static, regardless of what the linter/analyzer says... ;)
def _gauss_reparameterize_sample(
    z_mu: Tensor, z_log_var: Tensor, device: Optional[torch.DeviceObjType] = None
) -> Tensor:
    if device is None:
        device = z_mu.device
        if device != z_log_var.device:
            raise RuntimeError(
                f"Device mismatch among 'z_mu' ({device}) and 'z_log_var' ({z_log_var.device})!"
            )
    return z_mu.to(device) + torch.randn_like(z_mu).to(device) * torch.exp(
        z_log_var * 0.5
    ).to(device)


class GaussianReparameterizerSampler(nn.Module):
    def __init__(self):  # skipcq: PYL-W0235
        super().__init__()

    # Do not make static!
    def forward(self, z_mu: Tensor, z_log_var: Tensor) -> Tensor:  # skipcq: PYL-R0201
        return _gauss_reparameterize_sample(z_mu, z_log_var)


class SGRUHCell(nn.Module):

    """
    Stateful (i.e. implicit hidden state) GRU HyperCell (i.e. arbitrary
    time-order and depth) with ReadIn and ReadOut custom heads
    """

    def __init__(
        self,
        recurrent_input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: realnum = 0,
        bidirectional: bool = False,
        tbptt: Union[bool, int] = False,
        hx: Optional[Tensor] = None,
        readin_head: nn.Module = nn.Identity(),
        readout_head: nn.Module = nn.Identity(),
    ) -> None:
        super().__init__()

        # Validate tbptt (otherwise it could become impossible to catch mistakes!)
        if not (
            (isinstance(tbptt, bool) and not tbptt)
            or (isinstance(tbptt, int) and tbptt >= 0)
        ):
            raise ValueError(
                f"Parameter 'tbptt' must be either False or a positive integer. Given: {tbptt}"
            )
        self._tbptt: int = int(tbptt)  # False == 0

        # Copy and store read-heads
        self._readin: nn.Module = copy.deepcopy(readin_head)
        self._readout: nn.Module = copy.deepcopy(readout_head)

        # Instantiate GRU Cell (eventually deep, of arbitrary order)
        self._GRU: nn.GRU = nn.GRU(
            input_size=recurrent_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Copy and store given hidden state, preserving the computational graph
        if hx is not None:
            self._hx: Tensor = hx.clone()
        else:
            self._hx = hx

        # Track recurrence
        self._recurrence_idx: int = 0

    def forward(self, x: Tensor) -> Tensor:
        # Implement Truncated BPTT (if requested to do so)
        if (
            self._tbptt > 0
            and self._recurrence_idx > 0
            and self._recurrence_idx % self._tbptt == 0
        ):
            self._hx: Tensor = self._hx.detach()

        # Read input in, through the readin head
        x: Tensor = self._readin(x)

        # Perform one step of recurrent dynamics
        out: Tensor
        if self._recurrence_idx == 0 and self._hx is None:
            out, self._hx = self._GRU(input=x)
        else:
            # If self._recurrence_idx != 0, we always manage hx ourselves
            out, self._hx = self._GRU(input=x, hx=self._hx)

        self._recurrence_idx += 1

        # Spit output out, through the readout head
        out: Tensor = self._readout(out)

        # Return
        return out

    def reset_hx(self, hx: Optional[Tensor] = None) -> None:
        # Register the detachment of self._hx in PyTorch computational graph
        if self._hx is not None:
            self._hx: Tensor = self._hx.detach()
        else:
            self._hx = None

        # Re-initialize self._hx
        if hx is not None:
            self._hx: Tensor = hx.clone()
        else:
            self._hx = hx

        # Restart the recurrence index
        self._recurrence_idx: int = 0


class ArgMaxLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:  # Do not make static!
        return torch.argmax(x, dim=1)


class BinarizeLayer(nn.Module):
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold: float = threshold

    def forward(self, x: Tensor) -> Tensor:
        return (x > self.threshold).float()  # (...): th.Tensor
