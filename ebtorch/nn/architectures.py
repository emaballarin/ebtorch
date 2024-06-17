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
from collections.abc import Callable
from math import copysign
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle

from .functional import silhouette_score
from .penalties import beta_gaussian_kldiv
from .utils import fxfx2module

__all__ = [
    "pixelwise_bce_sum",
    "pixelwise_bce_mean",
    "beta_reco_bce",
    "beta_reco_bce_splitout",
    "FCBlock",
    "CausalConv1d",
    "build_repeated_sequential",
    "GaussianReparameterizerSampler",
    "SGRUHCell",
    "ArgMaxLayer",
    "BinarizeLayer",
    "InnerProduct",
    "RBLinear",
    "DeepRBL",
    "ResBlock",
    "SirenSine",
    "BasicAE",
    "BasicVAE",
    "Clamp",
    "SwiGLU",
    "TupleDecouple",
    "SilhouetteScore",
    "Concatenate",
    "DuplexLinearNeck",
    "SharedDuplexLinearNeck",
    "GaussianReparameterizerSamplerLegacy",
    "lexsemble",
    "GenerAct",
]

# CUSTOM TYPES
realnum = Union[float, int]


# Ensembling functions
@torch.jit.script
def lexsemble(x: Tensor, cls_dim: int = -2, ens_dim: int = -1) -> Tensor:
    out: Tensor = torch.softmax(
        torch.exp(x).sum(dim=ens_dim), dim=cls_dim - int(copysign(1, cls_dim))
    )
    return torch.log(out / (1 - out))


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
    beta: float = 1.0,
):
    kldiv = beta_gaussian_kldiv(mu, sigma, beta)
    pwbce = pixelwise_bce_sum(input_reco, input_orig)
    return pwbce + kldiv


@torch.jit.script
def beta_reco_bce_splitout(
    input_reco: Tensor,
    input_orig: Tensor,
    mu: Tensor,
    sigma: Tensor,
    beta: float = 1.0,
):
    kldiv = beta_gaussian_kldiv(mu, sigma, beta)
    pwbce = pixelwise_bce_sum(input_reco, input_orig)
    return pwbce + kldiv, pwbce, kldiv


# Utility functions
def _rbm_mask_generator(
    size: int, ood_width: int = 0, dens: float = 1.0, rand_diag: bool = False
) -> torch.Tensor:
    """
    Generate a random band matrix mask.
    """

    # Check if arguments are valid
    if ood_width < 0:
        raise ValueError("Out-of-diagonal band width must be non-negative")
    if ood_width > size - 1:
        raise ValueError("Out-of-diagonal band width must be less than size - 1")
    if dens < 0 or dens > 1:
        raise ValueError("Out-of-diagonal band density must be in [0, 1]")

    mask: Tensor = torch.logical_or(torch.diag(torch.rand(size) <= dens), (not rand_diag) * torch.eye(size))  # type: ignore
    for i in range(ood_width):
        offset: int = i + 1
        mask: Tensor = torch.logical_or(
            mask, torch.diag(torch.rand(size - offset) <= dens, diagonal=offset)  # type: ignore
        )
        mask: Tensor = torch.logical_or(
            mask, torch.diag(torch.rand(size - offset) <= dens, diagonal=-offset)  # type: ignore
        )

    return mask


def _masked_gradient_hook_factory(
    mask: Union[Tensor, None]
) -> Callable[[Union[Tensor, None]], Union[Tensor, None]]:
    """
    Return a backward hook that masks the gradient with the given mask.
    """

    def _masked_gradient_hook(grad: Union[Tensor, None]) -> Union[Tensor, None]:
        """
        Backward hook that masks the gradient with nonlocal variable `mask`.
        """

        if grad is None or mask is None:
            return grad

        if grad.shape != mask.shape:
            raise ValueError(
                f"Gradient shape {grad.shape} is not equal to mask shape {mask.shape}"
            )

        return grad * mask.to(grad.device)

    return _masked_gradient_hook


def build_repeated_sequential(
    depth: int, rep_builder: Callable[[int], nn.Module]
) -> nn.Sequential:
    repeated: nn.Sequential = nn.Sequential()
    i: int
    for i in range(depth):
        repeated.append(rep_builder(i))
    return repeated


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
        error_illegal_dropout: str = (
            "The 'dropout' argument must be either False, a float, or an iterable of floats and/or False."
        )

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
        z_log_var * 0.5  # type: ignore
    ).to(device)


class GaussianReparameterizerSamplerLegacy(nn.Module):
    def __init__(self):  # skipcq: PYL-W0235
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, z_mu: Tensor, z_log_var: Tensor) -> Tensor:  # skipcq: PYL-R0201
        return _gauss_reparameterize_sample(z_mu, z_log_var)


class GaussianReparameterizerSampler(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, z_mu: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        return z_mu + torch.randn_like(z_mu, device=z_mu.device) * torch.exp(
            z_log_var * 0.5
        )


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

    # noinspection PyMethodMayBeStatic
    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-R0201
        return torch.argmax(x, dim=1)


class BinarizeLayer(nn.Module):
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold: float = threshold

    def forward(self, x: Tensor) -> Tensor:
        return (x > self.threshold).float()  # type: ignore


class InnerProduct(nn.Module):
    """
    Compute row-wise dot-product of two batches of tensors (vectors).
    (https://github.com/UnconsciousBias/ublib/blob/master/ublib/torch.py)
    """

    def __init__(self, dim=1):
        super(InnerProduct, self).__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.bmm(a.unsqueeze(self.dim), b.unsqueeze(self.dim + 1)).squeeze()


class Clamp(nn.Module):
    def __init__(self, cmin: float = 0.0, cmax: float = 1.0) -> None:
        super().__init__()
        self.min: float = cmin
        self.max: float = cmax

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp(self.min, self.max)


# ──────────────────────────────────────────────────────────────────────────────
class RBLinear(nn.Linear):
    """
    Random Band Linear Layer
    """

    __constants__ = [
        "features",
        "in_features",
        "out_features",
        "w_mask",
        "b_mask",
        "ood_width",
        "dens",
        "rand_diag",
        "rand_bias",
    ]

    features: int
    w_mask: Tensor
    b_mask: Union[Tensor, None]
    ood_width: int
    dens: float
    rand_diag: bool
    rand_bias: bool

    def __init__(
        self,
        features: int,
        ood_width: int = 0,
        dens: float = 1.0,
        rand_diag: bool = False,
        bias: bool = True,
        rand_bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        # Call to parent constructor
        self._is_parent_initialized = False
        super().__init__(
            in_features=features,
            out_features=features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._is_parent_initialized = True

        # Save arguments for extra_repr (only)
        self.ood_width = ood_width
        self.dens = dens
        self.rand_diag = rand_diag
        self.rand_bias = rand_bias

        # Instantiate masks
        _w_mask: Tensor = _rbm_mask_generator(
            size=features, ood_width=ood_width, dens=dens, rand_diag=rand_diag
        ).to(self.weight.device)
        self.register_buffer(name="w_mask", tensor=_w_mask, persistent=True)

        if self.bias is not None:
            if rand_bias:
                _b_mask: Tensor = (torch.rand_like(self.bias) <= dens).to(self.bias.device)  # type: ignore
            else:
                _b_mask: Tensor = torch.ones_like(self.bias, dtype=torch.bool).to(self.bias.device)  # type: ignore
        else:
            _b_mask = None  # type: ignore
        self.register_buffer(name="b_mask", tensor=_b_mask, persistent=True)

        # Mask and hook weight and bias
        self.hook_handles: List[RemovableHandle] = []
        self._mask_and_hook()

    def _mask_and_hook(self) -> None:
        # Mask weight and bias
        with torch.no_grad():
            self.weight *= self.w_mask
            if self.bias is not None:
                self.bias *= self.b_mask

        # Register hooks for masked weight and bias
        self.hook_handles.clear()
        w_handle = self.weight.register_hook(_masked_gradient_hook_factory(self.w_mask))
        self.hook_handles.append(w_handle)
        if self.bias is not None:
            b_handle = self.bias.register_hook(
                _masked_gradient_hook_factory(self.b_mask)
            )
            self.hook_handles.append(b_handle)

    def _reset_parameters(self) -> None:
        # Remove existing hooks, if any
        for handle in self.hook_handles:
            handle.remove()

        # Call to parent method
        super().reset_parameters()

        # Re-mask and re-hook weight and bias
        self._mask_and_hook()

    def reset_parameters(self) -> None:
        if self._is_parent_initialized:
            self._reset_parameters()
        else:
            super().reset_parameters()

    def _recast_masks_to_device(self) -> None:
        self.w_mask = self.w_mask.to(self.weight.device)
        if self.bias is not None:
            self.b_mask = self.b_mask.to(self.bias.device)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, ood_width={self.ood_width}, dens={self.dens}, rand_diag={self.rand_diag}, rand_bias={self.rand_bias}"


class DeepRBL(nn.Module):
    """
    Deep (homogeneous) Random Band Linear Network
    """

    def __init__(
        self,
        features: int,
        depth: int = 1,
        act_fx: Union[nn.Module, Callable] = nn.Identity(),
        act_final_fx: Optional[Union[nn.Module, Callable]] = None,
        batchnorm: bool = False,
        ood_width: int = 0,
        dens: float = 1.0,
        rand_diag: bool = False,
        bias: bool = True,
        rand_bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        # Sanitize unsanitized arguments
        if depth <= 0:
            raise ValueError("Network depth must be positive")

        # Set default arguments
        self._all_layers_internal = False
        if act_final_fx is None:
            act_final_fx = act_fx
            self._all_layers_internal = True

        # Store arguments
        self.features: int = features
        self.depth: int = depth
        self.act_fx: Union[nn.Module, Callable] = act_fx
        self.act_final_fx: Union[nn.Module, Callable] = act_final_fx
        self.batchnorm: bool = batchnorm
        self.ood_width: int = ood_width
        self.dens: float = dens
        self.rand_diag: bool = rand_diag
        self.bias: bool = bias
        self.rand_bias: bool = rand_bias
        self.device = device
        self.dtype = dtype

        # Start with an empty module list
        self.module_battery = nn.ModuleList(modules=None)

        # Build the network
        for _ in range(depth - 1):
            self._build_internal_block()
        self._build_final_block()

    def _build_rblinear(self) -> None:
        self.module_battery.append(
            RBLinear(
                features=self.features,
                ood_width=self.ood_width,
                dens=self.dens,
                rand_diag=self.rand_diag,
                bias=self.bias,
                rand_bias=self.rand_bias,
                device=self.device,
                dtype=self.dtype,
            )
        )

    def _build_internal_block(self) -> None:
        self._build_rblinear()
        self.module_battery.append(copy.deepcopy(self.act_fx))
        if self.batchnorm:
            self.module_battery.append(nn.BatchNorm1d(num_features=self.features))

    def _build_final_block(self) -> None:
        self._build_rblinear()
        self.module_battery.append(copy.deepcopy(self.act_final_fx))
        if self._all_layers_internal and self.batchnorm:
            self.module_battery.append(nn.BatchNorm1d(num_features=self.features))

    def forward(self, x: Tensor) -> Tensor:
        for module_idx in enumerate(self.module_battery):
            x = self.module_battery[module_idx[0]](x)
        return x

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()


# Residual block
class ResBlock(nn.Module):
    def __init__(
        self,
        block: Union[nn.Module, Callable[[Tensor], Tensor]],
        shortcut: Union[nn.Module, Callable[[Tensor], Tensor]] = nn.Identity(),
        postall: Union[nn.Module, Callable[[Tensor], Tensor]] = nn.Identity(),
    ) -> None:
        super().__init__()
        self.block: Union[nn.Module, Callable[[Tensor], Tensor]] = block
        self.shtct: Union[nn.Module, Callable[[Tensor], Tensor]] = shortcut
        self.postl: Union[nn.Module, Callable[[Tensor], Tensor]] = postall

    def forward(self, x: Tensor) -> Tensor:
        return self.postl(self.shtct(x) + self.block(x))


# SIREN-like Sine activation(s)
class SirenSine(nn.Module):
    def __init__(self, w0: float = 30.0, learn_w0: bool = False) -> None:
        super().__init__()
        if learn_w0:
            self.w0: nn.Parameter = nn.Parameter(torch.tensor([w0]))
        else:
            self.w0: float = w0

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * x)


# AutoEncoder scaffolds
class BasicAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        extract_z: bool = False,
    ) -> None:
        super().__init__()
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder
        self.extract_z: bool = extract_z

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        z: Tensor = self.encoder(x)
        y: Tensor = self.decoder(z)
        if self.extract_z:
            return y, z
        else:
            return y


class BasicVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        mean_neck: nn.Module,
        logvar_neck: nn.Module,
        decoder: nn.Module,
        extract_z: bool = False,
        extract_mv: bool = False,
    ) -> None:
        super().__init__()
        self.encoder: nn.Module = encoder
        self.mean_neck: nn.Module = mean_neck
        self.logvar_neck: nn.Module = logvar_neck
        self.grps: GaussianReparameterizerSampler = GaussianReparameterizerSampler()
        self.decoder: nn.Module = decoder
        self.extract_z: bool = extract_z
        self.extract_mv: bool = extract_mv

    def forward(self, x: Tensor) -> Union[
        Tensor,
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor],
    ]:
        shared: Tensor = self.encoder(x)
        mean: Tensor = self.mean_neck(shared)
        logvar: Tensor = self.logvar_neck(shared)
        z: Tensor = self.grps(mean, logvar)
        y: Tensor = self.decoder(z)

        if self.extract_z and self.extract_mv:
            return y, z, mean, logvar
        elif self.extract_z:
            return y, z
        elif self.extract_mv:
            return y, mean, logvar
        else:
            return y


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        super().__init__()

        self.w1: nn.Linear = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2: nn.Linear = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3: nn.Linear = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TupleDecouple(nn.Module):
    def __init__(self, module: nn.Module, idx: int = 0) -> None:
        super().__init__()
        self.module: nn.Module = module
        self.idx: int = idx

    def forward(self, xtuple: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return (
            *xtuple[: self.idx],
            self.module(xtuple[self.idx]),
            *xtuple[self.idx + 1 :],
        )


class SilhouetteScore(nn.Module):
    """
    Layerized computation of the Silhouette Score.
    """

    @staticmethod
    def forward(features: Tensor, labels: Tensor) -> Tensor:
        return silhouette_score(features, labels)


class Concatenate(nn.Module):
    def __init__(self, dim: int = 1, flatten: bool = False):
        super().__init__()
        self.dim: int = dim
        self.flatten: bool = flatten

    def forward(self, tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]):
        tensors = (
            [tensor.flatten(start_dim=1) for tensor in tensors]
            if self.flatten
            else tensors
        )
        return torch.cat(tensors, dim=self.dim)


class DuplexLinearNeck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.x_to_mu: nn.Linear = nn.Linear(in_dim, latent_dim)
        self.x_to_log_var: nn.Linear = nn.Linear(in_dim, latent_dim)

    def forward(
        self, xc: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cxc: torch.Tensor = torch.cat(xc, dim=1)
        return self.x_to_mu(cxc), self.x_to_log_var(cxc)


class SharedDuplexLinearNeck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.shared_layer: nn.Linear = nn.Linear(in_dim, 2 * latent_dim)

    def forward(
        self, xc: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cxc: torch.Tensor = torch.cat(xc, dim=1)
        # noinspection PyTypeChecker
        return torch.chunk(self.shared_layer(cxc), 2, dim=1)


class GenerAct(nn.Module):
    def __init__(
        self,
        act: Union[Callable[[Tensor], Tensor], nn.Module],
        subv: Optional[float] = None,
        maxv: Optional[float] = None,
        minv: Optional[float] = None,
    ):
        super().__init__()
        self.act: nn.Module = fxfx2module(act)
        self.subv: Optional[float] = subv
        self.maxv: Optional[float] = maxv
        self.minv: Optional[float] = minv

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.act(x)
        x: Tensor = x - self.subv if self.subv is not None else x
        x: Tensor = x.clamp_max(self.maxv) if self.maxv is not None else x
        x: Tensor = x.clamp_min(self.minv) if self.minv is not None else x
        return x
