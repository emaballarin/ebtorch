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
# ──────────────────────────────────────────────────────────────────────────────
from contextlib import ExitStack
from copy import deepcopy
from functools import partial
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import device as torch_device
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "model_reqgrad_",
    "model_reqgrad",
    "repr_fx_flat_adapter",
    "repr_sizes_flat_adapter",
    "store_repr_fx",
    "store_repr_hook",
    "store_repr_autohook",
    "gather_model_repr",
]
# ──────────────────────────────────────────────────────────────────────────────


def model_reqgrad_(model: Module, set_to: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = set_to


def model_reqgrad(model: Module, set_to: bool) -> Module:
    new_model = deepcopy(model)
    model_reqgrad_(model=new_model, set_to=set_to)
    return new_model


def repr_fx_flat_adapter(representation: List[Tensor]) -> Tensor:
    return torch.cat([torch.flatten(t, start_dim=1) for t in representation], dim=1)


def repr_sizes_flat_adapter(representation: List[Tensor]) -> List[int]:
    sizes: List[int] = [l := 0]
    return sizes + [l := l + t.numel() for t in representation]


def store_repr_fx(
    representation: List[Tensor],
    x: Tensor,
    device: Union[str, torch_device],
    preserve_graph: bool = False,
) -> List[Tensor]:

    with ExitStack() as stack:
        if not preserve_graph:
            stack.enter_context(torch.no_grad())

    if representation and x.shape[0] != representation[0].shape[0]:
        raise ValueError(
            "Tensor batch-size mismatch! "
            f"Representation has batch size {representation[0].shape[0]}, whereas acquired tensor has {x.shape[0]}."
        )

    if preserve_graph:
        representation.append(x.clone().to(device))
    else:
        representation.append(x.clone().detach().to(device))

    return representation


def store_repr_hook(
    representation: List[Tensor],
    mod: Module,
    inp: Tensor,
    out: Tensor,
    device: Union[str, torch_device],
    preserve_graph: bool = False,
) -> None:
    _ = mod, inp
    store_repr_fx(representation, out, device, preserve_graph)


def store_repr_autohook(
    model: Module,
    representation: List[Tensor],
    device: Union[str, torch_device],
    layers: Optional[Tuple[str, ...]] = None,
    preserve_graph: bool = False,
) -> List[RemovableHandle]:

    handles: List[RemovableHandle] = []

    for name, mod in model.named_modules():
        if layers is None or name in layers:
            handle: RemovableHandle = mod.register_forward_hook(
                partial(
                    store_repr_hook,
                    representation,
                    device=device,
                    preserve_graph=preserve_graph,
                )
            )
            handles.append(handle)

    return handles


def gather_model_repr(
    model: Module,
    xin: Tensor,
    layers: Optional[Tuple[str, ...]] = None,
    preserve_graph: bool = False,
) -> Tuple[Tensor, List[Tensor]]:

    representation: List[Tensor] = []

    handles: List[RemovableHandle] = store_repr_autohook(
        model, representation, xin.device, layers, preserve_graph
    )

    xout: Tensor = model(xin)

    handle: RemovableHandle
    for handle in handles:
        handle.remove()

    return xout, representation


# ──────────────────────────────────────────────────────────────────────────────
