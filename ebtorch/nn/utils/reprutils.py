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
from contextlib import ExitStack
from copy import deepcopy
from functools import partial
from typing import List
from typing import Optional
from typing import Tuple

import torch as th
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


def store_repr_fx(
    representation: Tensor,
    x: Tensor,
    device: str,
    preserve_graph: bool = False,
) -> Tuple[Tensor, int]:
    with ExitStack() as stack:
        if not preserve_graph:
            stack.enter_context(th.no_grad())  # It's fine!

        if not isinstance(representation, th.Tensor):
            raise ValueError(
                "Representation is not a tensor. If you need to initialize it empty, use a 0-dimensional one"
            )

        if representation.shape[0] == 0:
            representation = th.tensor(
                [[] for _ in range(x.shape[0])], requires_grad=preserve_graph
            ).to(device)

        elif representation.shape[0] != x.shape[0]:
            raise ValueError(
                "Tensor batch-size mismatch!"
                f"Representation has batch size {representation.shape[0]}, whereas acquired tensor has {x.shape[0]}"
            )

        if preserve_graph:
            repr_additional = th.flatten(x, start_dim=1).clone()
        else:
            repr_additional = th.flatten(x, start_dim=1).clone().detach()

        try:
            starting_idx = representation.shape[1]
        except IndexError:
            starting_idx = 0

        new_repr = th.cat((representation, repr_additional), dim=1)

        return new_repr, starting_idx


def store_repr_hook(
    representation_list: List[Tensor],
    starting_indices: List[int],
    mod: Module,
    inp: Tensor,
    out: Tensor,
    device: str,
    preserve_graph: bool = False,
) -> None:
    with ExitStack() as stack:
        if not preserve_graph:
            stack.enter_context(th.no_grad())  # It's fine!

        _ = mod, inp  # Fake-gather

        if (
            representation_list is None
            or len(representation_list) != 1
            or not isinstance(representation_list[0], th.Tensor)
        ):
            raise ValueError(
                "representation_list is invalid. It should have only one torch.Tensor element."
                "If you need to initialize it empty, use a 0-dimensional tensor as the only element."
            )

        if representation_list[0].shape[0] == 0:
            representation_list[0] = th.tensor(
                [[] for _ in range(out.shape[0])], requires_grad=preserve_graph
            ).to(device)

        elif representation_list[0].shape[0] != out.shape[0]:
            raise ValueError(
                "Tensor batch-size mismatch!"
                f"Representation has batch size {representation_list[0].shape[0]}, whereas acquired tensor has {out.shape[0]}"
            )

        if preserve_graph:
            repr_additional = th.flatten(out, start_dim=1).clone()
        else:
            repr_additional = th.flatten(out, start_dim=1).clone().detach()

        try:
            starting_idx = representation_list[0].shape[1]
        except IndexError:
            starting_idx = 0

        starting_indices.append(starting_idx)
        representation_list[0] = th.cat(
            (representation_list[0], repr_additional), dim=1
        )


def store_repr_autohook(
    model: Module,
    representation_list: List[Tensor],
    starting_indices: List[int],
    device: str,
    named_layers: Optional[Tuple[str, ...]] = None,
    preserve_graph: bool = False,
) -> List[RemovableHandle]:
    if starting_indices:
        raise ValueError(
            "starting_indices list is not empty. Use an empty list instead."
        )

    with ExitStack() as stack:
        if not preserve_graph:
            stack.enter_context(th.no_grad())  # It's fine!

        handles: List[RemovableHandle] = []

        name: str
        mod: Module
        for name, mod in model.named_modules():
            if (
                named_layers is not None and name in named_layers
            ) or named_layers is None:
                handle: RemovableHandle = mod.register_forward_hook(
                    partial(
                        store_repr_hook,
                        representation_list,
                        starting_indices,
                        device=device,
                        preserve_graph=preserve_graph,
                    )
                )
                handles.append(handle)
    return handles


def gather_model_repr(
    model: Module,
    xin: Tensor,
    named_layers: Optional[Tuple[str, ...]] = None,
    preserve_graph: bool = False,
) -> Tuple[Tensor, Tensor, Tuple[int]]:
    my_repr: List[Tensor] = [th.tensor([]).to(xin.device)]
    my_sizes: List[int] = []

    handles: List[RemovableHandle] = store_repr_autohook(
        model, my_repr, my_sizes, xin.device, named_layers, preserve_graph
    )
    xout: Tensor = model(xin)

    ret_sizes: Tuple[int] = deepcopy(tuple(my_sizes))

    if preserve_graph:
        ret_repr, xout = my_repr[0].clone(), xout.clone()
    else:
        ret_repr, xout = my_repr[0].clone().detach(), xout.clone().detach()

    handle: RemovableHandle
    for handle in handles:
        handle.remove()

    return xout, ret_repr, ret_sizes


def model_reqgrad_(model: Module, set_to: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = set_to


def model_reqgrad(model: Module, set_to: bool) -> Module:
    new_model = deepcopy(model)
    model_reqgrad_(model=new_model, set_to=set_to)
    return new_model
