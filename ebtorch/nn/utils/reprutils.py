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

# IMPORTS
from typing import Union, List, Optional, Tuple

from contextlib import ExitStack

from functools import partial

from collections import OrderedDict

from copy import deepcopy

import torch as th  # lgtm [py/import-and-import-from]
from torch import Tensor


def store_repr_fx(
    representation: Union[Tensor, None], x: Tensor, preserve_graph: bool = False
) -> Tuple[Tensor, int]:

    with ExitStack() as stack:
        if not preserve_graph:
            stack.enter_context(th.no_grad())  # It's fine!

        if not isinstance(representation, th.Tensor):
            raise ValueError(
                "Known representation is not a torch.Tensor. If you need to initialize it empty, use a 0-dimensional tensor"
            )

        if representation.shape[0] == 0:
            representation = th.tensor(
                [[] for _ in range(x.shape[0])], requires_grad=preserve_graph
            )

        elif representation.shape[0] != x.shape[0]:
            raise ValueError(
                "Tensor batch-size mismatch!"
                "Known representation has batch size {}, whereas acquired tensor has {}".format(
                    representation.shape[0], x.shape[0]
                )
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

        return ret_repr, starting_idx


def store_repr_hook(
    representation_list: List[Tensor],
    starting_indices: List[int],
    mod,
    inp: Tensor,
    out: Tensor,
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
            )

        elif representation_list[0].shape[0] != out.shape[0]:
            raise ValueError(
                "Tensor batch-size mismatch!"
                "Known representation has batch size {}, whereas acquired tensor has {}".format(
                    representation_list[0].shape[0], out.shape[0]
                )
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
        representation_list[0] = th.cat((representation_list[0], repr_additional), dim=1)


def store_repr_autohook(
    model,
    representation_list: List[Tensor],
    starting_indices: List[int],
    named_layers: Union[List[str], None] = None,
    preserve_graph: bool = False,
) -> None:

    if len(starting_indices) != 0:
        raise ValueError("starting_indices list is not empty. Use an empty list instead.")

    with ExitStack() as stack:
        if not preserve_graph:
            stack.enter_context(th.no_grad())  # It's fine!

        for name, mod in model.named_modules():

            if (
                named_layers is not None and name in named_layers
            ) or named_layers is None:
                mod.register_forward_hook(
                    partial(
                        store_repr_hook,
                        representation_list,
                        starting_indices,
                        preserve_graph=preserve_graph,
                    )
                )

def gather_model_repr(
    model,
    xin: Tensor,
    named_layers: Union[List[str], None] = None,
    preserve_graph: bool = False,
) -> Tuple[Tensor, tuple, Tensor]:

    my_repr = [th.tensor([])]
    my_sizes = []

    store_repr_autohook(model, my_repr, my_sizes, named_layers, preserve_graph)
    xout = model(xin)

    ret_sizes = deepcopy(tuple(my_sizes))
    if preserve_graph:
        ret_repr, xout = my_repr[0].clone(), xout.clone()
    else:
        ret_repr, xout = my_repr[0].clone().detach(), xout.clone().detach()

    return ret_repr, ret_sizes, xout
