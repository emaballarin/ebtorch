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
import torch as th
from torch import Tensor

from typing import Union, List

from functools import partial

# ------------------------------------------------------------------------------


def store_repr_fx(representation: Union[Tensor, None], x: Tensor) -> Tensor:

    with th.no_grad():

        if not isinstance(representation, th.Tensor):
            raise ValueError(
                "Known representation is not a torch.Tensor. If you need to initialize it empty, use a 0-dimensional tensor"
            )

        elif representation.shape[0] == 0:
            representation = th.tensor(
                [[] for _ in range(x.shape[0])], requires_grad=False
            )

        elif representation.shape[0] != x.shape[0]:
            raise ValueError(
                "Tensor batch-size mismatch!"
                "Known representation has batch size {}, whereas acquired tensor has {}".format(
                    representation.shape[0], x.shape[0]
                )
            )

        return th.cat(
            (representation, th.flatten(x, start_dim=1).clone().detach()), dim=1
        )


# ------------------------------------------------------------------------------


def store_repr_fx_conditional(
    representation: Union[Tensor, None], x: Tensor, doit: bool
) -> Tensor:

    with th.no_grad():

        if doit:
            return store_repr(representation, x)


# ------------------------------------------------------------------------------


def store_repr_hook(
    representation_list: List[Tensor], mod, inp: Tensor, out: Tensor
) -> None:

    with th.no_grad():

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

        elif representation_list[0].shape[0] == 0:
            representation_list[0] = th.tensor(
                [[] for _ in range(out.shape[0])], requires_grad=False
            )

        elif representation_list[0].shape[0] != out.shape[0]:
            raise ValueError(
                "Tensor batch-size mismatch!"
                "Known representation has batch size {}, whereas acquired tensor has {}".format(
                    representation_list[0].shape[0], out.shape[0]
                )
            )

        representation_list[0] = th.cat(
            (representation_list[0], th.flatten(out, start_dim=1).clone().detach()),
            dim=1,
        )


# ------------------------------------------------------------------------------


def store_repr_autohook(
    model, named_layers: List[str], representation_list: List[Tensor]
) -> None:

    with th.no_grad():

        for name, mod in model.named_modules():
            if name in named_layers:
                mod.register_forward_hook(partial(store_repr_hook, representation_list))
