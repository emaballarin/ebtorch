#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright 2024 Emanuele Ballarin <emanuele@ballarin.cc>
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
from collections.abc import Callable
from typing import List
from typing import Tuple
from typing import Union

import torch as th
from advertorch.attacks import Attack as ATAttack
from torchattacks.attack import Attack as TAAttack

__all__ = [
    "AdverApply",
    "TA2ATAdapter",
]


class TA2ATAdapter:
    """
    Adapt a TorchAttacks adversarial attack to the AdverTorch `perturb` API.
    """

    def __init__(self, attack: TAAttack) -> None:
        self.attack: TAAttack = attack

    def perturb(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        return self.attack(x, y)


class AdverApply:
    """
    Create fractionally-adversarially-perturbed batches for adversarial training and variations.
    """

    def __init__(
        self,
        adversaries: Union[
            List[Union[ATAttack, TA2ATAdapter]],
            Tuple[Union[ATAttack, TA2ATAdapter], ...],
        ],
        pre_process_fx: Callable = lambda x: x,
        post_process_fx: Callable = lambda x: x,
    ) -> None:
        self.adversaries = adversaries
        self.pre_process_fx = pre_process_fx
        self.post_process_fx = post_process_fx

    def __call__(
        self,
        x: List[th.Tensor],
        device,
        perturbed_fraction: float = 0.5,
        output_also_clean: bool = False,
    ) -> Tuple[th.Tensor, ...]:
        _batch_size: int = x[0].shape[0]

        _adv_number: int = len(self.adversaries)
        _atom_size: int = int((_batch_size * perturbed_fraction) // _adv_number)
        _perturbed_size: int = _atom_size * _adv_number

        _tensor_list_xclean: List[th.Tensor] = []
        _tensor_list_yclean: List[th.Tensor] = []
        _tensor_list_xpertu: List[th.Tensor] = []

        x = [self.pre_process_fx(x[0].to(device)), x[1].to(device)]

        # Clean fraction
        if _perturbed_size < _batch_size:
            _tensor_list_xclean.append(
                x[0][0 : -_perturbed_size + int(_perturbed_size == 0) * _batch_size]
            )
            _tensor_list_yclean.append(
                x[1][0 : -_perturbed_size + int(_perturbed_size == 0) * _batch_size]
            )
            _tensor_list_xpertu.append(
                x[0][0 : -_perturbed_size + int(_perturbed_size == 0) * _batch_size]
            )

        # Perturbed fraction
        if _perturbed_size > 0:
            for _adv_idx, _adversary in enumerate(self.adversaries):
                _start_idx = _batch_size - _perturbed_size + _adv_idx * _atom_size
                _end_idx = _batch_size - _perturbed_size + (_adv_idx + 1) * _atom_size

                # Clean subfraction
                _tensor_list_xclean.append(x[0][_start_idx:_end_idx].detach())
                _tensor_list_yclean.append(x[1][_start_idx:_end_idx].detach())

                # Perturbed subfraction
                _xpertu: th.Tensor = (
                    _adversary.perturb(
                        x[0][_start_idx:_end_idx],
                        x[1][_start_idx:_end_idx],
                    )
                    .reshape(x[0][_start_idx:_end_idx].shape)
                    .detach()
                )
                _tensor_list_xpertu.append(_xpertu)

        if output_also_clean:
            return (
                self.post_process_fx(th.concat(_tensor_list_xpertu, 0)).detach(),
                th.concat(_tensor_list_yclean, 0).detach(),
                self.post_process_fx(th.concat(_tensor_list_xclean, 0)).detach(),
            )
        else:
            return (
                self.post_process_fx(th.concat(_tensor_list_xpertu, 0)).detach(),
                th.concat(_tensor_list_yclean, 0).detach(),
            )
