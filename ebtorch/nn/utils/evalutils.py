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
from collections.abc import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

__all__ = [
    "eval_model_on_test",
]


def eval_model_on_test(  # NOSONAR
    model: Module,
    test_data_loader: DataLoader,
    device: torch.device,
    model_is_classifier: bool = True,
    criterion_non_classifier: Optional[Callable] = None,
    extract_z_non_classifier: bool = False,
    verbose: bool = False,
) -> Union[Union[int, float], Tuple[Union[int, float], Tensor, Tensor]]:

    if not model_is_classifier and criterion_non_classifier is None:
        raise ValueError("Criterion must be provided for non-classifier models.")

    num_elem: int = 0
    if model_is_classifier:
        trackingmetric: int = 0
    else:
        trackingmetric: float = 0.0

    model.eval()

    with torch.no_grad():
        for batch_idx_e, batched_datapoint_e in tqdm(
            enumerate(test_data_loader),
            total=len(test_data_loader),
            desc="Testing batch",
            leave=False,
            disable=not verbose,
        ):

            # Explicitly type-hint `x_e` and `y_e`
            x_e: Tensor
            y_e: Tensor

            if model_is_classifier:
                x_e, y_e = batched_datapoint_e
                x_e, y_e = x_e.to(device), y_e.to(device)
                modeltarget_e: Tensor = model(x_e)
                ypred_e: Tensor = torch.argmax(modeltarget_e, dim=1, keepdim=True)
                trackingmetric += ypred_e.eq(y_e.view_as(ypred_e)).sum().item()

            else:  # not model_is_classifier
                x_e, y_e = batched_datapoint_e
                x_e: Tensor = x_e.to(device)
                modeltarget_e_tuple: Tuple[Tensor, Tensor] = model(x_e)
                modeltarget_e: Tensor = modeltarget_e_tuple[0]
                if extract_z_non_classifier:
                    z_to_cat: Tensor = modeltarget_e_tuple[1]
                    z: Tensor = (
                        torch.cat(tensors=(z, z_to_cat), dim=0)
                        if batch_idx_e > 0
                        else z_to_cat
                    )
                    y_e_to_cat: Tensor = y_e
                    y_e: Tensor = (
                        torch.cat(tensors=(y_e, y_e_to_cat), dim=0)
                        if batch_idx_e > 0
                        else y_e_to_cat
                    )
                trackingmetric += criterion_non_classifier(modeltarget_e, x_e).item()

            num_elem += x_e.shape[0]

    if extract_z_non_classifier and not model_is_classifier:
        return trackingmetric / num_elem, z, y_e
    else:
        return trackingmetric / num_elem
