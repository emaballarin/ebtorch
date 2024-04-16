#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2021-* Emanuele Ballarin <emanuele@ballarin.cc>
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
# Copyright (c) 2019-* Diganta Misra. All Rights Reserved. MIT Licensed.
#                      [orig. paper: https://www.bmvc2020-conference.com/assets/papers/0928.pdf;
#                       orig. code: https://github.com/digantamisra98/Mish ;
#                       license text: https://github.com/digantamisra98/Mish/blob/master/LICENSE]
#
# ==============================================================================
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0
# IMPORTS
import math
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "field_transform",
    "mish",
    "serlu",
    "smelu",
    "serf",
    "oldtranspose",
    "silhouette_score",
    "cummatmul",
    "tensor_replicate",
    "logit_to_prob",
    "bisided_thresholding",
]


# FUNCTIONS


def field_transform(
    x_input: Tensor,
    pre_sum: Union[float, Tensor] = 0.0,
    mult_div: Union[float, Tensor] = 1.0,
    post_sum: Union[float, Tensor] = 0.0,
    div_not_mul: bool = False,
) -> Tensor:
    if div_not_mul:
        return torch.add(
            input=torch.div(
                input=torch.add(input=x_input, other=pre_sum), other=mult_div
            ),
            other=post_sum,
        )
    else:
        return torch.add(
            input=torch.mul(
                input=torch.add(input=x_input, other=pre_sum), other=mult_div
            ),
            other=post_sum,
        )


@torch.jit.script
def mish(x_input: Tensor) -> Tensor:
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    return F.mish(x_input)


@torch.jit.script
def serlu(x_input: Tensor, lambd: float = 1.07862, alph: float = 2.90427) -> Tensor:
    """
    Applies the SERLU function element-wise,
    defined after [Zhang & Li, 2018]
    """
    return torch.where(  # type: ignore
        x_input >= 0.0,
        torch.mul(x_input, lambd),
        torch.mul(torch.mul(x_input, torch.exp(x_input)), lambd * alph),
    )


@torch.jit.script
def smelu(x_input: Tensor, beta: float = 2.0) -> Tensor:
    """
    Applies the SmeLU function element-wise,
    defined after [Shamir & Ling, 2022]
    """
    assert beta >= 0
    return torch.where(  # type: ignore
        torch.abs(x_input) <= beta,
        torch.div(torch.pow(torch.add(x_input, beta), 2), 4.0 * beta),
        F.relu(x_input),
    )


@torch.jit.script
def serf(x: Tensor) -> Tensor:
    """Applies the Scaled ERror Function, element-wise."""
    return torch.erf(x / math.sqrt(2.0))  # type: ignore


def oldtranspose(x: Tensor) -> Tensor:
    """
    Transpose a tensor along all dimensions, emulating x.T.

    Args:
        x: Tensor to be transposed.

    Returns:
        Transposed of x.
    """
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def silhouette_score(feats: Tensor, labels: Tensor) -> Union[float, Tensor]:  # NOSONAR
    if feats.shape[0] != labels.shape[0]:
        raise ValueError(
            f"`feats` (shape {feats.shape}) and `labels` (shape {labels.shape}) must have same length"
        )
    device, dtype = feats.device, feats.dtype
    unique_labels: Union[Tensor, Tuple[Tensor, ...]] = torch.unique(labels)
    num_samples: int = feats.shape[0]
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("The number of unique `labels` must be ∈ (1, `num_samples`)")
    scores: List[Tensor] = []
    for l_label in unique_labels:
        curr_cluster: Tensor = feats[labels == l_label]
        num_elements: int = len(curr_cluster)
        if num_elements > 1:
            intra_cluster_dists: Tensor = torch.cdist(curr_cluster, curr_cluster)
            mean_intra_dists: Tensor = torch.sum(intra_cluster_dists, dim=1) / (
                num_elements - 1
            )
            dists_to_other_clusters: List[Tensor] = []
            for other_l in unique_labels:
                if other_l != l_label:
                    other_cluster: Tensor = feats[labels == other_l]
                    inter_cluster_dists: Tensor = torch.cdist(
                        curr_cluster, other_cluster
                    )
                    mean_inter_dists: Tensor = torch.sum(inter_cluster_dists, dim=1) / (
                        len(other_cluster)
                    )
                    dists_to_other_clusters.append(mean_inter_dists)
            dists_to_other_clusters_t: Tensor = torch.stack(
                dists_to_other_clusters, dim=1
            )
            min_dists: Tensor = torch.min(dists_to_other_clusters_t, dim=1)[0]
            curr_scores: Tensor = (min_dists - mean_intra_dists) / (
                torch.maximum(min_dists, mean_intra_dists)
            )
        else:
            curr_scores: Tensor = torch.tensor([0], device=device, dtype=dtype)

        scores.append(curr_scores)

    scores_t: Tensor = torch.cat(scores, dim=0)
    if len(scores_t) != num_samples:
        raise ValueError(
            f"`scores_t` (shape {scores_t.shape}) should have same length as `feats` (shape {feats.shape})"
        )
    return torch.mean(scores_t)


def cummatmul(
    input_list: Union[List[Tensor], Tensor], tensorize: Optional[bool] = None
) -> Union[List[Tensor], Tensor]:
    tensorize = isinstance(input_list, Tensor) if tensorize is None else tensorize
    cmm_list: List[Tensor] = [input_list[0]]
    mat: Tensor
    for mat in input_list[1:]:
        cmm_list.append(torch.matmul(cmm_list[-1], mat))
    if tensorize:
        return torch.stack(cmm_list)
    else:
        return cmm_list


def tensor_replicate(x: Tensor, ntimes: int, dim: int) -> Tensor:
    return x.unsqueeze(dim).expand(*x.shape[:dim], ntimes, *x.shape[dim:])


def logit_to_prob(logit: Tensor) -> Tensor:
    return torch.exp(logit) / torch.exp(logit).sum()


def bisided_thresholding(x: Tensor, thresh_ile: float) -> Tensor:
    lq = min(thresh_ile, 1 - thresh_ile)
    return torch.where(
        x > torch.quantile(x, 1 - lq),
        torch.ones_like(x),
        torch.where(
            x < torch.quantile(x, lq), -torch.ones_like(x), torch.zeros_like(x)
        ),
    )
