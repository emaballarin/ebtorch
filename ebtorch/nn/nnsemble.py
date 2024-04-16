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
# Imports
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
from typing import Optional
from typing import TypeVar
from typing import Union

import torch as th
from functorch import combine_state_for_ensemble
from functorch import vmap
from torch import nn as thnn
from torch.optim.swa_utils import AveragedModel

__all__ = ["NNEnsemble"]


# Custom types
T = TypeVar("T", bound="NNEnsemble")


# Functions
def _tensor_no_op(
    x: Union[th.Tensor, Sequence[th.Tensor]]
) -> Union[th.Tensor, Sequence[th.Tensor]]:
    return x


def _warn_empty_models() -> None:
    warnings.warn(
        "No models were added to the ensemble but you requested a state update."
        "This is probably not what you want. State was not updated.",
        UserWarning,
    )


def _warn_swa_aggregation() -> None:
    warnings.warn(
        "swa_ensemble is True, but aggregation is not None."
        "Aggregation will be ignored.",
        UserWarning,
    )


# Classes
class NNEnsemble(thnn.Module):
    def __init__(
        self: T,
        models: Optional[Iterable[thnn.Module]] = None,
        aggregation: Optional[
            Union[thnn.Module, Callable[[th.Tensor], th.Tensor]]
        ] = None,
        swa_ensemble: bool = False,
    ) -> None:
        super().__init__()

        # Handle models
        if models is None:
            models: Iterable[thnn.Module] = []
        self.models: thnn.ModuleList = thnn.ModuleList(models)

        # Handle aggregation function
        if aggregation is None:
            self.aggregation: Callable[[th.Tensor], th.Tensor] = _tensor_no_op
        elif isinstance(aggregation, thnn.Module):
            self.aggregation: thnn.Module = aggregation
        elif callable(aggregation):
            self.aggregation: Callable[[th.Tensor], th.Tensor] = aggregation
        else:
            raise TypeError(
                "aggregation (if any) must be either a torch.nn.Module or a Callable[[th.Tensor], th.Tensor]"
            )

        # Handle SWA
        self.swa_ensemble: bool = swa_ensemble
        if swa_ensemble and (aggregation is not None):
            _warn_swa_aggregation()

        # Members required by SWA only
        self.swa_model: Optional[AveragedModel] = None
        self.bn_loader: Optional[Iterable] = None

        # Members required by "pure" ensembles only
        self.vfmodel: Optional[Any] = None
        self.vparams: Optional[Any] = None
        self.vbuffers: Optional[Any] = None
        self.vensemble: Optional[Any] = None

        # Test/Train safeguards
        self.notify_train_eval_changes_is_armed: bool = False
        self.notify_train_eval_changes_is_hardened: bool = False

    # Internal methods required by SWA only
    def _update_swa_model_nofail(self: T) -> None:
        if not self.models:
            return
        self.swa_model = AveragedModel(self.models[0])
        for m in self.models:
            self.swa_model.update_parameters(m)

    def _update_swa_bn_nofail(self: T) -> None:
        if self.swa_model is None:
            return
        if self.bn_loader is not None:
            th.optim.swa_utils.update_bn(self.bn_loader, self.swa_model)
        else:
            warnings.warn(
                "No batch normalization loader has been provided."
                "Batch normalization statistics will not be updated.",
                UserWarning,
            )

    def _update_swa_everything_nofail(self: T) -> None:
        self._update_swa_model_nofail()
        self._update_swa_bn_nofail()

    # Internal methods required by "pure" ensembles only
    def _update_vensemble_nofail(self: T) -> None:
        if not self.models:
            return
        self.vfmodel, self.vparams, self.vbuffers = combine_state_for_ensemble(
            self.models
        )
        self.vensemble = vmap(func=self.vfmodel, in_dims=(0, 0, None))

    # Internal methods for both SWA and "pure" ensembles
    def _update_state_nofail(self: T, force_both: bool = False) -> None:
        if self.swa_ensemble or force_both:
            self._update_swa_everything_nofail()
        if (not self.swa_ensemble) or force_both:
            self._update_vensemble_nofail()

    def _check_ensemble_is_usable(self: T, force_both: bool = False) -> None:
        if (self.swa_ensemble or force_both) and self.swa_model is None:
            raise RuntimeError(
                "No SWA model has been created. Did you forget to call update_state()?"
            )
        if ((not self.swa_ensemble) or force_both) and self.vensemble is None:
            raise RuntimeError(
                "No ensemble has been created. Did you forget to call update_state()?"
            )

    # Public methods
    def reset_state(self: T) -> T:
        self.models: thnn.ModuleList = thnn.ModuleList([])
        self.swa_model: Optional[AveragedModel] = None
        self.vfmodel: Optional[Any] = None
        self.vparams: Optional[Any] = None
        self.vbuffers: Optional[Any] = None
        self.vensemble: Optional[Any] = None
        return self

    def update_state(self: T, force_both: bool = False) -> T:
        if not self.models:
            _warn_empty_models()
            return self
        self._update_state_nofail(force_both)
        return self

    def append_model(self: T, model: thnn.Module, update_state: bool = True) -> T:
        self.models.append(model)
        if update_state:
            self._update_state_nofail()
        return self

    def toggle_swa_ensemble(
        self: T,
        swa_ensemble: bool,
        update_state: bool = True,
        update_dormant_state: bool = False,
    ) -> T:
        self.swa_ensemble = swa_ensemble
        if swa_ensemble and (  # pylint: disable=W0143
            self.aggregation != _tensor_no_op  # pylint: disable=W0143
        ):  # pylint: disable=W0143
            _warn_swa_aggregation()
        if update_state:
            self._update_state_nofail(force_both=update_dormant_state)
        return self

    def set_aggregation(
        self,
        aggregation: Union[thnn.Module, Callable[[th.Tensor], th.Tensor], None],
    ):
        if aggregation is None:
            self.aggregation: Callable[[th.Tensor], th.Tensor] = _tensor_no_op
        elif isinstance(aggregation, thnn.Module):
            self.aggregation: thnn.Module = aggregation
        elif callable(aggregation):
            self.aggregation: Callable[[th.Tensor], th.Tensor] = aggregation
        else:
            raise TypeError(
                "aggregation (if any) must be either a torch.nn.Module or a Callable[[th.Tensor], th.Tensor]"
            )

    def set_bn_loader(self: T, bn_loader: Iterable):
        if not self.swa_ensemble:
            warnings.warn(
                "Batch normalization loader has been provided, but SWA is not enabled."
                "Remembering the loader for when SWA is enabled, but not using it for now.",
                UserWarning,
            )
        self.bn_loader: Iterable = bn_loader

    def notify_train_eval_changes(
        self: T, armed: bool = True, hardened: bool = False
    ) -> T:
        self.notify_train_eval_changes_is_armed: bool = armed
        self.notify_train_eval_changes_is_hardened: bool = hardened
        return self

    def train(
        self: T, mode: bool = True, override_safetynet: bool = False
    ) -> thnn.Module:
        if self.notify_train_eval_changes_is_armed and self.training != mode:
            if self.notify_train_eval_changes_is_hardened:
                raise RuntimeError(
                    f"Change of training mode from {self.training} to {mode} detected,"
                    "but denied since the model has been hardened."
                )
            warnings.warn(
                f"Change of training mode from {self.training} to {mode} detected."
                "Allowing it.",
                UserWarning,
            )
        if self.swa_ensemble and mode:
            if override_safetynet:
                warnings.warn(
                    "Training the ensemble is not compatible with SWA."
                    "Allowing mode change, but this may cause unexpected behaviour!",
                    UserWarning,
                )
            else:
                warnings.warn(
                    "Training the ensemble is not compatible with SWA."
                    "Not doing anything. Use `override_safetynet=True` to force mode change anyway.",
                    UserWarning,
                )
                mode = False
        return super().train(mode=mode)

    def eval(self: T, override_safetynet: bool = False) -> thnn.Module:
        return self.train(mode=False, override_safetynet=override_safetynet)

    def forward(self: T, x: th.Tensor) -> th.Tensor:
        if self.swa_ensemble:
            return self.swa_model(x)
        else:
            return self.aggregation(self.vensemble(self.vparams, self.vbuffers, x))
