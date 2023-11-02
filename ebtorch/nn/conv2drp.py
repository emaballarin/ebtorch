#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections.abc import Generator
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch.nn.common_types import _size_2_t
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.dropout import Dropout2d
from torch.nn.parameter import Parameter
from tqdm.auto import tqdm

# TODO: This is half-baked.

__all__ = []


class BatchNorm2dRP(BatchNorm2d):
    """2D batch normalization layer compatible with random patch learning. Identical to BatchNorm2d."""


class Dropout2dRP(Dropout2d):
    """2D dropout layer compatible with random patch learning. Identical to Dropout2d."""


class Conv2dRP(Conv2d):
    """2D convolutional layer compatible with random patch learning."""

    # Parent __init__
    def __init__(
        self,  # NOSONAR
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
        # Additional arguments
        patches_stride: Optional[_size_2_t] = None,
        patches_online_sampling_ratio: float = 1.0,
        patches_online_trimming_target: int = 1,
        track_inputs: bool = False,
    ) -> None:
        # Default argument substitution
        if patches_stride is None:
            patches_stride = stride

        # Parent class constructor
        super(Conv2dRP, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        # Additional member variables
        self.patches_stride: _size_2_t = patches_stride
        self.patches_online_sampling_ratio: float = patches_online_sampling_ratio
        self.patches_online_trimming_target: int = patches_online_trimming_target
        self.is_tracking_inputs: bool = track_inputs
        # self.inputs: List[torch.Tensor] = []

    # Additional private methods

    def _produce_patches(self) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(self.kernel_size, int):
                _kernel_size: Tuple[int, int] = (self.kernel_size, self.kernel_size)
            else:
                self.kernel_size: Tuple[int, int]
                _kernel_size: Tuple[int, int] = self.kernel_size

            if isinstance(self.patches_stride, int):
                _patches_stride: Tuple[int, int] = (
                    self.patches_stride,
                    self.patches_stride,
                )
            else:
                _patches_stride: Tuple[int, int] = self.patches_stride

            _inputs: torch.Tensor = torch.cat(self.inputs, dim=0)

            # Normalize inputs gathered so far, channel-wise
            _inputs = (
                _inputs - _inputs.mean(dim=(0, 2, 3), keepdim=True)
            ) / _inputs.std(dim=(0, 2, 3), keepdim=True)

            _patches: torch.Tensor = (
                _inputs.unfold(2, _kernel_size[0], _patches_stride[0])
                .unfold(3, _kernel_size[1], _patches_stride[1])
                .permute(0, 2, 3, 1, 4, 5)
                .contiguous()
            )

            _patches = _patches.view(-1, *_patches.shape[-3:]).contiguous()

            return _patches[torch.randperm(_patches.shape[0])]

    @staticmethod
    def _harvest_patches(
        randomized_patches: torch.Tensor, how_many: int
    ) -> torch.Tensor:
        # Baseline: get first k patches (assuming they are randomized)
        # Override this method to implement a better patch harvesting strategy
        with torch.no_grad():
            return randomized_patches[:how_many]

    # Public methods
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avoids tracking inputs when not compatible with forward pass
        # (e.g. in case of errors)
        to_be_returned: torch.Tensor = super(Conv2dRP, self).forward(x)
        if self.is_tracking_inputs:
            self.inputs.append(x.detach().clone())
        return to_be_returned

    # Additional public methods
    def set_tracking(self, track_inputs: bool) -> None:
        self.is_tracking_inputs: bool = track_inputs

    def clear_tracked_inputs(self) -> None:
        self.inputs: List[torch.Tensor] = []

    def patch_rp_train(self, resume_usual_after: bool = True) -> None:
        with torch.no_grad():
            self.weight.data = self._harvest_patches(
                self._produce_patches(), self.weight.shape[0]
            )
        if resume_usual_after:
            self.set_tracking(False)
            self.clear_tracked_inputs()

    def non_rp_parameters(self) -> Generator[Parameter, Any, None]:
        return (param[1] for param in self.named_parameters() if param[0] == "bias")


# Training tools for whole networks


def patch_rp_train_network(
    net: torch.nn.Module,
    data,
    resume_usual_after: bool = True,
    return_net: bool = False,
) -> Optional[torch.nn.Module]:
    _train_toggle_status: bool = net.training  # Save training value

    net.eval()  # Do not track statistics
    with torch.no_grad():  # Do not track gradients
        # Iterate over all layers in the network
        for layer in tqdm(  # type: ignore
            [
                _lay
                for _lay in net.modules()
                if isinstance(_lay, (BatchNorm2dRP, Conv2dRP, Dropout2dRP))
            ],
            leave=False,
            desc="RP-training network",
        ):
            # PREPARATION | Case 1: BatchNorm2d layer
            if isinstance(layer, BatchNorm2dRP):
                layer.running_mean = torch.zeros_like(layer.running_mean)
                layer.running_var = torch.ones_like(layer.running_var)
                layer.train()

            # PREPARATION | Case 2: Dropout2dRP layer
            elif isinstance(layer, Dropout2dRP):
                layer.eval()

            # PREPARATION | Case 3: Conv2dRP layer
            elif isinstance(layer, Conv2dRP):
                layer.clear_tracked_inputs()
                layer.eval()
                layer.set_tracking(True)

            # DATA PASS | All cases
            if isinstance(layer, (BatchNorm2dRP, Conv2dRP, Dropout2dRP)):
                for batched_datapoint in tqdm(  # type: ignore
                    data, leave=True, desc=f"RP-training layer {layer}"
                ):
                    _ = net(batched_datapoint[0])

            # POST | Case 1; Case 2
            if isinstance(layer, (BatchNorm2dRP, Dropout2dRP)):
                layer.eval()

            # POST | Case 3: Conv2dRP layer
            elif isinstance(layer, Conv2dRP):
                layer.patch_rp_train(resume_usual_after)
                layer.eval()

    # POST | Whole network
    net.train(_train_toggle_status)  # Restore training value

    # Optional return
    if return_net:
        return net
