#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# TODO: This file is a work in progress and is not yet functional.
# TODO: Do not expose this file to the public API until it is fully functional.
# ──────────────────────────────────────────────────────────────────────────────
from collections.abc import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    "convnext_stem",
    "convstem_block",
    "stem_blocks",
    "smallconv_featurizer",
    "MetaAILayerNorm",
]


class _ModularizedFX(nn.Module):
    def __init__(self, fx: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.fx: Callable[[Tensor], Tensor] = fx

    def forward(self, x: Tensor) -> Tensor:
        return self.fx(x)


class MetaAILayerNorm(nn.Module):
    """
    `LayerNorm` that supports two data formats: `channels_last` (default) or `channels_first`.
    The ordering of the dimensions of the the inputs `channels_last` corresponds to inputs with
    shape (`batch_size`, `height`, `width`, `channels`), whereas `channels_first` corresponds
    to inputs with shape (`batch_size`, `channels`, `height`, `width`).
    From Meta AI's ConvNeXtV2 implementation
    (https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(normalized_shape))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(normalized_shape))
        self.eps: float = eps
        self.data_format: str = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape: Tuple[int] = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u: Tensor = x.mean(1, keepdim=True)
            s: Tensor = (x - u).pow(2).mean(1, keepdim=True)
            x: Tensor = (x - u) / torch.sqrt(s + self.eps)
            x: Tensor = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def convnext_stem(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 4,
    padding: int = 0,
    normalize: bool = True,
) -> nn.Sequential:

    stem_block: nn.ModuleList = nn.ModuleList(
        [nn.Conv2d(in_channels, out_channels, kernel_size, kernel_size, padding)]
    )

    if normalize:
        stem_block.append(MetaAILayerNorm(out_channels, data_format="channels_first"))

    return nn.Sequential(*stem_block)


def convstem_block(
    in_channels: int,
    post_conv_channels: int,
    out_channels: int,
    normalize: bool = True,
) -> nn.Sequential:

    stem_block: nn.ModuleList = nn.ModuleList(
        [nn.Conv2d(in_channels, post_conv_channels, 3, 2, 1, bias=False)]
    )

    if normalize:
        stem_block.append(nn.BatchNorm2d(post_conv_channels))

    stem_block.extend([nn.ReLU(), nn.Conv2d(post_conv_channels, out_channels, 1)])

    return nn.Sequential(*stem_block)


def smallconv_featurizer(
    in_channels: int,
    out_channels: int,
    pool: bool = True,
):

    stem_block: nn.ModuleList = nn.ModuleList(
        [nn.Conv2d(in_channels, out_channels, 3), nn.ReLU()]
    )

    if pool:
        stem_block.append(nn.MaxPool2d(2))

    return nn.Sequential(*stem_block)


def stem_blocks(  # NOSONAR
    kind: str,
    channels: Union[Tuple[int, ...], List[int]],
    kernsize: Union[int, Tuple[int, ...], List[int]],
    strides: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
    normalize: Optional[Union[bool, Tuple[bool, ...], List[bool]]] = None,
    activation: Optional[
        Union[
            nn.Module,
            Callable[[Tensor], Tensor],
            nn.ModuleList,
            List[Union[nn.Module, Callable[[Tensor], Tensor]]],
            Tuple[Union[nn.Module, Callable[[Tensor], Tensor]], ...],
        ]
    ] = None,
):
    # Preparations on `kind`
    allowed_kinds: List[str] = ["ViTConv", "ViTPatch"]
    if kind not in allowed_kinds:
        raise ValueError(
            f"Invalid value {kind} in `type`. Expected one of {allowed_kinds}."
        )

    # Preparations on `channels`
    if isinstance(channels, (list, tuple)):
        for elem in channels:
            if not isinstance(elem, int) or elem <= 0:
                raise ValueError(
                    f"Invalid value {elem} in `channels`. "
                    "Expected a positive integer."
                )
    else:
        raise ValueError(
            f"Invalid type {type(channels)} for `channels`. "
            "Expected a list or tuple of integers."
        )
    if not isinstance(channels, tuple):
        channels: Tuple[int, ...] = tuple(channels)

    lcmo: int = len(channels) - 1

    # Preparations on `kernsize`
    if isinstance(kernsize, int):
        kernsize: Tuple[int, ...] = (kernsize,) * lcmo
    elif isinstance(kernsize, (list, tuple)):
        if len(kernsize) != lcmo:
            raise ValueError(
                "Length of `kernsize` must be one less than the length of `channels`."
            )
        for elem in kernsize:
            if not isinstance(elem, int) or elem <= 0:
                raise ValueError(
                    f"Invalid value {elem} in `kernsize`. "
                    "Expected a positive integer."
                )
    else:
        raise ValueError(
            f"Invalid type {type(kernsize)} for `kernsize`. "
            "Expected a list or tuple of integers."
        )
    if not isinstance(kernsize, tuple):
        kernsize: Tuple[int, ...] = tuple(kernsize)

    # Preparations on `strides`
    if strides is None:
        strides: int = 2
    if isinstance(strides, int):
        strides: Tuple[int, ...] = (strides,) * lcmo
    elif isinstance(strides, (list, tuple)):
        if len(strides) != lcmo:
            raise ValueError(
                "Length of `strides` must be one less than the length of `channels`."
            )
        for elem in strides:
            if not isinstance(elem, int) or elem <= 0:
                raise ValueError(
                    f"Invalid value {elem} in `strides`. "
                    "Expected a positive integer."
                )
    else:
        raise ValueError(
            f"Invalid type {type(strides)} for `strides`. "
            "Expected a list or tuple of integers."
        )
    if not isinstance(strides, tuple):
        strides: Tuple[int, ...] = tuple(strides)

    # Preparations on `normalize`
    if normalize is None:
        normalize: bool = True
    if isinstance(normalize, bool):
        normalize: Tuple[bool, ...] = (normalize,) * lcmo
    elif isinstance(normalize, (list, tuple)):
        if len(normalize) != lcmo:
            raise ValueError(
                "Length of `normalize` must be one less than the length of `channels`."
            )
        for elem in normalize:
            if not isinstance(elem, bool):
                raise ValueError(
                    f"Invalid value {elem} in `normalize`. " "Expected a boolean."
                )
    else:
        raise ValueError(
            f"Invalid type {type(normalize)} for `normalize`. "
            "Expected a list or tuple of booleans."
        )
    if not isinstance(normalize, tuple):
        normalize: Tuple[bool, ...] = tuple(normalize)

    # Preparations on `activation`
    if activation is None:
        activation: nn.Module = nn.ReLU()
    if isinstance(activation, nn.Module):
        activation: Tuple[nn.Module, ...] = (activation,) * lcmo
    elif isinstance(activation, Callable):
        activation: Tuple[nn.Module, ...] = (_ModularizedFX(activation),) * lcmo
    elif isinstance(activation, (list, tuple)):
        if len(activation) != lcmo:
            raise ValueError(
                "Length of `activation` must be one less than the length of `channels`."
            )
        for elem in activation:
            if not isinstance(elem, (nn.Module, Callable)):
                raise ValueError(
                    f"Invalid value {elem} in `activation`. "
                    "Expected a `Callable[[Tensor], Tensor]` or an instance of `nn.Module`."
                )
        activation: Tuple[nn.Module, ...] = tuple(
            [
                (_ModularizedFX(elem) if not isinstance(elem, nn.Module) else elem)
                for elem in activation
            ]
        )
    else:
        raise ValueError(
            f"Invalid type {type(activation)} for `activation`. "
            "Expected an nn.ModuleList, a list or tuple of `Callable[[Tensor], Tensor]`s or `nn.Module`s, \
            a Callable[[Tensor], Tensor] or an instance of `nn.Module`."
        )
    if not isinstance(activation, nn.ModuleList):
        activation: nn.ModuleList = nn.ModuleList(activation)

    # Constructing the stem blocks
    ...
