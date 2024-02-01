#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    "convnext_stem",
    "convstem_block",
    "smallconv_featurizer",
    "MetaAILayerNorm",
]


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
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

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
