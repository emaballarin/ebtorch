#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from copy import deepcopy
from itertools import repeat
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from safe_assert import safe_assert as sassert
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from .utils import fxfx2module

# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    "ConvStem",
    "MetaAILayerNorm",
    "GRNorm",
    "ConvNeXtStem",
    "ViTStem",
]


# ──────────────────────────────────────────────────────────────────────────────
def _ntuple(n: int) -> Callable[[Union[Any, Iterable[Any]]], Tuple[Any, ...]]:
    """
    Return a function that converts a single value or an iterable of values to a tuple of `n` values.
    (from: `pytorch-image-models/timm/layers/helpers.py`)
    """

    def parse(x: Union[Any, Iterable[Any]]) -> Tuple[Any, ...]:
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple: Callable[[Union[Any, Iterable[Any]]], Tuple[Any, ...]] = _ntuple(2)
# ──────────────────────────────────────────────────────────────────────────────


class ConvStem(nn.Module):
    """
    ConvStem for Vision Transformer (ViT) models.
    (from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881;
    implementation from: moco-v3/vits.py)
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 768,
        embed_init_compr: int = 8,
        depth: int = 4,
        embed_step_expn: int = 2,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        flatten: bool = True,
        activation: Union[Callable[[Tensor], Tensor], nn.Module] = nn.ReLU(
            inplace=True
        ),
    ) -> None:
        super().__init__()

        sassert(
            not (embed_dim % embed_init_compr),
            "ConvStem only supports `embed_dim` divisible by `embed_init_compr`",
        )

        self.activation: nn.Module = fxfx2module(activation)
        self.flatten: bool = flatten

        stem: nn.ModuleList = nn.ModuleList()
        input_dim: int = in_chans
        output_dim: int = embed_dim // embed_init_compr
        for _ in range(depth):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(deepcopy(self.activation))
            input_dim, output_dim = output_dim, output_dim * embed_step_expn
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))

        self.proj: nn.Sequential = nn.Sequential(*stem)
        self.norm: nn.Module = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.proj(x)
        x: Tensor = x.flatten(2).transpose(1, 2) if self.flatten else x
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
class MetaAILayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    (from: ConvNeXt-V2/models/utils.py)
    """

    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int]],
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(normalized_shape))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(normalized_shape))
        self.eps: float = eps
        self.data_format: str = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(
                f"Data format {self.data_format} is not supported."
            )
        self.normalized_shape: Tuple[int, ...] = (
            tuple(normalized_shape)
            if isinstance(normalized_shape, Sequence)
            else (normalized_shape,)
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        # elif self.data_format == "channels_first":
        u: Tensor = x.mean(1, keepdim=True)
        s: Tensor = (x - u).pow(2).mean(1, keepdim=True)
        x: Tensor = (x - u) / torch.sqrt(s + self.eps)
        x: Tensor = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ──────────────────────────────────────────────────────────────────────────────


class GRNorm(nn.Module):
    """
    GRN (Global Response Normalization) layer
    (from: https://arxiv.org/abs/2301.00808;
    implementation from: ConvNeXt-V2/models/utils.py)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma: nn.Parameter = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta: nn.Parameter = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        gx: Tensor = torch.norm(x, p=2, dim=[1, 2], keepdim=True)
        nx: Tensor = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


# ──────────────────────────────────────────────────────────────────────────────


class ConvNeXtStem(nn.Module):
    """
    ConvNeXt stem for ConvNeXt-V2 models
    (from: https://arxiv.org/abs/2301.00808;
    implementation from: ConvNeXt-V2/models/convnextv2.py)
    """

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        out_chans: int = 96,
    ):
        super().__init__()
        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size),
            MetaAILayerNorm(out_chans, eps=1e-6, data_format="channels_first"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


# ──────────────────────────────────────────────────────────────────────────────


class ViTStem(ConvNeXtStem):
    def __init__(
        self, patch_size: int = 16, in_chans: int = 3, out_chans: int = 96
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=out_chans,
        )
