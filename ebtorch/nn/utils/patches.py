#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__all__ = ["patchify_2d", "patchify_batch", "patchify_dataset"]


def patchify_2d(img: torch.Tensor, patch_size: int, stride: int = 1) -> torch.Tensor:
    """
    Extract square patches from a (multi-channel) 2D image.

    Args:
        img (torch.Tensor): 3D tensor of a 2D image (C, H, W)
        patch_size (int): square patch side length
        stride (int): horizontal/vertical stride; default: 1

    Returns:
        patches (torch.Tensor): 4D tensor of patches (N, C, patch_size, patch_size)
    """
    patches: torch.Tensor = img.unfold(1, patch_size, stride).unfold(
        2, patch_size, stride
    )
    patches: torch.Tensor = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, *patches.shape[-3:])


def patchify_batch(
    imgbatch: torch.Tensor, patch_size: int, stride: int = 1
) -> torch.Tensor:
    """
    Extract square patches from a batch of (multi-channel) 2D images.

    Args:
        imgbatch (torch.Tensor): 4D tensor of 2D images (N, C, H, W)
        patch_size (int): square patch side length
        stride (int): horizontal/vertical stride; default: 1

    Returns:
        patches (torch.Tensor): 5D tensor of patches (N, M, C, patch_size, patch_size)
    """
    patches: torch.Tensor = imgbatch.unfold(2, patch_size, stride).unfold(
        3, patch_size, stride
    )
    patches: torch.Tensor = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(imgbatch.shape[0], -1, *patches.shape[-3:])


def patchify_dataset(
    imgdata: torch.Tensor, patch_size: int, stride: int = 1
) -> torch.Tensor:
    """
    Extract square patches from a tensorized (multi-channel) 2D image dataset.

    Args:
        imgdata (torch.Tensor): 4D tensor of 2D images (N, C, H, W)
        patch_size (int): square patch side length
        stride (int): horizontal/vertical stride; default: 1

    Returns:
        patches (torch.Tensor): 4D tensor of patches (N*M, C, patch_size, patch_size)
    """
    patches = patchify_batch(imgdata, patch_size, stride).contiguous()
    return patches.view(-1, *patches.shape[-3:])
