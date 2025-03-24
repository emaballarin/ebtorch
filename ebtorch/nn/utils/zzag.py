#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from typing import List
from typing import Optional

import numpy as np
import torch as th
from torch import Tensor

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: List[str] = ["zigzag", "zigzag_indices"]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────


def zigzag_indices(n: int) -> List[int]:
    """
    Compute zigzag indices for a sequence of length n.
    Returns indices: 0, n-1, 1, n-2, ...
    """
    indices: List[int] = []
    for i in range((n + 1) // 2):
        indices.append(i)
        if i != n - 1 - i:
            indices.append(n - 1 - i)
    return indices


def zigzag(data, dim: Optional[int] = None):
    """
    Reorder the elements of the input in a zigzag pattern.

    For different input types:
        - Non-materialized iterable (e.g. generator): returns a generator.
        - Materialized iterable (e.g. list, tuple): returns an object of the same type.
        - PyTorch tensor: returns a tensor with slices along `dim` in zigzag order.
        - NumPy array: returns an array with slices along `dim` in zigzag order.

    Parameters:
        data: The input data.
        dim: For tensor/array inputs, the dimension/axis along which to reorder.

    Returns:
        Reordered data in zigzag order.
    """

    if isinstance(data, Tensor):
        if dim is None:
            raise ValueError("For a PyTorch tensor, `dim` must be specified.")
        n: int = data.size(dim)
        indices: List[int] = zigzag_indices(n)
        idx_tensor: Tensor = th.tensor(indices, dtype=th.long, device=data.device)
        return data.index_select(dim, idx_tensor)

    elif isinstance(data, np.ndarray):
        if dim is None:
            raise ValueError("For a NumPy array, `dim` must be specified.")
        n: int = data.shape[dim]
        indices: List[int] = zigzag_indices(n)
        return np.take(data, indices, axis=dim)

    elif isinstance(data, (list, tuple)):
        n = len(data)
        indices: List[int] = zigzag_indices(n)
        if isinstance(data, list):
            return [data[i] for i in indices]
        else:
            return tuple(data[i] for i in indices)

    elif hasattr(data, "__iter__"):
        data_list: list = list(data)
        n: int = len(data_list)
        indices: List[int] = zigzag_indices(n)
        return (data_list[i] for i in indices)

    else:
        raise TypeError("Unsupported input type for zigzag ordering.")
