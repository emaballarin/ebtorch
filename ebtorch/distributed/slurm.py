#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#
#  Copyright (c) 2020-2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ──────────────────────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: MIT
#
# ──────────────────────────────────────────────────────────────────────────────
import os
from typing import Tuple

__all__ = ["slurm_nccl_env"]


def slurm_nccl_env() -> Tuple[int, int, int, int, int, str]:
    """
    Get SLURM environment variables for NCCL-based distributed training.
    Remember to define in the SLURM script the following environment variable(s):
    ```
    export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
    ```
    """

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
    local_rank = int(rank - gpus_per_node * (rank // gpus_per_node))
    device = "cuda:" + str(local_rank)

    return rank, world_size, gpus_per_node, cpus_per_task, local_rank, device
