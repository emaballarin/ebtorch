#!/usr/bin/env python3
# ~~ NOTE ~~ ───────────────────────────────────────────────────────────────────
# This function provides a minimal subset of PyTorch Lightning's global seeding
# utility, compatible with Python's native random module, NumPy, and PyTorch. It
# currently does not conflict with the original seeding utility in terms of
# defaults and environment variables.
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import os
import random
from warnings import warn

import numpy as np
import torch as th

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = ["seed_everything"]

# ~~ Constants ~~ ──────────────────────────────────────────────────────────────
max_seed_value: int = 4294967295
min_seed_value: int = 0


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
def seed_everything(seed: int | None = None) -> int:  # type: ignore
    """
    Function that sets the seed for pseudo-random number generators in: torch,
    numpy, and Python's random module. In addition, sets the environment
    variable `EB_GLOBAL_SEED`, which can be passed to spawned subprocesses.

    Args:
        seed: the integer value seed for global random state.
            If `None`, it will read the seed from `EB_GLOBAL_SEED` env variable.
            If ``None`` and the `EB_GLOBAL_SEED` env variable is not set, then
            the seed defaults to `0`.
    """
    if seed is None:
        env_seed: str | None = os.environ.get("EB_GLOBAL_SEED")
        if env_seed is None:
            seed: int = 0
            warn(message=f"No seed found, seed set to {seed}")
        else:
            try:
                seed: int = int(env_seed)
            except ValueError:
                seed: int = 0
                warn(message=f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")

    if not (min_seed_value <= seed <= max_seed_value):
        given_seed: int = seed
        seed: int = 0
        warn(
            message=f"{given_seed} is not in bounds: numpy accepts from{min_seed_value} to {max_seed_value}, seed set to {seed}"
        )

    os.environ["EB_GLOBAL_SEED"] = str(object=seed)
    random.seed(a=seed)
    np.random.seed(seed=seed)
    th.manual_seed(seed=seed)

    return seed
