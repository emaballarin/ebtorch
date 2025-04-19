#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
import bisect
import math
import warnings
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# ──────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")

# ──────────────────────────────────────────────────────────────────────────────

__all__: List[str] = ["MultiPhaseScheduler"]

# ──────────────────────────────────────────────────────────────────────────────


def _normalize_list(
    value: Optional[Union[T, Sequence[T]]],
    name: str,
    length: int,
    caster: Callable[[T], T],
    default: T,
) -> List[T]:
    """
    Casts a scalar or sequence to a list of `length`, using `caster`.
    If value is None, returns [default]*length.
    """
    if value is None:
        return [default] * length
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = [caster(v) for v in value]  # will error if wrong length
        if len(seq) != length:
            raise ValueError(f"{name} must have length {length}, got {len(seq)}")
        return seq
    return [caster(value)] * length


# ──────────────────────────────────────────────────────────────────────────────


class MultiPhaseScheduler(_LRScheduler):
    """
    Multi-phase LR scheduler with:
        1) an initial warmup (linear or cosine),
        2) zero or more cycles of (steady -> anneal),
        3) a final plateau at `final_lrs`.

    If `steady_lrs` is None, only a single warmup (if warmup_steps>0)
    or an immediate jump to `final_lrs` is used.
    """

    def __init__(
        self,
        optim: Optimizer,
        init_lrs: Union[float, Sequence[float]],
        final_lrs: Union[float, Sequence[float]],
        warmup_steps: int = 0,
        steady_lrs: Optional[Union[float, Sequence[float]]] = None,
        steady_steps: Optional[Union[int, Sequence[int]]] = None,
        anneal_steps: Optional[Union[int, Sequence[int]]] = None,
        cos_warmup: bool = False,
        cos_annealing: Optional[Union[bool, Sequence[bool]]] = None,
        verbose: bool = False,
    ) -> None:
        # normalize per-param-group lrs
        num_groups = len(optim.param_groups)
        self.init_lrs = _normalize_list(init_lrs, "init_lrs", num_groups, float, 0.0)
        self.final_lrs = _normalize_list(final_lrs, "final_lrs", num_groups, float, 0.0)

        # basic params
        self.warmup_steps: int = max(0, int(warmup_steps))
        self.cos_warmup: bool = cos_warmup
        self.verbose: bool = verbose

        # normalize steady phases
        if steady_lrs is not None:
            if (
                isinstance(steady_lrs, Sequence)
                and not isinstance(steady_lrs, (str, bytes))
                and len(steady_lrs) == 0
            ):
                raise ValueError(
                    "steady_lrs cannot be an empty list; use None to skip steady phases"
                )
            steadys = [
                float(v)
                for v in (
                    steady_lrs
                    if isinstance(steady_lrs, Sequence)
                    and not isinstance(steady_lrs, (str, bytes))
                    else [steady_lrs]
                )
            ]
        else:
            steadys = []
        num_phases = len(steadys)
        self.steady_lrs: List[float] = steadys

        # normalize related lists
        self.steady_steps = _normalize_list(
            steady_steps, "steady_steps", num_phases, int, 0
        )
        self.anneal_steps = _normalize_list(
            anneal_steps, "anneal_steps", num_phases, int, 0
        )
        self.cos_annealing = _normalize_list(
            cos_annealing, "cos_annealing", num_phases, bool, False
        )

        # Build phases
        self.phases: List[Dict[str, Any]] = []
        lengths: List[int] = []

        # Warmup
        if self.warmup_steps > 0:
            # determine end LRs per-group
            if self.steady_lrs:
                end_lrs = [self.steady_lrs[0]] * num_groups
            else:
                end_lrs = self.final_lrs.copy()
            self.phases.append(
                {
                    "type": "warmup",
                    "length": self.warmup_steps,
                    "start": self.init_lrs.copy(),
                    "end": end_lrs,
                    "cos": self.cos_warmup,
                }
            )
            lengths.append(self.warmup_steps)

        # steady and anneal
        for i, lr_plateau in enumerate(self.steady_lrs):
            if self.steady_steps[i] > 0:
                self.phases.append(
                    {
                        "type": "steady",
                        "length": self.steady_steps[i],
                        "lrs": [lr_plateau] * num_groups,
                    }
                )
                lengths.append(self.steady_steps[i])
            if self.anneal_steps[i] > 0:
                # next phase end LRs per-group
                if i + 1 < len(self.steady_lrs):
                    end_vals = [self.steady_lrs[i + 1]] * num_groups
                else:
                    end_vals = self.final_lrs.copy()
                self.phases.append(
                    {
                        "type": "anneal",
                        "length": self.anneal_steps[i],
                        "start": [lr_plateau] * num_groups,
                        "end": end_vals,
                        "cos": self.cos_annealing[i],
                    }
                )
                lengths.append(self.anneal_steps[i])

        # total lengths
        self.lengths = lengths
        self.cum_lengths: List[int] = []
        c = 0
        for L in lengths:
            c += L
            self.cum_lengths.append(c)
        self.total_length: int = c

        if self.total_length == 0:
            # no schedule
            for idx, pg in enumerate(optim.param_groups):
                pg["lr"] = self.final_lrs[idx]
            if any(i != f for i, f in zip(self.init_lrs, self.final_lrs)):
                warnings.warn(
                    "Scheduler has zero total length and init_lrs != final_lrs; "
                    "jumping to final_lrs immediately."
                )
        else:
            for idx, pg in enumerate(optim.param_groups):
                pg["lr"] = self.init_lrs[idx]

        super().__init__(optim, last_epoch=-1)

    def get_lr(self) -> List[float]:
        t = self.last_epoch

        # before start or empty schedule
        if t < 0:
            return self.init_lrs.copy()
        if self.total_length == 0 or t >= self.total_length:
            return self.final_lrs.copy()

        # find current phase
        idx = bisect.bisect_right(self.cum_lengths, t)
        phase = self.phases[idx]
        prev_cum = self.cum_lengths[idx - 1] if idx > 0 else 0
        offset = t - prev_cum
        cl = phase["length"]

        if phase["type"] == "steady":
            lrs = phase["lrs"]
        else:
            start = phase["start"]
            end = phase["end"]
            if phase["cos"]:
                factor = 0.5 * (1 + math.cos(math.pi * offset / cl))
                lrs = [e + (s - e) * factor for s, e in zip(start, end)]
            else:
                lrs = [s + (e - s) * (offset / cl) for s, e in zip(start, end)]

        if self.verbose:
            print(f"step={t}, phase_idx={idx}, type={phase['type']}, lrs={lrs}")

        return lrs
