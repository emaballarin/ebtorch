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
    length: int = max(0, int(length))
    if value is None:
        return [default] * length
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq: List[T] = [caster(v) for v in value]
        if len(seq) != length:
            raise ValueError(f"{name} must have length {length}, got {len(seq)}")
        return seq
    return [caster(value)] * length


def _dilate_int(
    x: Optional[Union[int, Sequence[int]]] = None, dilation: Optional[int] = None
) -> Optional[Union[int, List[int]]]:
    """Multiply an int or sequence of ints by a dilation factor."""

    if x is None:
        return None

    if dilation is None:
        dilation = 1

    if isinstance(x, int):
        return x * dilation

    if isinstance(x, Sequence):
        try:
            return [elem * dilation for elem in x]
        except TypeError:
            raise TypeError("All elements of `x` must be ints")

    raise TypeError("`x` must be None, an int, or a Sequence[int]")


# ──────────────────────────────────────────────────────────────────────────────


class MultiPhaseScheduler(_LRScheduler):
    """Multi-phase learning rate scheduler for warmup/plateau/annealing cycles and everything in between."""

    def __init__(
        self,
        optim: Optimizer,
        init_lr: Union[float, Sequence[float]],
        final_lr: Union[float, Sequence[float]],
        warmup_steps: int = 0,
        steady_lr: Optional[Union[float, Sequence[float]]] = None,
        steady_steps: Optional[Union[int, Sequence[int]]] = None,
        anneal_steps: Optional[Union[int, Sequence[int]]] = None,
        cos_warmup: bool = False,
        cos_annealing: Optional[Union[bool, Sequence[bool]]] = None,
        step_dilation: Optional[int] = None,
        verbose: bool = False,
    ) -> None:

        warmup_steps: int = _dilate_int(warmup_steps, step_dilation)
        steady_steps: Optional[Union[int, Sequence[int]]] = _dilate_int(
            steady_steps, step_dilation
        )
        anneal_steps: Optional[Union[int, Sequence[int]]] = _dilate_int(
            anneal_steps, step_dilation
        )

        num_groups: int = len(optim.param_groups)

        self.init_lr: List[float] = _normalize_list(
            init_lr, "init_lr", num_groups, float, 0.0
        )
        self.final_lr: List[float] = _normalize_list(
            final_lr, "final_lr", num_groups, float, 0.0
        )

        self.warmup_steps: int = max(0, int(warmup_steps))
        self.cos_warmup: bool = cos_warmup
        self.verbose: bool = verbose

        if steady_lr is not None:
            if (
                isinstance(steady_lr, Sequence)
                and not isinstance(steady_lr, (str, bytes))
                and len(steady_lr) == 0
            ):
                raise ValueError(
                    "`steady_lr` cannot be an empty list; use `None` to skip steady phases"
                )
            steadys: List[float] = [
                float(v)
                for v in (
                    steady_lr
                    if isinstance(steady_lr, Sequence)
                    and not isinstance(steady_lr, (str, bytes))
                    else [steady_lr]
                )
            ]
        else:
            steadys: List[float] = []

        num_phases: int = len(steadys)

        self.steady_lr: List[float] = steadys

        self.steady_steps: List[int] = _normalize_list(
            steady_steps, "steady_steps", num_phases, int, 0
        )
        self.anneal_steps: List[int] = _normalize_list(
            anneal_steps, "anneal_steps", num_phases, int, 0
        )
        self.cos_annealing: List[int] = _normalize_list(
            cos_annealing, "cos_annealing", num_phases, bool, False
        )

        self.phases: List[Dict[str, Any]] = []
        lengths: List[int] = []

        if self.warmup_steps > 0:
            if self.steady_lr:
                end_lr: List[float] = [self.steady_lr[0]] * num_groups
            else:
                end_lr: List[float] = self.final_lr.copy()

            self.phases.append(
                {
                    "type": "warmup",
                    "length": self.warmup_steps,
                    "start": self.init_lr.copy(),
                    "end": end_lr,
                    "cos": self.cos_warmup,
                }
            )
            lengths.append(self.warmup_steps)

        for i, lr_plateau in enumerate(self.steady_lr):
            if self.steady_steps[i] > 0:
                self.phases.append(
                    {
                        "type": "steady",
                        "length": self.steady_steps[i],
                        "lr": [lr_plateau] * num_groups,
                    }
                )
                lengths.append(self.steady_steps[i])

            if self.anneal_steps[i] > 0:
                if i + 1 < len(self.steady_lr):
                    end_vals = [self.steady_lr[i + 1]] * num_groups
                else:
                    end_vals = self.final_lr.copy()

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

        self.lengths: List[int] = lengths
        self.cum_lengths: List[int] = []
        c: int = 0
        for L in lengths:
            c += L
            self.cum_lengths.append(c)
        self.total_length: int = c

        if self.total_length == 0:
            for idx, pg in enumerate(optim.param_groups):
                pg["lr"] = self.final_lr[idx]

            if any(i != f for i, f in zip(self.init_lr, self.final_lr)):
                warnings.warn(
                    "Scheduler is empty and `init_lr != final_lr`; "
                    "`init_lr` will be ignored: jumping to `final_lr` immediately."
                )
        else:
            for idx, pg in enumerate(optim.param_groups):
                pg["lr"] = self.init_lr[idx]

        super().__init__(optim, last_epoch=-1)

    def get_lr(self) -> List[float]:
        t: int = self.last_epoch

        if t < 0:
            return self.init_lr.copy()
        if self.total_length == 0 or t >= self.total_length:
            return self.final_lr.copy()

        idx: int = bisect.bisect_right(self.cum_lengths, t)
        phase: Dict[str, Any] = self.phases[idx]
        prev_cum: int = self.cum_lengths[idx - 1] if idx > 0 else 0
        offset: int = t - prev_cum
        cl: int = phase["length"]

        if phase["type"] == "steady":
            lr: List[float] = phase["lr"]
        else:
            start: List[float] = phase["start"]
            end: List[float] = phase["end"]
            if phase["cos"]:
                factor: float = 0.5 * (1 + math.cos(math.pi * offset / cl))
                lr: List[float] = [e + (s - e) * factor for s, e in zip(start, end)]
            else:
                lr: List[float] = [
                    s + (e - s) * (offset / cl) for s, e in zip(start, end)
                ]

        if self.verbose:
            print(f"step={t}, phase_idx={idx}, type={phase['type']}, lr={lr}")

        return lr
