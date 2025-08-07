#!/usr/bin/env python3
# ~~ NOTE ~~ ───────────────────────────────────────────────────────────────────
# This is a re-implementation of the famous "AverageMeter" class from the
# Authors of PyTorch, whose exact origin is curiously hard to trace back. This
# implementation supports windowed averaging, so as to provide a full, robust
# replacement for most sum-and-divide training/testing loop loss tracking
# mechanisms.
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from collections import deque
from collections.abc import Sequence

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = ["AverageMeter", "MultiAverageMeter"]


# ~~ Classes ~~ ────────────────────────────────────────────────────────────────
class AverageMeter:
    """
    A class to compute and store the average and sum of a series of values.
    It maintains a rolling average and sum of the most recent values up to a
    specified window size. If the window size is None, all values are stored
    indefinitely.
    It provides methods to update the values, reset the meter, and retrieve
    the current average, sum, and count of values.
    It also supports batch updates for efficiency.

    Args:
        name (str): The name of the metric.
        window_size (int | None): The maximum number of values to store.
            If None, all values are stored.
        fmt (str): The format string for displaying the value.

    Raises:
        ValueError: If `window_size` is less than 1 when specified.
    Methods:
        reset() -> None: Reset the meter, clearing all stored values.
        update(val: float, n: int = 1) -> None: Update the meter
            with a new value `val`, adding it `n` times.
        update_batch(values: Sequence[float]) -> None: Update the meter
            with a batch of values from the sequence `values`.

    String Representation:
        __str__() -> str: Return a string representation of the meter,
            showing the current value, average, and name.

    Properties:
        avg (float): The average of the values in the current window.
        sum (float): The sum of the values in the current window.
        count (int): The number of values in the current window.

    Representation:
        __repr__() -> str: Return a detailed string representation of the meter,
            including its name, window size, count of values, and average.
    """

    def __init__(self, name: str = "Metric", window_size: int | None = None, fmt: str = ":f") -> None:
        """
        Initialize the AverageMeter with a name, window size, and format string.

        Args:
            name (str): The name of the metric.
            window_size (int | None): The maximum number of values to store.
                If None, all values are stored indefinitely.
            fmt (str): The format string for displaying the value.

        Raises:
            ValueError: If `window_size` is less than 1 when specified.
        """

        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be positive or None")

        self.name: str = name
        self.window_size: int | None = window_size
        self.fmt: str = fmt
        self.values: deque[float] = deque(maxlen=window_size) if window_size else deque()
        self.val: float = 0
        self._sum: float = 0

    @property
    def avg(self) -> float:
        """
        Return the average of the values in the current window.
        """
        return self._sum / len(self.values) if self.values else 0

    @property
    def sum(self) -> float:
        """
        Return the sum of the values in the current window.
        """
        return self._sum

    @property
    def count(self) -> int:
        """
        Return the number of values in the current window.
        """
        return len(self.values)

    def reset(self) -> None:
        self.values.clear()
        self.val: float = 0
        self._sum: float = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value `val`, adding it `n` times.

        Args:
            val (float): The new value to add.
            n (int): The number of times to add the value. Defaults to 1.

        Raises:
            ValueError: If `n` is negative.
        """
        if n < 0:
            raise ValueError("n must be non-negative")

        self.val: float = val
        for _ in range(n):
            if self.window_size and len(self.values) == self.window_size:
                self._sum -= self.values[0]
            self.values.append(val)
            self._sum += val

    def update_batch(self, values: Sequence[float]) -> None:
        """
        Update the meter with a batch of values from the sequence `values`.

        Args:
            values (Sequence[float]): A sequence of values to add.

        Raises:
            ValueError: If `values` is empty.
        """
        if not values:
            return

        self.val: float = values[-1]
        for val in values:
            if self.window_size and len(self.values) == self.window_size:
                self._sum -= self.values[0]
            self.values.append(val)
            self._sum += val

    def __str__(self) -> str:
        """
        Return a string representation of the meter, showing the current value,
        average, and name.
        """
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the meter, including its name,
        window size, count of values, and average.
        """
        return (
            f"WindowedAverageMeter(name='{self.name}', "
            f"window_size={self.window_size}, "
            f"count={self.count}, avg={self.avg})"
        )


class MultiAverageMeter:
    def __init__(
        self, n: int, name: str | Sequence[str] | None = None, window_size: int | Sequence[int] | None = None
    ) -> None:
        if n < 1:
            raise ValueError("`n` must be positive")

        self.n: int = n

        if isinstance(name, str):
            self.name: list[str] = [name] * n
        elif name is None:
            self.name: list[str] = [f"Metric {i}" for i in range(n)]
        else:
            self.name: list[str] = list(name)

        if isinstance(window_size, int):
            self.window_size: list[int | None] = [window_size] * n
        elif window_size is None:
            self.window_size: list[int | None] = [None] * n
        else:
            self.window_size: list[int | None] = list(window_size)

        self.meters: list[AverageMeter] = [
            AverageMeter(name=self.name[i], window_size=self.window_size[i]) for i in range(n)
        ]
        self.history: list[list[float]] = [[] for _ in range(n)]

    def reset(self, history: bool = False) -> None:
        for meter in self.meters:
            meter.reset()
        if history:
            self.history = [[] for _ in range(self.n)]

    def update(self, *vals, n: int = 1) -> None:
        for i, val in enumerate(iterable=vals):
            self.meters[i].update(val=val, n=n)

    def record(self) -> None:
        for i, meter in enumerate(iterable=self.meters):
            self.history[i].append(meter.avg)

    def printout(self, fmt: str | Sequence[str] | None = None) -> str:
        if fmt is None:
            fmt = "{name}: {value:.8f}"
        if isinstance(fmt, str):
            fmt = [fmt] * self.n
        else:
            fmt = list(fmt)

        # Print each name, colon, avg of that variable, vertical bar
        return " | ".join(
            f"{name}: {meter.avg:{fmt[i]}}" for i, (name, meter) in enumerate(iterable=zip(self.name, self.meters))
        )

    def avg(self, which: int | Sequence[int] | None = None) -> float | tuple[float, ...]:
        """
        Return the average of meter(s) in the MultiAverageMeter.
        """
        if which is None:
            return tuple(meter.avg for meter in self.meters)
        elif isinstance(which, int):
            return self.meters[which].avg
        else:
            return tuple(self.meters[i].avg for i in which)
