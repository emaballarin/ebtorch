#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from collections.abc import Iterable
from functools import partial
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: List[str] = ["variadic_attrs"]


# ~~ Utilities ~~ ──────────────────────────────────────────────────────────────
def _str_to_bool(s: str, onesym: bool = False) -> bool:
    osl: List[str] = ["t", "y", "1"]
    if onesym:
        return s.lower() in osl
    return s.lower() in (osl + ["true", "yes"])


def _any_to_bool(x, onesym: bool = False) -> bool:
    if isinstance(x, str):
        return _str_to_bool(x, onesym)
    return bool(x)


def _str_to_booltuple(s: str, sep: Optional[str] = None) -> Tuple[bool, ...]:
    if sep is not None:
        return tuple(map(_str_to_bool, s.split(sep)))
    return tuple(map(partial(_str_to_bool, onesym=True), [*s]))


def _any_to_booltuple(
    x: Union[str, Iterable[Union[str, bool]]], sep: Optional[str] = None
) -> Tuple[bool, ...]:
    if isinstance(x, str):
        return _str_to_booltuple(x, sep)
    return tuple(map(_any_to_bool, x))


def variadic_attrs(
    selfobj,
    varsel: Optional[Iterable[Union[str, bool]]] = None,
    insep: Optional[str] = None,
    outsep: str = "_",
):
    odict: dict = selfobj.__getstate__()
    odkeys: Tuple[str, ...] = tuple(odict.keys())
    lodk: int = len(odkeys)
    varsel: Iterable = varsel if varsel is not None else ([True] * lodk)
    bvsel: Tuple[bool, ...] = _any_to_booltuple(varsel, insep)
    strtuple: Tuple[str, ...] = tuple(
        str(odict[odkeys[i]]) if bvsel[i] else "" for i in range(lodk)
    )
    return (outsep.join(strtuple)).strip().strip(outsep)
