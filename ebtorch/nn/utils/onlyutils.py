#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
# All Rights Reserved. Unless otherwise explicitly stated.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
#
# SPDX-License-Identifier: Apache-2.0
#
# Imports
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy as deepcp
from functools import partial as fpartial
from os import environ
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import requests
import torch as th
from httpx import Client
from safe_assert import safe_assert as sassert
from torch import dtype as _dtype
from torch import nn
from torch import Tensor

from ...typing import actvt
from ...typing import numlike
from ...typing import strdev

__all__ = [
    "argser_f",
    "emplace_kv",
    "download_gdrive",
    "argsink",
    "no_op",
    "subset_state_dict",
    "fxfx2module",
    "act_opclone",
    "suppress_std",
    "TelegramBotEcho",
    "stablediv",
    "hermitize",
    "randhermn",
]


# Functions
def _isnn(c):
    """Functional shorthand for `c is not None`"""
    return c is not None


def stablediv(
    num: numlike, den: numlike, eps: numlike, stabilize_both: bool = False
) -> numlike:
    """Numerically stable division of two numbers.

    Args:
        num (numlike): Numerator.
        den (numlike): Denominator.
        eps (numlike): Numerical stability factor.
        stabilize_both (bool, optional): Whether to stabilize both terms. Defaults to False.
    """
    return (num + eps * stabilize_both) / (den + eps)


def argser_f(f, arglist: Union[list, tuple, dict]):
    error_listerror = "Function arguments must be either an args tuple or a kwargs dictionary, or both in this order inside a list."

    if not isinstance(arglist, (list, tuple, dict)):
        raise TypeError(error_listerror)
    if isinstance(arglist, list):
        # if not a list, it may not have len(...) defined
        if len(arglist) > 2:
            raise ValueError(error_listerror)

    # Input is already of correct type(s):
    if isinstance(arglist, list):
        if not arglist:  # len(arglist) == 0:
            return fpartial(f)
        if len(arglist) == 2:
            return fpartial(f, *arglist[0], **arglist[1])
        else:
            if isinstance(arglist[0], tuple):
                return fpartial(f, *arglist[0])
            if isinstance(arglist[0], dict):
                return fpartial(f, **arglist[0])
    elif isinstance(arglist, tuple):
        return fpartial(f, *arglist)
    else:  # isinstance(arglist, dict)
        return fpartial(f, **arglist)


def emplace_kv(dictionary: dict, k, v) -> dict:
    """Returns input dict with added k:v pair, overwriting if k already exists"""
    return {**dictionary, k: v}


def download_gdrive(gdrive_id, fname_save):
    # https://github.com/RobustBench/robustbench/blob/1a9c24fa69363d8130f8cdf67ca3ce8a7c481aa8/robustbench/utils.py#L34
    def get_confirm_token(_response):
        for key, value in _response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(_response, _fname_save):
        chunk_size = 32768

        with open(_fname_save, "wb") as f:
            for chunk in _response.iter_content(chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print(f"Download started: path={fname_save} (gdrive_id={gdrive_id})")

    url_base = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()

    response = session.get(url_base, params={"id": gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": gdrive_id, "confirm": token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print(f"Download finished: path={fname_save} (gdrive_id={gdrive_id})")


def argsink(*args) -> None:
    """Make static analysis happy and memory lighter :)"""
    _: Tuple[Any, ...] = args
    del _


def subset_state_dict(d: dict, subset_key: str) -> dict:
    return {
        key[(len(subset_key) + 1) :]: d[key]
        for key in (key for key in d.keys() if key.startswith(subset_key))
    }


def no_op() -> None:
    """
    A function that does nothing, by design.
    """
    pass


def act_opclone(act: actvt) -> actvt:
    """
    Clone an activation function as an `nn.Module`.
    """
    return deepcp(act) if isinstance(act, nn.Module) else _FxToModule(act)


def fxfx2module(fx: Union[Callable[[Tensor], Tensor], nn.Module]) -> nn.Module:
    return fx if isinstance(fx, nn.Module) else _FxToModule(fx)


@contextmanager
def suppress_std(which: str = "all") -> None:
    if which not in ("none", "out", "err", "all"):
        raise ValueError("`which` must be either: 'none', 'out', 'err', 'all'")

    with open(os.devnull, "w") as devnull:

        if which in ("out", "all"):
            old_stdout = sys.stdout
            sys.stdout = devnull
        if which in ("err", "all"):
            old_stderr = sys.stderr
            sys.stderr = devnull

        try:
            yield  # NOSONAR
        finally:
            if which in ("out", "all"):
                sys.stdout = old_stdout
            if which in ("err", "all"):
                sys.stderr = old_stderr


def hermitize(x: Tensor) -> Tensor:
    return (x + x.conj().t()) / 2


def randhermn(
    n: int,
    dtype: Optional[_dtype] = th.cdouble,
    device: Optional[strdev] = None,
):
    return 2 * hermitize(th.randn(n, n, dtype=dtype, device=device))


# Classes
class _FxToFxobj:  # NOSONAR
    __slots__ = ("fx",)

    def __init__(self, fx: Callable[[Tensor], Tensor]):
        self.fx: Callable[[Tensor], Tensor] = fx

    def __call__(self, x: Tensor) -> Tensor:
        return self.fx(x)


class _FxToModule(nn.Module):
    def __init__(self, fx: Callable[[Tensor], Tensor]):
        super().__init__()
        self.fx: _FxToFxobj = _FxToFxobj(fx)

    def forward(self, x: Tensor) -> Tensor:
        return self.fx(x)


class TelegramBotEcho:  # NOSONAR
    __slots__: Tuple[str, str, str] = ("_url", "_jdata", "_client")

    def __init__(
        self,
        tok_var: Optional[str] = None,
        chid_var: Optional[str] = None,
        *,
        tok: Optional[str] = None,
        chid: Optional[str] = None,
    ) -> None:
        sassert(
            (_isnn(tok) ^ _isnn(tok_var)) and (_isnn(chid) ^ _isnn(chid_var)),
            "Exactly one among `tok` and `tok_var`, and exactly one among `chid` and `chid_var` must be defined.",
        )
        _tok: str = tok if _isnn(tok) else environ.get(tok_var)
        _chid: str = chid if _isnn(tok) else environ.get(chid_var)

        self._url: str = f"https://api.telegram.org/bot{_tok}/sendMessage"
        self._jdata: Dict[str, str] = {
            "chat_id": _chid,
            "text": "",
        }
        self._client = Client(http2=True)

    def send(self, msg: str) -> None:
        _ = self._client.post(url=self._url, json=emplace_kv(self._jdata, "text", msg))
