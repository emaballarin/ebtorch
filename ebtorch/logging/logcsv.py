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
# IMPORTS
import csv
from pathlib import Path
from typing import Union

__all__ = ["LogCSV"]


class LogCSV:
    def __init__(
        self,
        file_path: Union[str, Path],
        open_now: bool = False,
        clear_now: bool = False,
    ) -> None:
        self.constants: list = []
        self.buffer: list = []
        self.filepath: Union[str, Path] = file_path

        if clear_now:
            open(self.filepath, "w").close()  # clear

        if open_now:
            self.file = open(self.filepath, "a+", newline="")
            self.isopen: bool = True
            self.writer = csv.writer(self.file, quoting=csv.QUOTE_MINIMAL)
        else:
            self.file = None
            self.isopen: bool = False
            self.writer = None

    def clear_buffer(self) -> None:
        self.buffer: list = []

    def reset_constants(self) -> None:
        self.constants: list = []

    def set_constants(self, *elements) -> None:
        self.buffer: list = list(elements)

    def open(self) -> None:
        if not self.isopen:
            self.file = open(self.filepath, "a+", newline="")
            self.isopen: bool = True
            self.writer = csv.writer(self.file, quoting=csv.QUOTE_MINIMAL)

    def close(self) -> None:
        if self.isopen:
            self.writer = None
            self.file.close()
            self.isopen: bool = False

    def clear_file(self, preserve_status: bool = True) -> None:
        reopen: bool = preserve_status and self.isopen
        self.close()
        open(self.filepath, "w").close()  # clear
        if reopen:
            self.open()

    def clear_all(self) -> None:
        self.reset_constants()
        self.clear_buffer()
        self.clear_file(preserve_status=True)

    def flush(self) -> None:
        if not self.isopen:
            raise RuntimeError("Cannot flush to a closed file. Open it first!")
        self.file.flush()

    def accum_buffer(self, *elements) -> None:
        self.buffer.extend(list(elements))

    def write_buffer(
        self,
        skip_constants: bool = False,
        flush_now: bool = False,
        preserve_status: bool = True,
    ) -> None:
        reclose: bool = preserve_status and not self.isopen
        self.open()
        if skip_constants:
            self.writer.writerow(self.buffer)
        else:
            self.writer.writerow(self.constants + self.buffer)
        if flush_now:
            self.flush()
        self.clear_buffer()
        if reclose:
            self.close()

    def write_list(
        self,
        input_list: list,
        skip_constants: bool = False,
        flush_now: bool = False,
        preserve_status: bool = True,
    ) -> None:
        reclose: bool = preserve_status and not self.isopen
        self.open()
        if skip_constants:
            self.writer.writerow(input_list)
        else:
            self.writer.writerow(self.constants + input_list)
        if flush_now:
            self.flush()
        if reclose:
            self.close()
