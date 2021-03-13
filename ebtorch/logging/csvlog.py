#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright (c) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
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
# SPDX-License-Identifier: Apache-2.0

# IMPORTS
from typing import Union
from pathlib import Path
import csv


class LogCSV:
    def __init__(self, file_path: Union[str, Path]) -> None:
        self.filepath: Union[str, Path] = file_path
        self.file = None
        self.isopen: bool = False
        self.writer = None
        self.buffer: list = []

    def clear_buffer(self) -> None:
        self.buffer: list = []

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

    def clear_all(self):
        self.clear_buffer()
        self.clear_file(preserve_status=True)

    def flush(self) -> None:
        if not self.isopen:
            raise RuntimeError("Cannot flush to a closed file. Open it first!")
        self.file.flush()

    def accum_buffer(self, element):
        self.buffer.append(element)

    def write_buffer(self, flush_now: bool = False, preserve_status: bool = False):
        reclose: bool = preserve_status and not self.isopen
        self.open()
        self.writer.writerow(self.buffer)
        if flush_now:
            self.flush()
        self.clear_buffer()
        if reclose:
            self.close()

    def write_list(
        self, input_list: list, flush_now: bool = False, preserve_status: bool = False
    ):
        reclose: bool = preserve_status and not self.isopen
        self.open()
        self.writer.writerow(input_list)
        if flush_now:
            self.flush()
        if reclose:
            self.close()
