#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright 2019-* Yana Hasson <yana.hasson.inria@gmail.com>
# Copyright 2019-* Linxi (Jim) Fan <jimfanspire@gmail.com>
#
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
import os
import warnings

from setuptools import find_packages
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


def check_dependencies():
    missing_dependencies = []
    for package_name in DEPENDENCY_PACKAGE_NAMES:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        warnings.warn(f"Missing dependencies: {missing_dependencies}")


DEPENDENCY_PACKAGE_NAMES = [
    "advertorch",
    "nengolib",
    "numpy",
    "requests",
    "torch",
    "torchattacks",
]

check_dependencies()


setup(
    name="ebtorch",
    version="0.7.0",
    author="Emanuele Ballarin",
    author_email="emanuele@ballarin.cc",
    url="https://github.com/emaballarin/ebtorch",
    description="Collection of PyTorch additions, extensions, utilities, uses and abuses... for personal use.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Deep Learning", "Machine Learning"],
    license="Apache-2.0",
    packages=[package for package in find_packages() if package.startswith("ebtorch")],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
