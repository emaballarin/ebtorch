#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#  Copyright (c) 2020-2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
#
# SPDX-License-Identifier: MIT
#
# ------------------------------------------------------------------------------
import os
import warnings

from setuptools import find_packages
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


def check_dependencies(dependencies: list[str]):
    missing_dependencies: list[str] = []
    package_name: str
    for package_name in dependencies:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        warnings.warn(f"Missing dependencies: {missing_dependencies}")


DEPENDENCY_PACKAGE_NAMES: list[str] = [
    "advertorch",
    "numpy",
    "requests",
    "torch",
    "torchattacks",
    "torchvision",
    "tqdm",
]

check_dependencies(DEPENDENCY_PACKAGE_NAMES)

PACKAGENAME: str = "ebtorch"


setup(
    name=PACKAGENAME,
    version="0.17.4",
    author="Emanuele Ballarin",
    author_email="emanuele@ballarin.cc",
    url="https://github.com/emaballarin/ebtorch",
    description="Collection of PyTorch additions, extensions, utilities, uses and abuses",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Deep Learning", "Machine Learning"],
    license="Apache-2.0",
    packages=[
        package for package in find_packages() if package.startswith(PACKAGENAME)
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
