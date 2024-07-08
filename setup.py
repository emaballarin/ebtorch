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
import os

from setuptools import find_packages
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


PACKAGENAME: str = "ebtorch"

setup(
    name=PACKAGENAME,
    version="0.25.10",
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
    python_requires=">=3.11",
    install_requires=[
        "advertorch>=0.2.4",  # pip install git+https://github.com/BorealisAI/advertorch.git
        "matplotlib>=3.8",
        "medmnist>=3",
        "numpy>=1.24",
        "pillow>=10.3.0",
        "requests>=2.25",
        "torch-lr-finder>=0.2.1",
        "torch>=2",
        "torchattacks>=3.5.1",
        "torchvision>=0.15",
        "tqdm>=4.65",
    ],
    include_package_data=False,
    zip_safe=True,
)
