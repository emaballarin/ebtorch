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
# Imports
from .architectures import ArgMaxLayer
from .architectures import beta_reco_bce
from .architectures import BinarizeLayer
from .architectures import CausalConv1d
from .architectures import FCBlock
from .architectures import FCBlockLegacy
from .architectures import GaussianReparameterizerSampler
from .architectures import InnerProduct
from .architectures import pixelwise_bce_mean
from .architectures import pixelwise_bce_sum
from .architectures import SGRUHCell
from .conv2drp import BatchNorm2dRP
from .conv2drp import Conv2dRP
from .conv2drp import Dropout2dRP
from .conv2drp import patch_rp_train_network
from .convolutional_flatten import ConvolutionalFlattenLayer
from .coordconv import CoordConv1d
from .coordconv import CoordConv2d
from .coordconv import CoordConv3d
from .debuglayers import ProbePrintLayer
from .fieldtransform import FieldTransform
from .functional import field_transform
from .functional import mish
from .kwta import BrokenReLU
from .kwta import KWTA1d
from .kwta import KWTA2d
from .laplacenet import MultiSolvePoissonTensor
from .laplacenet import PoissonNetCifar
from .laplacenet import SolvePoisson
from .laplacenet import SolvePoissonTensor
from .mish import Mish
from .mish import mishlayer_init
from .nnsemble import NNEnsemble
from .reshapelayers import FlatChannelize2DLayer
from .reshapelayers import ReshapeLayer
from .serf import ScaledERF
from .serlu import SERLU
from .sinlu import SinLU
from .smelu import SmeLU
from .utils import *

# Deletions (from .)
del architectures
del conv2drp
del convolutional_flatten
del coordconv
del debuglayers
del fieldtransform
del kwta
del laplacenet
del mish
del nnsemble
del reshapelayers
del serlu
del sinlu
del smelu
del serf

# Deletions (from .functional)
# del mish  # (already done by chance!)
# del serf  # (already done by chance!)

# Deletions (from .utils)
del AdverApply
del AutoClipper
del TA2ATAdapter
del argser_f
del argsink
del download_gdrive
del emplace_kv
del gather_model_repr
del model_reqgrad
del model_reqgrad_
del patchify_2d
del patchify_batch
del patchify_dataset
del store_repr_autohook
del store_repr_fx
del store_repr_hook
