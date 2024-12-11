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
# Imports (wildcard)
from .data import *
from .distributed import *
from .logging import *
from .nn import *
from .optim import *
from .typing import *

# Deletions (from .data)
del FastCollateMixup
del Mixup
del cifarhundred_dataloader_dispatcher
del cifarten_dataloader_dispatcher
del data_prep_dispatcher_1ch
del data_prep_dispatcher_3ch
del fashionmnist_dataloader_dispatcher
del imagenette_dataloader_dispatcher
del kmnist_dataloader_dispatcher
del mnist_dataloader_dispatcher
del octmnist_dataloader_dispatcher
del pathmnist_dataloader_dispatcher
del tinyimagenet_dataloader_dispatcher
del tissuemnist_dataloader_dispatcher

# Deletions (from .distributed)
del reduce_accumulate_keepalive
del slurm_nccl_env

# Deletions (from .nn)
del ArgMaxLayer
del BasicAE
del BasicVAE
del BinarizeLayer
del BrokenReLU
del CausalConv1d
del Clamp
del Concatenate
del ConvNeXtStem
del ConvStem
del ConvolutionalFlattenLayer
del CoordConv1d
del CoordConv2d
del CoordConv3d
del DeepRBL
del DuplexLinearNeck
del FCBlock
del FieldTransform
del FlatChannelize2DLayer
del GRNorm
del GaussianReparameterizerSampler
del GaussianReparameterizerSamplerLegacy
del GenerAct
del InnerProduct
del KWTA1d
del KWTA2d
del MetaAILayerNorm
del Mish
del MultiSolvePoissonTensor
del NNEnsemble
del PoissonNetCifar
del PreActResNet
del ProbePrintLayer
del RBLinear
del ResBlock
del ReshapeLayer
del SERLU
del SGRUHCell
del ScaledERF
del SharedDuplexLinearNeck
del SilhouetteScore
del SimpleDuplexLinearNeck
del SinLU
del SingleNeckVAE
del SirenSine
del SmeLU
del SolvePoisson
del SolvePoissonTensor
del StatefulTupleSelect
del SwiGLU
del TupleDecouple
del TupleSelect
del ViTStem
del WideResNet
del beta_gaussian_kldiv
del beta_reco_bce
del beta_reco_bce_splitout
del beta_reco_mse
del beta_reco_mse_splitout
del bisided_thresholding
del build_repeated_sequential
del cummatmul
del field_transform
del lexsemble
del logit_to_prob
del mishlayer_init
del multilasso
del multiridge
del oldtranspose
del pixelwise_bce_mean
del pixelwise_bce_sum
del pixelwise_mse_mean
del pixelwise_mse_sum
del silhouette_score
del tensor_replicate
del var_of_lap

# Deletions (from .optim)
del ADOPT
del AdaBound
del AdamP
del Adan
del CosineLRScheduler
del Lamb
del Lion
del Lookahead
del Lookaround
del RAdam
del SAM
del SGDP
del alah_optim
del epochwise_onecycle
del make_beta_scheduler
del onecycle_lincos
del onecycle_linlin
del onecycle_linlin_updown
del ralah_optim
del tricyc1c
del warmed_up_annealer
del warmed_up_linneal
del wfneal

# Deletions (from .logging)
del AverageMeter
del LogCSV

# Deletions (from .typing)
del numlike
del realnum
del strdev
del tensorlike
del actvt
