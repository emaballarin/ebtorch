#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2019-* Brent Komer. All Rights Reserved.
#                      [orig. code: https://github.com/bjkomer/pytorch-legendre-memory-unit]
#
#
# Copyright (c) 2019-* Applied Brain Research. All Rights Reserved.
#                      [orig. work: https://papers.nips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf;
#                       orig. code: https://github.com/nengo/keras-lmu]
#
# Copyright (c) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
#                      [maintainance, adaptation, extension]
#
# ==============================================================================
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nengolib.signal import cont2discrete
from nengolib.signal import Identity
from nengolib.synapses import LegendreDelay


# from: https://github.com/deepsound-project/samplernn-pytorch/blob/master/nn.py#L46
def lecun_uniform(tensor):
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")  # skipcq: PYL-W0212
    nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))


# based on the tensorflow implementation:
# https://github.com/abr/neurips2019/blob/master/lmu/lmu.py
class LMUCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        order,
        theta=100,  # relative to dt=1
        method="zoh",
        trainable_input_encoders=True,
        trainable_hidden_encoders=True,
        trainable_memory_encoders=True,
        trainable_input_kernel=True,
        trainable_hidden_kernel=True,
        trainable_memory_kernel=True,
        trainable_a=False,
        trainable_b=False,
        input_encoders_initializer=lecun_uniform,
        hidden_encoders_initializer=lecun_uniform,
        memory_encoders_initializer=partial(torch.nn.init.constant_, val=0),
        input_kernel_initializer=torch.nn.init.xavier_normal_,
        hidden_kernel_initializer=torch.nn.init.xavier_normal_,
        memory_kernel_initializer=torch.nn.init.xavier_normal_,
        hidden_activation="tanh",
    ):
        super(LMUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.order = order

        if hidden_activation == "tanh":
            self.hidden_activation = torch.tanh
        elif hidden_activation == "relu":
            self.hidden_activation = torch.relu
        else:
            raise NotImplementedError(
                f"hidden activation '{hidden_activation}' is not implemented"
            )

        realizer = Identity()
        self._realizer_result = realizer(LegendreDelay(theta=theta, order=self.order))
        self._ss = cont2discrete(
            self._realizer_result.realization, dt=1.0, method=method
        )
        self._A = self._ss.A - np.eye(order)  # puts into form: x += Ax
        self._B = self._ss.B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI

        self.input_encoders = nn.Parameter(
            torch.Tensor(1, input_size), requires_grad=trainable_input_encoders
        )
        self.hidden_encoders = nn.Parameter(
            torch.Tensor(1, hidden_size), requires_grad=trainable_hidden_encoders
        )
        self.memory_encoders = nn.Parameter(
            torch.Tensor(1, order), requires_grad=trainable_memory_encoders
        )
        self.input_kernel = nn.Parameter(
            torch.Tensor(hidden_size, input_size), requires_grad=trainable_input_kernel
        )
        self.hidden_kernel = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size),
            requires_grad=trainable_hidden_kernel,
        )
        self.memory_kernel = nn.Parameter(
            torch.Tensor(hidden_size, order), requires_grad=trainable_memory_kernel
        )
        self.AT = nn.Parameter(torch.Tensor(self._A), requires_grad=trainable_a)
        self.BT = nn.Parameter(torch.Tensor(self._B), requires_grad=trainable_b)

        # Initialize parameters
        input_encoders_initializer(self.input_encoders)
        hidden_encoders_initializer(self.hidden_encoders)
        memory_encoders_initializer(self.memory_encoders)
        input_kernel_initializer(self.input_kernel)
        hidden_kernel_initializer(self.hidden_kernel)
        memory_kernel_initializer(self.memory_kernel)

    def forward(self, finput, hx):
        h, m = hx

        u = (
            F.linear(finput, self.input_encoders)
            + F.linear(h, self.hidden_encoders)
            + F.linear(m, self.memory_encoders)
        )

        m = m + F.linear(m, self.AT) + F.linear(u, self.BT)

        h = self.hidden_activation(
            F.linear(finput, self.input_kernel)
            + F.linear(h, self.hidden_kernel)
            + F.linear(m, self.memory_kernel)
        )

        return h, m
