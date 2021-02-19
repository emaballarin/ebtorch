# ==============================================================================
#
# Copyright (c) 2019-* Chang Xiao, Peilin Zhong and Changxi Zheng
#                      (Columbia University). All Rights Reserved.
#                      [orig. work: https://github.com/a554b554/kWTA-Activation]
#
# Copyright (c) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
#                      All Rights Reserved.
#                      [maintainance and adaptation]
#
# ==============================================================================


import torch
import torch.nn as nn


class Sparsify1d(nn.Module):
    # Sparsify (by ratio) keeping the k top-valued neurons
    def __init__(self, sparsity_ratio=0.5):
        super(Sparsify1d, self).__init__()
        self.sr = sparsity_ratio

    def forward(self, x):
        k = int(self.sr * x.shape[1])
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x


class SparsifyKAct1d(nn.Module):
    # Sparsify (by number) keeping the k top-valued neurons
    def __init__(self, k=1):
        super(SparsifyKAct1d, self).__init__()
        self.k = k

    def forward(self, x):
        k = self.k
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2d(nn.Module):
    # Sparsify (by ratio) keeping the k top-valued neurons, per-channel
    def __init__(self, sparsity_ratio=0.5):
        super(Sparsify2d, self).__init__()
        self.sr = sparsity_ratio

        self.preact = None
        self.act = None

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        tmpx = x.view(x.shape[0], x.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:, :, -1]
        topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(
            2, 3, 0, 1
        )
        comp = (x >= topval).to(x)
        return comp * x


class SparsifyVol2d(nn.Module):
    # Sparsify (by ratio) keeping the k top-valued neurons, cross-channel
    def __init__(self, sparsity_ratio=0.5):
        super(SparsifyVol2d, self).__init__()
        self.sr = sparsity_ratio

    def forward(self, x):
        size = x.shape[1] * x.shape[2] * x.shape[3]
        k = int(self.sr * size)

        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(x)
        comp = (x >= topval).to(x)
        return comp * x


class SparsifyKAct2d(nn.Module):
    # Sparsify (by number) keeping the k top-valued neurons, per-channel
    def __init__(self, k):
        super(SparsifyKAct2d, self).__init__()
        self.k = k

    def forward(self, x):
        k = self.k
        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).view_as(x)
        comp = (x >= topval).to(x)
        return comp * x


class SparsifyAbs2d(nn.Module):
    # Sparsify (by ratio) keeping the k top-absolute-valued neurons, per-channel
    def __init__(self, sparsity_ratio=0.5):
        super(SparsifyAbs2d, self).__init__()
        self.sr = sparsity_ratio

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:, :, -1]
        topval = topval.expand(
            absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]
        ).permute(2, 3, 0, 1)
        comp = (absx >= topval).to(x)
        return comp * x


class SparsifyInvAbs2d(nn.Module):
    # Sparsify (by ratio) keeping the k least-absolute-valued neurons, per-channel
    def __init__(self, sparsity_ratio=0.5):
        super(SparsifyInvAbs2d, self).__init__()
        self.sr = sparsity_ratio

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2, largest=False)[0][:, :, -1]
        topval = topval.expand(
            absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]
        ).permute(2, 3, 0, 1)
        comp = (absx >= topval).to(x)
        return comp * x


class BreakReLU(nn.Module):
    # Break ReLU activation function
    def __init__(self, sparsity_ratio=5):
        super(BreakReLU, self).__init__()
        self.h = sparsity_ratio
        self.thre = nn.Threshold(0, -self.h)

    def forward(self, x):
        return self.thre(x)
