#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
# Copyright (c) 2021-* Ali Rahimi. All Rights Reserved.
#                      Licensed according to the MIT License.
#                      [orig. code: https://github.com/a-rahimi/laplace-net ]
#
# Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc> (minor edits)
#                      All Rights Reserved.
#                      Licensed according to the MIT License.
#
# ==============================================================================
from collections.abc import Callable
from collections.abc import Iterable
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "SolvePoisson",
    "SolvePoissonTensor",
    "MultiSolvePoissonTensor",
    "PoissonBasicBlock",
    "PoissonNetCifar",
]


def lattice_edges(
    image_height: int, image_width: int
) -> Tuple[Iterable[int], Iterable[int]]:
    """The set of edges in a 2D 4-connected graph.

    The edges in the network connect each vertex to its four adjacent vertices.
    A vertex (i,j) is represented as its flat index into the 2D image, as the
    integer i * image_width + j.

    An edge connects two vertices, so we represent the edges as two lists: the
    flat index of the start vertex, and the flat index of the end vertex.

    Every pair of connected vertices appears twice in this representation: once
    where the vertex is the center, and once again when it's a neighbor.
    """

    # (n x 2) array of coordinates (i,j)
    mat_is, mat_js = np.meshgrid(range(image_height), range(image_width))
    centers = np.vstack((mat_is.flatten(), mat_js.flatten())).T

    adjacency = np.array(((0, -1), (0, +1), (-1, 0), (+1, 0)))

    # (4n x 2) array of neighbors of (i,j)
    neighbors = (centers[:, None, :] + adjacency).reshape(-1, 2)

    # Index of valid neighbors
    i_valid = (
        (neighbors[:, 0] >= 0)
        * (neighbors[:, 0] < image_height)
        * (neighbors[:, 1] >= 0)
        * (neighbors[:, 1] < image_width)
    )

    # Represent the edges in the network. Each entry of edges_center is
    # (i,j), and its corresponding entry in edges_neighbor is an element of
    # N(i,j)
    edges_center = centers.repeat(len(adjacency), axis=0)[i_valid].dot((image_width, 1))
    edges_neighbor = neighbors[i_valid].dot((image_width, 1))

    return edges_center, edges_neighbor


def _make_copies(num_copies: int, cls, *args, **kwargs):
    return nn.Sequential(*[cls(*args, **kwargs) for _ in range(num_copies)])


class SolvePoisson(nn.Module):
    """Solve the Poisson equation on a 2D grid.

    Solve a problem of the form

       ∇²(R y) = x

    on a 2D grid for y. Here, R is a 2D array with dimension (height, width),
    and x, and y are batched 2D arrays with dimensions as batch_size x height x
    width.

    Since the solver is written in torch, you can compute the gradient of the
    solution with respect to both x and R.

    This is a generic solver for Poisson's equation, but the nomenclature is
    specific to solving a problem for a resistive sheet whose conductance
    varies over space.

    Example: Suppose a current I(i,j) is driven through each point (i,j) of a
    resistive sheet whose resistance at point (i,j) is R(i,j).  To enforce that
    resistance must always be positive, the instantaneous resistance at node
    (i,j) is supplied as the log-resistances r[i,j]. The instantaneous
    resistance at node (i,j) is R[i,j] = exp(r[i,j]).

    To compute the voltage at every point (i,j), we would solve

          ∇²(exp(log_resistances) voltages) = currents

    by running

          solver = SolvePoissoin(log_resistances)
          voltages = solver(currents)

    This is a closed form solution and does not require you to run additional
    gradient descent steps to refine the solution.

    Here is how the solution is computed: Let (u,v) = N(i,j) denote the
    neighbors of node (i,j).  The resistance between two adjacent nodes (i,j)
    and (u,v) can be computed from the instantaneous resistance at the nodes.
    It's the sum of their instantaneous resistance:

           R[(i,j), (u,v)] = exp(r[i,j]) + exp(r[u,v])

    To compute the voltage at every node (i,j), use the fact that the
    current flowing out of the node must equal the current flowing in:

           I[i,j] = sum_{(u,v) in N(i,j)} (V[i,j] - V[u,v]) / R[(i,j), (u,v)]

    In matrix form, this gives

           Z V = I,

    where row (i,j) of matrix Z has the form

          [... 1/R[(i,j), (u1,v1)] ... -Z_ii ...  1/R[(i,j), (ui,vi)] ...]

    where Z_ii is the sum of the all the other entries in the row.  The
    non-zero entries are at columns (u,v) in N(i,j), the neighbors of (i,j).
    """

    def __init__(self, log_resistances: torch.Tensor):
        super().__init__()
        self.log_resistances = nn.Parameter(log_resistances)
        self.edges_center, self.edges_neighbor = lattice_edges(*log_resistances.shape)

    def forward(self, input_currents: torch.Tensor):
        params_height, params_width = self.log_resistances.shape
        n_batches, image_height, image_width = input_currents.shape
        assert params_height == image_height
        assert params_width == image_width

        # This slab can't store or leak currents. Ensure the total current flux is 0.
        input_currents = input_currents - input_currents.mean()

        # Element (i,j) of this vector is R[i,j]
        center_conductances = torch.exp(self.log_resistances).flatten()

        mat_z_off_diagonal = torch.sparse_coo_tensor(
            (self.edges_center, self.edges_neighbor),  # type: ignore
            center_conductances[self.edges_center]
            + center_conductances[self.edges_neighbor],
            (image_width * image_height, image_width * image_height),
        ).to_dense()

        zmat = torch.diag(mat_z_off_diagonal.sum(axis=1)) - mat_z_off_diagonal

        # Z is symmetric and has a null-space, with Z 1 = 0, so it won't do to
        # just run solve(Z, I). Furthermore, pytorch doesn't have a built-in
        # way to find th minimum norm solution to an over-determined system of
        # equations. There's an easy work-around: We know 1'I = 0 also, because
        # the sum of currents flowing into the resistor network must equal the
        # sum of outgoing currents. This implies the following:
        #
        #   Claim: Z+11' has full rank. Furthermore, if x satisfies
        #        (Z + 11') x = I, it also satisfies Z x = I.
        #
        # An easy proof is to write the SVD of Z+11' in terms of the SVD Z=USV',
        # and to notice that Z (Z+11')^-1 y = y.
        #
        # All this to say that instead of solve(Z, y), we run solve(Z + 2, y)

        return (
            torch.linalg.solve(
                zmat + 1,
                input_currents.double().reshape(
                    n_batches, image_width * image_height, 1
                ),
            )
            .reshape(input_currents.shape)
            .float()
        )


class SolvePoissonTensor(nn.Module):
    """Solve one Poisson equation per channel and sum the results.

    Args:
        x: (num_batches, in_planes, height, width)
        R: (in_planes, height, width)

    Returns:
        y: (num_batches, height, width)

    For each plane c, solve for y_c in

        ∇²(R_c y_c) = x_c

    Then return y = bias + sum_c w_c y_c.
    """

    def __init__(self, in_planes: int, image_height: int, image_width: int):
        super().__init__()

        self.solvers = nn.ModuleList(
            [
                SolvePoisson(torch.rand(image_height, image_width, dtype=torch.float64))
                for _ in range(in_planes)
            ]
        )
        self.weights = nn.Parameter(torch.ones(in_planes))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_currents: torch.Tensor):
        in_planes = input_currents.shape[1]
        assert in_planes == len(self.solvers)

        # ys has shape (in_planes, num_batches, height, width)
        ys = torch.stack(
            [
                self.weights[i] * self.solvers[i](input_currents[:, i, :, :])
                for i in range(in_planes)
            ]
        )

        return self.bias + ys.sum(axis=0)


class MultiSolvePoissonTensor(nn.Module):
    """A bank of tensoriized poisson solvers.

    Multiple replicas of SolvePoissonTensor, stacked to create a multi-channel image.
    """

    def __init__(
        self, in_planes: int, image_height: int, image_width: int, out_planes: int
    ):
        super().__init__()

        self.tensor_solvers = nn.ModuleList(
            SolvePoissonTensor(in_planes, image_height, image_width)
            for _ in range(out_planes)
        )

    def forward(self, input_currents: torch.Tensor) -> torch.Tensor:
        # r is (out_planes, num_batches, height, width)
        r = torch.stack(
            [tensor_solver(input_currents) for tensor_solver in self.tensor_solvers]
        )
        # Convert to (num_batches, out_planes, height, width)
        return r.transpose(0, 1)


class PoissonBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        image_height: int,
        image_width: int,
        out_planes: int,
        activation_fx: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        super().__init__()
        self.solver1 = MultiSolvePoissonTensor(
            in_planes, image_height, image_width, out_planes
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.solver2 = MultiSolvePoissonTensor(
            out_planes, image_height, image_width, out_planes
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.activation_fx = deepcopy(activation_fx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation_fx(self.bn1(self.solver1(x)))
        out = self.bn2(self.solver2(out))

        # The final output may have more planes (aka channels) than the input x.
        # If so, pad x to match the output.
        in_planes = x.shape[1]
        out_planes = out.shape[1]
        if in_planes < out_planes:
            if (out_planes - in_planes) % 2:
                raise ValueError(
                    f"For now, planes may vary by even numbers only: in_planes = {in_planes}, out_planes = {out_planes}"
                )
            pad = (out_planes - in_planes) // 2
            shortcut = F.pad(
                x,
                (0, 0, 0, 0, pad, pad),
            )
        elif in_planes == out_planes:
            shortcut = x
        else:
            raise ValueError(
                f"For now, planes may only increase with layers: in_planes = {in_planes}, out_planes = {out_planes}"
            )

        return self.activation_fx(out + shortcut)


class PoissonNetCifar(nn.Module):
    """A resnet-like network where convolutions have been replaced by Poisson solver."""

    def __init__(
        self,
        image_height: int = 32,
        image_width: int = 32,
        num_classes: int = 10,
        activation_fx: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        _ = image_height, image_width  # Fake-gather
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.group1 = _make_copies(
            1,
            PoissonBasicBlock,
            in_planes=2,
            image_height=16,
            image_width=16,
            out_planes=2,
        )

        self.group1_to_group2 = PoissonBasicBlock(
            in_planes=2, image_height=8, image_width=8, out_planes=4
        )

        self.group2 = _make_copies(
            1,
            PoissonBasicBlock,
            in_planes=4,
            image_height=8,
            image_width=8,
            out_planes=4,
        )

        self.group2_to_group3 = PoissonBasicBlock(
            in_planes=4, image_height=4, image_width=4, out_planes=8
        )

        self.group3 = _make_copies(
            1,
            PoissonBasicBlock,
            in_planes=8,
            image_height=4,
            image_width=4,
            out_planes=8,
        )
        self.linear = nn.Linear(8, num_classes)
        self.activation_fx = deepcopy(activation_fx)

    def forward(self, x):
        # x is (batch_size x 3 x 32 x 32)

        out = self.activation_fx(self.bn1(self.conv1(x)))
        # out is (batch_size x 2 x 32 x 32)

        out = F.avg_pool2d(out, 2)
        # out is (batch_size x 2 x 16 x 16)

        out = self.group1(out)
        # out is (batch_size x 2 x 16 x 16)

        out = F.avg_pool2d(out, 2)
        # out is (batch_size x 2 x 8 x 8)

        out = self.group1_to_group2(out)
        # out is (batch_size x 4 x 8 x 8)

        out = self.group2(out)
        # out is (batch_size x 4 x 8 x 8)

        out = F.avg_pool2d(out, 2)
        # out is (batch_size x 4 x 4 x 4)

        out = self.group2_to_group3(out)
        # out is (batch_size x 8 x 4 x 4)

        out = self.group3(out)
        # out is (batch_size x 8 x 4 x 4)

        out = out.sum(axis=(2, 3))
        # out is (batch_size x 16)

        out = self.linear(out)
        # out is (batch_size x num_classes)

        return out
