#!/usr/bin/env python3
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import math
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager
from typing import List
from typing import Tuple
from typing import Union

import torch as th
from advertorch.attacks import Attack as ATAttack
from torchattacks.attack import Attack as TAAttack

__all__ = [
    "AdverApply",
    "TA2ATAdapter",
    "sample_ndball",
    "randpert",
]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
@contextmanager
def manual_seed(seed: int | None = None) -> Generator[None, None, None]:
    if seed is None:
        yield
        return
    old_state: th.Tensor = th.random.get_rng_state()
    try:
        if seed is not None:
            th.random.manual_seed(seed=seed)
        yield
    finally:
        if seed is not None:
            th.random.set_rng_state(new_state=old_state)


def sample_ndball(  # NOSONAR
    n: int = 1,
    d: int = 1,
    within: bool = True,
    qmc: bool = False,
    seed: int | None = None,
    device: th.device | None = None,
    dtype: th.dtype | None = None,
    sobol: th.quasirandom.SobolEngine | None = None,
) -> th.Tensor:
    """Sample `n` points uniformly from a `d`-dimensional unit hyperball or
        its surface.

    Args:
        n (int): Number of samples to harvest
        d (int): Dimension of hyperball
        within (bool): If True, sample inside the hyperball; if False, sample on its surface
        qmc (bool): Use quasi-Monte Carlo (Sobol) instead of pseudo-random sampling
        seed (int | None): Random seed
        device (torch.device | None): device for the sampled points
        dtype (torch.dtype | None): dtype for the sampled points
        sobol (torch.quasirandom.SobolEngine | None): SobolEngine instance for quasi-Monte Carlo sampling

    Returns:
        Tensor of shape (n, d) with points inside the unit hyperball (within=True)
        or on its surface (within=False)

    Raises:
        ValueError: If `d` is non-positive
    """
    if d <= 0:
        raise ValueError("`d` must be positive")
    if n <= 0:
        return th.empty(0, d, device=device, dtype=dtype)

    if not qmc:
        with manual_seed(seed=seed):
            if d == 1:
                if within:
                    return th.rand(n, 1, device=device, dtype=dtype) * 2 - 1
                else:
                    return th.randint(high=2, size=(n, 1), device=device, dtype=dtype) * 2 - 1
            pts: th.Tensor = th.randn(n, d, device=device, dtype=dtype)
            normed_pts: th.Tensor = pts / pts.norm(dim=1, keepdim=True)
            if within:
                return th.rand(n, 1, device=device, dtype=dtype).pow(exponent=1.0 / d) * normed_pts
            else:
                return normed_pts
    else:
        d_normal: int = 2 * math.ceil(d / 2)
        if sobol is not None:
            if sobol.dimension != d_normal + int(within):
                raise ValueError(f"`sobol.dimension` must be {d_normal + int(within)}")
            sobolengine: th.quasirandom.SobolEngine = sobol
        else:
            sobolengine = th.quasirandom.SobolEngine(dimension=d_normal + int(within), scramble=True, seed=seed)
        usamples: th.Tensor = sobolengine.draw(n=n).to(device=device, dtype=dtype)
        upairs: th.Tensor = usamples[:, :d_normal].view(n, d_normal // 2, 2)
        r: th.Tensor = th.sqrt(input=-2 * th.log(input=upairs[:, :, 0]))
        theta: th.Tensor = 2 * th.pi * upairs[:, :, 1]
        pts: th.Tensor = (
            th.stack(tensors=[r * th.cos(input=theta), r * th.sin(input=theta)], dim=2).view(n, d_normal)
        )[:, :d]
        normed_pts: th.Tensor = pts / pts.norm(dim=1, keepdim=True)
        if within:
            return usamples[:, -1:].pow(exponent=1.0 / d) * normed_pts
        else:
            return normed_pts


def randpert(
    x: th.Tensor,
    radius: float = 0,
    within: bool = True,
    qmc: bool = False,
    seed: int | None = None,
    sobol: th.quasirandom.SobolEngine | None = None,
    condition: bool = True,
) -> th.Tensor:
    """
    Randomly perturb the input tensor `x` by a uniformly-sampled vector within a `radius`-bounded hyperball.

    Args:
        x (Tensor): Input tensor to be perturbed.
        radius (float | None): Radius of the hyperball for perturbation. Defaults to 0.
        within (bool | None): If True, sample inside the hyperball; if False, sample on its surface. Defaults to True.
        qmc (bool | None): Whether to use quasi-Monte Carlo sampling. Defaults to True.
        seed (int | None): Random seed for reproducibility. Defaults to None.
        sobol (torch.quasirandom.SobolEngine | None): SobolEngine instance for quasi-Monte Carlo sampling. Defaults to None.
        condition (bool | None): If True, apply perturbation; if False, return original tensor. Defaults to True.
    """

    if radius <= 0 or not condition:
        return x
    exs: th.Size = x.shape[1:]
    return x + radius * sample_ndball(
        n=x.shape[0], d=math.prod(exs), within=within, qmc=qmc, seed=seed, device=x.device, dtype=x.dtype, sobol=sobol
    ).view(x.shape[0], *exs)


# ~~ Classes ~~ ────────────────────────────────────────────────────────────────
class TA2ATAdapter:
    """
    Adapt a TorchAttacks adversarial attack to the AdverTorch `perturb` API.
    """

    def __init__(self, attack: TAAttack) -> None:
        self.attack: TAAttack = attack

    def perturb(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        return self.attack(x, y)


class AdverApply:
    """
    Create fractionally-adversarially-perturbed batches for adversarial training and variations.
    """

    def __init__(
        self,
        adversaries: Union[
            List[Union[ATAttack, TA2ATAdapter]],
            Tuple[Union[ATAttack, TA2ATAdapter], ...],
        ],
        pre_process_fx: Callable = lambda x: x,
        post_process_fx: Callable = lambda x: x,
    ) -> None:
        self.adversaries = adversaries
        self.pre_process_fx = pre_process_fx
        self.post_process_fx = post_process_fx

    def __call__(
        self,
        x: List[th.Tensor],
        device,
        perturbed_fraction: float = 0.5,
        output_also_clean: bool = False,
    ) -> Tuple[th.Tensor, ...]:
        _batch_size: int = x[0].shape[0]

        _adv_number: int = len(self.adversaries)
        _atom_size: int = int((_batch_size * perturbed_fraction) // _adv_number)
        _perturbed_size: int = _atom_size * _adv_number

        _tensor_list_xclean: List[th.Tensor] = []
        _tensor_list_yclean: List[th.Tensor] = []
        _tensor_list_xpertu: List[th.Tensor] = []

        x = [self.pre_process_fx(x[0].to(device)), x[1].to(device)]

        # Clean fraction
        if _perturbed_size < _batch_size:
            _tensor_list_xclean.append(x[0][0 : -_perturbed_size + int(_perturbed_size == 0) * _batch_size])
            _tensor_list_yclean.append(x[1][0 : -_perturbed_size + int(_perturbed_size == 0) * _batch_size])
            _tensor_list_xpertu.append(x[0][0 : -_perturbed_size + int(_perturbed_size == 0) * _batch_size])

        # Perturbed fraction
        if _perturbed_size > 0:
            for _adv_idx, _adversary in enumerate(self.adversaries):
                _start_idx = _batch_size - _perturbed_size + _adv_idx * _atom_size
                _end_idx = _batch_size - _perturbed_size + (_adv_idx + 1) * _atom_size

                # Clean subfraction
                _tensor_list_xclean.append(x[0][_start_idx:_end_idx].detach())
                _tensor_list_yclean.append(x[1][_start_idx:_end_idx].detach())

                # Perturbed subfraction
                _xpertu: th.Tensor = (
                    _adversary.perturb(
                        x[0][_start_idx:_end_idx],
                        x[1][_start_idx:_end_idx],
                    )
                    .reshape(x[0][_start_idx:_end_idx].shape)
                    .detach()
                )
                _tensor_list_xpertu.append(_xpertu)

        if output_also_clean:
            return (
                self.post_process_fx(th.concat(_tensor_list_xpertu, 0)).detach(),
                th.concat(_tensor_list_yclean, 0).detach(),
                self.post_process_fx(th.concat(_tensor_list_xclean, 0)).detach(),
            )
        else:
            return (
                self.post_process_fx(th.concat(_tensor_list_xpertu, 0)).detach(),
                th.concat(_tensor_list_yclean, 0).detach(),
            )
