import math
from collections.abc import Iterable, Sequence

import hyclib as lib
import numpy as np
import torch
from scipy import stats
from torch import Tensor

from niarb import random
from niarb.cell_type import CellType
from niarb.distributions import UniformEllipsoid
from niarb.nn.modules import frame
from niarb.tensors import categorical, periodic


def as_grid(
    n: int = 0,
    N_space: Sequence[int] = (),
    N_ori: int = 0,
    N_osi: int = 0,
    *,
    cell_types: Sequence[CellType | str] = tuple(CellType),
    space_extent: Sequence[float] = (1000.0, 1000.0, 300.0),
    ori_extent: Sequence[float] = (-90.0, 90.0),
    osi_prob: torch.distributions.Distribution | Sequence = ("Uniform", 0.0, 1.0),
) -> frame.ParameterFrame:
    """Generate a grid of neurons.

    Args:
        n (optional): Number of cell types.
        N_space (optional): Number of neurons to along each spatial dimension.
        N_ori (optional): Number of orientation samples.
        N_osi (optional): Number of OSI samples.
        cell_types (optional): Cell types to include.
        space_extent (optional):
            Lengths of the spatial dimensions. Must have at least as many elements as the number
            of spatial dimensions.
        ori_extent (optional): (lb, ub) tuple specifying the lower and upper bound of orientation.
        osi_prob (optional):
            Distribution of OSI. If a tuple, the first argument is the name of the distribution,
            and the rest are the distribution parameters. batch_shape must be either () or (n,).

    Returns:
        ParameterFrame of the grid of neurons, with shape ([n], [*N_space], [N_ori], [N_osi]).

    """
    if isinstance(osi_prob, Sequence):
        osi_prob = getattr(torch.distributions, osi_prob[0])(
            *[torch.as_tensor(v) for v in osi_prob[1:]]
        )

    if osi_prob.event_shape != () or osi_prob.batch_shape not in {(), (n,)}:
        raise ValueError(
            f"osi_prob must have event_shape () and batch_shape () or {(n,)=}, "
            f"but {osi_prob.event_shape=}, {osi_prob.batch_shape=}."
        )

    d = len(N_space)
    if d > len(space_extent):
        raise ValueError(
            f"space_extent must have at least {d} elements, but {len(space_extent)=}."
        )

    cell_types = tuple(CellType[ct] if isinstance(ct, str) else ct for ct in cell_types)

    dV, x, ndims = 1.0, {}, []
    if n > 0:
        x["cell_type"] = categorical.as_tensor(
            torch.arange(n), categories=[ct.name for ct in cell_types]
        )
        ndims.append(1)

    if d > 0:
        x["space"] = [
            periodic.linspace(-extent / 2, extent / 2, Ni)
            for extent, Ni in zip(space_extent, N_space)
        ]
        x["space"] = torch.cat(lib.pt.meshgrid(*x["space"], dims=-1), dim=-1)
        dV *= torch.prod(x["space"].period).item() / math.prod(N_space)
        space_dV = dV
        ndims.append(d)

    if N_ori > 0:
        x["ori"] = periodic.linspace(*ori_extent, N_ori)
        dV *= torch.prod(x["ori"].period).item() / N_ori
        ndims.append(1)

    dims = [(None, ndim) for ndim in ndims]
    x = dict(zip(x.keys(), lib.pt.meshgrid(*x.values(), dims=dims, sparse=True)))
    x = frame.ParameterFrame(x, ndim=sum(ndims))

    if N_osi > 0:
        m = max(1, n)
        osi = torch.linspace(0.0, 1.0, steps=N_osi)

        osi_prob = osi_prob.expand((m,))
        if isinstance(osi_prob, torch.distributions.Beta):
            # PyTorch currently does not support icdf for the beta distribution, so we use scipy.
            alpha, beta = osi_prob.concentration1, osi_prob.concentration0
            osi = [stats.beta.ppf(osi.numpy(), a, b) for a, b in zip(alpha, beta)]
            osi = torch.from_numpy(np.stack(osi)).float()  # (m, N_osi)
        else:
            osi = osi_prob.icdf(osi[:, None]).t()  # (m, N_osi)

        x = x.datailoc[..., None]
        x["osi"] = osi[(slice(None), *((None,) * (d + (N_ori > 0))), ...)].squeeze(0)
        dV *= 1 / N_osi

    x["dV"] = torch.tensor(dV)[(None,) * x.ndim]
    if d > 0:
        x["space_dV"] = torch.tensor(space_dV)[(None,) * x.ndim]

    return x


def sample(
    N: int,
    variables: Sequence[str],
    *,
    cell_types: Sequence[CellType | str] = tuple(CellType),
    cell_probs: torch.Tensor | Sequence[float] | None = None,
    cell_probs_strict: bool = False,
    space_extent: Sequence[float] = (1000.0, 1000.0, 300.0),
    space_geometry: str = "box",
    ori_extent: Sequence[float] = (-90.0, 90.0),
    osi_prob: torch.distributions.Distribution | Sequence = ("Uniform", 0.0, 1.0),
    min_dist: float = 0.0,
    min_dist_cell_types: dict[CellType | str, float] | None = None,
    w_dims: int | Iterable[int] | None = None,
) -> frame.ParameterFrame:
    """Generate samples of neurons.

    Args:
        N: Number of neurons to generate.
        variables: {"cell_type", "space", "ori", "osi"}. Variables to sample.
        cell_types (optional): Cell types to sample from.
        cell_probs (optional):
            Relative probabilities for each cell type. Defaults to the probabilities of each
            CellType Enum in cell_types, normalized such that the default E-I ratio is preserved.
        cell_probs_strict (optional): If True, instead of randomly assigning each neuron
            a cell type according to cell_probs by sampling from a categorical distribution,
            the total number of neurons of each cell type is fixed to N * cell_probs.
        space_extent (optional): Lengths of the spatial dimensions. If space_geometry is
            "ellipsoid", these are the half-lengths of the principal axes.
        space_geometry (optional): {"box", "ellipsoid"}. Spatial geometry.
        ori_extent (optional): (lb, ub) tuple of lower and upper bounds of the orientation extent.
        osi_prob (optional):
            Distribution of OSI. If a tuple, the first argument is the name of the distribution,
            and the rest are the distribution parameters. batch_shape must be either () or (n,).
        min_dist (optional): Minimum pairwise distance between neurons.
        min_dist_cell_types (optional): Minimum pairwise distance between neurons of
            a given cell type, for different cell types.
        w_dims (optional): spatial dimensions with periodic boundary conditions.
            If None, all spatial dimensions are periodic. Ignored if space_geometry is
            not "box".

    Returns:
        ParameterFrame of sampled neurons with shape (N,).

    """
    if len(variables) == 0:
        raise ValueError("At least one variable must be specified.")

    if space_geometry not in {"box", "ellipsoid"}:
        raise ValueError(
            f"space_geometry must be 'box' or 'ellipsoid', but got {space_geometry}."
        )

    cell_types = tuple(CellType[ct] if isinstance(ct, str) else ct for ct in cell_types)

    if min_dist_cell_types is None:
        min_dist_cell_types = {}
    elif len(min_dist_cell_types) >= len(cell_types):
        raise ValueError(
            f"min_dist_cell_types must have less than {len(cell_types)=} elements, "
            f"but got {len(min_dist_cell_types)=}."
        )
    else:
        min_dist_cell_types = {
            CellType[ct] if isinstance(ct, str) else ct: min_dist
            for ct, min_dist in min_dist_cell_types.items()
        }

    if cell_probs is None:
        total_E_prob = sum(ct.prob for ct in CellType if ct.sign == 1)
        total_I_prob = sum(ct.prob for ct in CellType if ct.sign == -1)
        subset_E_prob = sum(ct.prob for ct in cell_types if ct.sign == 1)
        subset_I_prob = sum(ct.prob for ct in cell_types if ct.sign == -1)
        E_ratio = total_E_prob / subset_E_prob if subset_E_prob > 0 else float("nan")
        I_ratio = total_I_prob / subset_I_prob if subset_I_prob > 0 else float("nan")
        cell_probs = [
            ct.prob * E_ratio if ct.sign == 1 else ct.prob * I_ratio
            for ct in cell_types
        ]

    cell_probs = torch.as_tensor(cell_probs)
    cell_probs = cell_probs / cell_probs.sum()

    if cell_probs_strict:
        # modify cell_probs such that (cell_probs * N) are integers but still sum to N
        cell_probs = hamiltons_method(cell_probs * N) / N

    if isinstance(osi_prob, Sequence):
        osi_prob = getattr(torch.distributions, osi_prob[0])(
            *[torch.as_tensor(v) for v in osi_prob[1:]]
        )

    n = len(cell_probs)
    if osi_prob.event_shape != () or osi_prob.batch_shape not in {(), (n,)}:
        raise ValueError(
            f"osi_prob must have event_shape () and batch_shape () or {(n,)=}, "
            f"but {osi_prob.event_shape=}, {osi_prob.batch_shape=}."
        )

    if osi_prob.batch_shape == (n,) and "cell_type" not in variables:
        raise ValueError(
            "If osi_prob has batch_shape (n,), 'cell_type' must be included in"
            " variables."
        )

    x = frame.ParameterFrame(ndim=1)
    dV = torch.tensor([1.0 / N])

    if "space" in variables:
        if space_geometry == "box":
            space_extent = [(-extent / 2, extent / 2) for extent in space_extent]
            m = torch.distributions.Uniform(*torch.tensor(list(zip(*space_extent))))
            ext = space_extent if w_dims is None else [space_extent[d] for d in w_dims]
            x["space"] = periodic.as_tensor(m.sample((N,)), extents=ext, w_dims=w_dims)
            dV = dV * math.prod((ub - lb for lb, ub in space_extent))
        else:
            m = UniformEllipsoid(torch.tensor(space_extent))
            x["space"] = m.sample((N,))
            dV = dV * m.volume().item()
        x["space"] = random.resample_with_min_dist(x["space"], m, min_dist)

    if "cell_type" in variables:
        if cell_probs_strict:
            n, N_ct = len(cell_types), (cell_probs * N).round().long()
            assert N_ct.sum().item() == N
            x["cell_type"] = torch.arange(n).repeat_interleave(N_ct)[torch.randperm(N)]
        else:
            x["cell_type"] = torch.distributions.Categorical(cell_probs).sample((N,))

        if "space" in variables:
            ct_indices = torch.tensor(
                [i for i, ct in enumerate(cell_types) if ct not in min_dist_cell_types]
            )
            for ct, min_dist_ct in min_dist_cell_types.items():
                cti = cell_types.index(ct)
                src_indices = torch.isin(x["cell_type"], ct_indices).nonzero().squeeze()
                target_indices = (x["cell_type"] == cti).nonzero().squeeze()
                loc = x["space"][target_indices]
                sampler = SwapLocSampler(x["space"], src_indices, target_indices)
                # this swaps the locations of neurons in-place to satisfy min_dist_ct
                random.resample_with_min_dist(loc, sampler, min_dist_ct)

        x["cell_type"] = categorical.as_tensor(
            x["cell_type"], categories=tuple(ct.name for ct in cell_types)
        )
        dV = dV / cell_probs  # (n,)
        dV = dV[x["cell_type"]]  # (N,)

    if "space" in variables:
        x["space_dV"] = dV

    if "ori" in variables:
        x["ori"] = periodic.as_tensor(
            torch.distributions.Uniform(*ori_extent).sample((N, 1)),
            extents=[ori_extent],
        )
        dV = dV * (ori_extent[1] - ori_extent[0])

    if "osi" in variables:
        osi = osi_prob.sample((N,))  # (N, n) or (N,) where n is number of cell types
        if osi.ndim == 1:
            x["osi"] = osi
        else:
            x["osi"] = osi.take_along_dim(x["cell_type"][:, None], dim=1).squeeze(1)

    x["dV"] = dV

    return x


def hamiltons_method(entitlement: Tensor) -> Tensor:
    """Hamilton's method for electoral (neurons) apportionment.

    Args:
        entitlement: Tensor of entitlements (expected number of neurons) for each party
            (cell type).

    Returns:
        Tensor of the number of seats (neurons) assigned to each party (cell type).

    """
    N = entitlement.sum().round().long()

    if not torch.isclose(entitlement.sum(), N.float()):
        raise ValueError(f"Sum of entitlements {entitlement.sum()} must be an integer.")

    if (entitlement < 0).any():
        raise ValueError(f"Entitlement must be non-negative, but got {entitlement=}.")

    if entitlement.ndim != 1:
        raise ValueError(
            f"Entitlement must be a 1D tensor, but got {entitlement.ndim=}."
        )

    apportionment = entitlement.floor().long()
    remainder = entitlement - apportionment
    _, idx = remainder.topk((N - apportionment.sum()).item())
    apportionment[idx] += 1

    return apportionment


class SwapLocSampler:
    def __init__(
        self,
        space_tensor: Tensor,
        src_indices: Tensor,
        target_indices: Tensor,
    ):
        self.space_tensor = space_tensor
        self.src_indices = src_indices[torch.randperm(src_indices.numel())]
        self.target_indices = target_indices

    def __call__(self, indices: Tensor) -> Tensor:
        N = indices.numel()
        if N > self.src_indices.numel():
            raise ValueError(
                f"Cannot sample {N=} locations: only {self.src_indices.numel()=} "
                "locations are available."
            )
        src_idx, self.src_indices = self.src_indices[:N], self.src_indices[N:]
        target_idx = self.target_indices[indices]
        self.space_tensor[torch.stack([target_idx, src_idx])] = self.space_tensor[
            torch.stack([src_idx, target_idx])
        ]
        return self.space_tensor[target_idx]
