import logging
from collections.abc import Sequence
from numbers import Number

import tdfl
import torch
from torch import Tensor

from niarb import nn
from niarb.cell_type import CellType
from niarb.tensors import categorical

from ..parameter import Parameter
from .frame import ParameterFrame, concat

logger = logging.getLogger(__name__)


class VisRelDeltaRate(torch.nn.Module):
    def __init__(
        self,
        a_optim: bool | Sequence[bool | Sequence] | Tensor = True,
        s_optim: bool | Sequence[bool | Sequence] | Tensor = True,
        k_optim: bool | Sequence[bool | Sequence] | Tensor = True,
        a_bounds: Sequence[float | Sequence] | Tensor = (0.0, 5.0),
        s_bounds: Sequence[float | Sequence] | Tensor = (0.0, 1e3),
        k_bounds: Sequence[float | Sequence] | Tensor = (-0.5, 0.5),
        a_init: float | Sequence[float | Sequence] | Tensor = 0.0,
        s_init: float | Sequence[float | Sequence] | Tensor = 100.0,
        k_init: float | Sequence[float | Sequence] | Tensor = 0.0,
        cell_types: Sequence[CellType | str] = tuple(CellType),
        var: str = "drf/rf",
        batch_shape: Sequence[int] = (),
        batch_dims: Sequence[str] = (),
        batch_coords: dict[str, Sequence[Number] | Tensor] | None = None,
    ):
        if len(batch_shape) == 0 and len(batch_dims) == 0 and batch_coords:
            batch_shape = tuple(len(v) for v in batch_coords.values())
            batch_dims = tuple(batch_coords.keys())

        if len(batch_dims) == 0:
            batch_dims = tuple(f"batch_dim_{i}" for i in range(len(batch_shape)))

        if len(batch_dims) != len(batch_shape):
            raise ValueError(f"{batch_dims=} incompatible with {batch_shape=}")

        n = len(cell_types)

        super().__init__()
        self.a = Parameter(
            torch.empty((*batch_shape, n)),
            bounds=a_bounds,
            requires_optim=a_optim,
            tag="vis_a",
        )
        self.s = Parameter(
            torch.empty((*batch_shape, n)),
            bounds=s_bounds,
            requires_optim=s_optim,
            tag="vis_s",
        )
        self.k = Parameter(
            torch.empty((*batch_shape, n)),
            bounds=k_bounds,
            requires_optim=k_optim,
            tag="vis_k",
        )
        self.register_buffer("a_init", torch.as_tensor(a_init), persistent=False)
        self.register_buffer("s_init", torch.as_tensor(s_init), persistent=False)
        self.register_buffer("k_init", torch.as_tensor(k_init), persistent=False)
        for si, batch_dim in zip(batch_shape, batch_dims, strict=True):
            batch_coord = torch.as_tensor(batch_coords.get(batch_dim, torch.arange(si)))
            self.register_buffer(batch_dim, batch_coord, persistent=False)
        self.batch_dims = batch_dims
        self.cell_types = tuple(
            ct if isinstance(ct, str) else ct.name for ct in cell_types
        )
        self.var = var

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.copy_(self.a, self.a_init)
        nn.init.copy_(self.s, self.s_init)
        nn.init.copy_(self.k, self.k_init)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        if x.data["cell_type"].categories != self.cell_types:
            raise ValueError("Provided cell types must match input cell types.")

        logger.debug(f"Pre-forward {x.shape=}")
        x = x.copy()

        r = x.data["space"].norm(dim=-1)  # (*shape,)
        cos = torch.cos(x.data["ori"].to_period(2 * torch.pi)).squeeze(-1)  # (*shape,)
        osi = x.data["osi"]  # (*shape,)

        # (*batch_shape, *shape)
        cell_types = x.data["cell_type"]
        cell_types = cell_types[(None,) * (x.ndim - cell_types.ndim) + (...,)]
        a = self.a[..., cell_types]
        s = self.s[..., cell_types]
        k = self.k[..., cell_types]

        for batch_dim in reversed(self.batch_dims):
            x = x.unsqueeze(0)
            x[batch_dim] = getattr(self, batch_dim)[(...,) + (None,) * (x.ndim - 1)]
            assert x[batch_dim].shape == x.shape

        x[self.var] = a * (1 + 2 * k * osi * cos) * torch.exp(-(r**2) / (2 * s**2))

        if (x.data[self.var] < -1).any():
            raise ValueError(
                f"Minimum relative change in rate {x.data[self.var].min().item()} is "
                "less than -1, implying negative firing rate."
            )

        assert x[self.var].shape == x.shape
        logger.debug(f"Post-forward {x.shape=}, {x.data[self.var].shape=}")
        logger.debug(f"\n{self.a.data=}\n{self.s.data=}\n{self.k.data=}")
        return x


class VisInput(torch.nn.Module):
    def __init__(
        self,
        stim_types: Sequence[str],
        cell_types: Sequence[CellType | str] = tuple(CellType),
        null_connections: Sequence[CellType | str] = (),
        a_optim: bool | Sequence[bool | Sequence] | Tensor = True,
        s_optim: bool | Sequence[bool | Sequence] | Tensor = True,
        k_optim: bool | Sequence[bool | Sequence] | Tensor = True,
        W_bounds: Sequence[float | Sequence] | Tensor = (0.0, 1e3),
        sigma_bounds: Sequence[float | Sequence] | Tensor = (0.0, 1e3),
        kappa_bounds: Sequence[float | Sequence] | Tensor = (-0.5, 0.5),
        a_bounds: Sequence[float | Sequence] | Tensor = (0.0, 5.0),
        s_bounds: Sequence[float | Sequence] | Tensor = (0.0, 1e3),
        k_bounds: Sequence[float | Sequence] | Tensor = (-0.5, 0.5),
        W_init_std: float = 1.0,
        a_init: Sequence[float | Sequence] | Tensor = ((1.0,), (0.0,)),
        batch_shape: Sequence[int] = (),
        batch_dims: Sequence[str] = (),
        batch_coords: dict[str, Sequence[Number] | Tensor] | None = None,
    ):
        if len(batch_shape) == 0 and len(batch_dims) == 0 and batch_coords:
            batch_shape = tuple(len(v) for v in batch_coords.values())
            batch_dims = tuple(batch_coords.keys())

        if len(batch_dims) == 0:
            batch_dims = tuple(f"batch_dim_{i}" for i in range(len(batch_shape)))

        if len(batch_dims) != len(batch_shape):
            raise ValueError(f"{batch_dims=} incompatible with {batch_shape=}")

        n_stims = len(stim_types)
        n = len(cell_types)
        cell_types = [CellType[ct] if isinstance(ct, str) else ct for ct in cell_types]
        null_connections = [
            CellType[ct] if isinstance(ct, str) else ct for ct in null_connections
        ]
        requires_optim = [ct not in null_connections for ct in cell_types]

        super().__init__()
        self.W = Parameter(
            torch.empty((*batch_shape, n)),
            requires_optim=requires_optim,
            bounds=W_bounds,
            tag="W_ff",
        )
        self.sigma = Parameter(
            torch.empty((*batch_shape, n)),
            requires_optim=requires_optim,
            bounds=sigma_bounds,
            tag="sigma_ff",
        )
        self.kappa = Parameter(
            torch.empty((*batch_shape, n)),
            requires_optim=requires_optim,
            bounds=kappa_bounds,
            tag="kappa_ff",
        )
        self.a = Parameter(
            torch.empty((*batch_shape, 2, n_stims)),
            bounds=a_bounds,
            requires_optim=a_optim,
            tag="a_L4",
        )
        self.s = Parameter(
            torch.empty((*batch_shape, 2, n_stims)),
            bounds=s_bounds,
            requires_optim=s_optim,
            tag="s_L4",
        )
        self.k = Parameter(
            torch.empty((*batch_shape, 2, n_stims)),
            bounds=k_bounds,
            requires_optim=k_optim,
            tag="k_L4",
        )

        self.register_buffer("a_init", torch.as_tensor(a_init), persistent=False)
        stim_types = categorical.tensor(list(range(n_stims)), categories=stim_types)
        self.register_buffer("stim_types", stim_types, persistent=False)
        # self.stim_types = stim_types
        for si, batch_dim in zip(batch_shape, batch_dims, strict=True):
            batch_coord = torch.as_tensor(batch_coords.get(batch_dim, torch.arange(si)))
            self.register_buffer(batch_dim, batch_coord, persistent=False)
        self.batch_dims = batch_dims
        self.W_init_std = W_init_std

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.W_(self.W, self.W_init_std, self.W.bounds, self.W.requires_optim)
        nn.init.uniform_(self.sigma, self.sigma.bounds, self.sigma.requires_optim)
        nn.init.uniform_(self.kappa, self.kappa.bounds, self.kappa.requires_optim)
        nn.init.copy_(self.a, self.a_init)
        nn.init.uniform_(self.s, self.s.bounds, self.s.requires_optim)
        nn.init.uniform_(self.k, self.k.bounds, self.k.requires_optim)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        logger.debug(f"Pre-forward {x.shape=}")
        x = x.copy()

        W = self.W[..., None, x["cell_type"]]  # (*batch_shape, 1, *shape)
        sigma = self.sigma[..., None, x["cell_type"]]  # (*batch_shape, 1, *shape)
        kappa = self.kappa[..., None, x["cell_type"]]  # (*batch_shape, 1, *shape)
        r = x["distance"]  # (*shape,)
        dori = x["ori"].to_period(2 * torch.pi)  # (*shape, 1)
        cos = torch.cos(dori).prod(dim=-1)  # (*shape,)

        # (2, *batch_shape, n_stims, *shape)
        ndim = len(self.batch_dims)
        a = self.a[(...,) + (None,) * x.ndim].movedim(ndim, 0)
        s = self.s[(...,) + (None,) * x.ndim].movedim(ndim, 0)
        k = self.k[(...,) + (None,) * x.ndim].movedim(ndim, 0)

        x = x.unsqueeze(0)  # (1, *shape)
        x["stim_type"] = self.stim_types[(...,) + (None,) * (x.ndim - 1)]
        assert x["stim_type"].shape == x.shape  # (n_stims, *shape)
        for batch_dim in reversed(self.batch_dims):
            x = x.unsqueeze(0)
            x[batch_dim] = getattr(self, batch_dim)[(...,) + (None,) * (x.ndim - 1)]
            assert x[batch_dim].shape == x.shape

        x["dr_L4"] = a[0] * (1 + 2 * k[0] * cos) * torch.exp(
            -(r**2) / (2 * s[0] ** 2)
        ) - a[1] * (1 + 2 * k[1] * cos) * torch.exp(
            -(r**2) / (2 * s[1] ** 2)
        )  # (*batch_shape, n_stims, *shape)
        x["dh"] = (
            x["dh"]
            * W
            * (
                (
                    a[0]
                    * (1 + 2 * k[0] * kappa * cos)
                    * torch.exp(-(r**2) / (2 * (s[0] ** 2 + sigma**2)))
                )
                - (
                    a[1]
                    * (1 + 2 * k[1] * kappa * cos)
                    * torch.exp(-(r**2) / (2 * (s[1] ** 2 + sigma**2)))
                )
            )
        )  # (*batch_shape, n_stims, *shape)
        assert x["dr_L4"].shape == x.shape
        assert x["dh"].shape == x.shape
        logger.debug(f"Post-forward {x.shape=}")
        logger.debug(f"\n{self.a.data=}\n{self.s.data=}\n{self.k.data=}")
        return x


class PreAnalysis(torch.nn.Module):
    def forward(self, x: ParameterFrame) -> tdfl.DataFrame:
        x0 = x.drop(columns=["dr_L4"])
        x1 = x.rename({"dr_L4": "dr"})

        cell_types = x["cell_type"].categories + ("L4",)
        n = len(cell_types)
        logger.debug(f"{cell_types=}")

        x0["cell_type"] = categorical.as_tensor(
            x0["cell_type"].tensor, categories=cell_types
        )
        x1["cell_type"] = categorical.as_tensor(
            torch.full_like(x1["cell_type"], n - 1), categories=cell_types
        )
        x = concat([x0, x1])
        return x.to_framelike(cls=tdfl.DataFrame, keep_indices=False, to_numpy=False)
