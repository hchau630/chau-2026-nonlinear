import abc
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import tdfl
import torch
from torch import Tensor

from niarb import nn
from niarb.cell_type import CellType
from niarb.distributions import RectLinePicking
from niarb.nn.modules.frame import ParameterFrame
from niarb.optimize import elementwise
from niarb.tensors import periodic

__all__ = [
    "Constraint",
    "DeterminantCon",
    "EISigmaDiagCon",
    "L4CenterRespCon",
    "LinearResponseCon",
    "LinearResponseOriAnalyticCon",
    "LinearResponseOriCon",
    "LinearResponseSpace2dCon",
    "LinearResponseSpace2dOriCon",
    "LinearResponseSpaceCon",
    "ParadoxicalCon",
    "RelativeParamCon",
    "ResponseSpaceOriCon",
    "StabilityCon",
]


class Constraint(abc.ABC):
    def __init__(
        self, is_equality: bool, module_name: str = "", requires_graph: bool = False
    ):
        if not isinstance(is_equality, bool):
            raise TypeError(f"is_equality must be a bool, but {type(is_equality)=}.")

        super().__init__()
        self.is_equality = is_equality
        self.module_name = module_name
        self.requires_graph = requires_graph

    def __repr__(self):
        properties = [
            f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")
        ]
        return f"{type(self).__name__}({', '.join(properties)})"

    @abc.abstractmethod
    def __call__(self, module: torch.nn.Module, args: Any, out: Any) -> torch.Tensor:
        pass


class StabilityCon(Constraint):
    def __init__(
        self,
        eps: float = 0.1,
        cell_types: Iterable[CellType | str] | None = None,
        rel_vf: Tensor | Iterable[float] | float = 1.0,
        stable: bool = True,
    ):
        super().__init__(is_equality=False)
        self.eps = eps
        self.cell_types = cell_types
        self.stable = stable
        self.rel_vf = torch.as_tensor(rel_vf)

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        assert isinstance(m, nn.V1)
        if (self.rel_vf != 1.0).any():
            dh = m.vf * (self.rel_vf.to(m.vf.device) - 1.0)
        else:
            dh = 0.0
        a = m.spectral_summary(cell_types=self.cell_types, kind="J", dh=dh).abscissa
        out = -a if self.stable else a
        return out - self.eps


class DeterminantCon(Constraint):
    def __init__(
        self,
        eps: float = 1e-5,
        exclude: tuple[CellType, CellType] | tuple[str, str] | None = None,
        positive: bool = True,
    ):
        super().__init__(is_equality=False)
        self.eps = eps
        self.exclude = (CellType[exclude[0]], CellType[exclude[1]]) if exclude else None
        self.positive = positive

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        assert isinstance(m, nn.V1)

        W = m.W()
        if self.exclude:
            idx0 = m.cell_types.index(self.exclude[0])
            idx1 = m.cell_types.index(self.exclude[1])
            _W = torch.empty_like(W)[..., :-1, :-1]
            _W[..., :idx0, :idx1] = W[..., :idx0, :idx1]
            _W[..., :idx0, idx1:] = W[..., :idx0, idx1 + 1 :]
            _W[..., idx0:, :idx1] = W[..., idx0 + 1 :, :idx1]
            _W[..., idx0:, idx1:] = W[..., idx0 + 1 :, idx1 + 1 :]
            W = _W
        out = W.det()
        out = out if self.positive else -out
        return out - self.eps


class ParadoxicalCon(Constraint):
    def __init__(self, eps: float = 1e-5, cell_type: CellType | str = CellType.PV):
        super().__init__(is_equality=False)
        self.eps = eps
        self.cell_type = (
            CellType[cell_type] if isinstance(cell_type, str) else cell_type
        )

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        assert isinstance(m, nn.V1)

        W = m.W(with_gain=True)
        L = torch.linalg.inv(torch.eye(m.n, device=W.device, dtype=W.dtype) - W)
        idx = m.cell_types.index(self.cell_type)
        out = -L[..., idx, idx]
        return out - self.eps


class LinearResponseCon(Constraint):
    def __init__(
        self,
        positive: bool,
        post: CellType | str,
        pre: CellType | str,
        eps: float = 0.0,
    ):
        super().__init__(is_equality=False)
        self.positive = positive
        self.eps = eps
        self.post = CellType[post] if isinstance(post, str) else post
        self.pre = CellType[pre] if isinstance(pre, str) else pre

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        assert isinstance(m, nn.V1)

        W = m.W(with_gain=True)
        L = torch.linalg.inv(torch.eye(m.n, device=W.device, dtype=W.dtype) - W)
        post_idx = m.cell_types.index(self.post)
        pre_idx = m.cell_types.index(self.pre)
        out = L[..., post_idx, pre_idx]
        if not self.positive:
            out = -out
        return out - self.eps


class LinearResponseOriAnalyticCon(Constraint):
    def __init__(
        self,
        positive: bool,
        post: CellType | str,
        pre: CellType | str,
        eps: float = 0.0,
    ):
        super().__init__(is_equality=False)
        self.positive = positive
        self.eps = eps
        self.post = CellType[post] if isinstance(post, str) else post
        self.pre = CellType[pre] if isinstance(pre, str) else pre

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        assert isinstance(m, nn.V1)

        W = m.W(with_gain=True)  # (*, n, n)
        kappa = m.kappa  # (*, n, n)
        k = m.osi_scale()  # () or (n)
        if isinstance(k, Tensor) and k.ndim > 0:
            assert k.shape[-1] == W.shape[-1]
            k = k[..., None]  # (*, n, 1)
        eye = torch.eye(m.n, device=W.device, dtype=W.dtype)
        tL0 = torch.linalg.inv(eye - W) - eye
        tL1 = (torch.linalg.inv(eye - k * kappa * W) - eye) / k
        post_idx = m.cell_types.index(self.post)
        pre_idx = m.cell_types.index(self.pre)
        out = tL1[..., post_idx, pre_idx]
        # print(torch.linalg.cond(eye - k * kappa * W))
        if not self.positive:
            out = -out
        return out - self.eps * tL0[..., post_idx, pre_idx].abs()


class LinearResponseSpaceCon(Constraint):
    def __init__(
        self,
        d: int,
        r: np.ndarray | Tensor,
        eps: float = 1e-5,
        cell_types: Iterable[CellType | str] | None = None,
        perturbed_cell_type: CellType | str = CellType.PYR,
        positive: bool = True,
    ):
        super().__init__(is_equality=False)
        self.d = d
        self.r = torch.as_tensor(r, dtype=torch.float)
        if self.r.ndim != 1:
            raise ValueError(f"r must be 1-dimensional, but {r.ndim=}.")

        self.eps = eps
        if cell_types is not None:
            self.cell_types = [
                CellType[ct] if isinstance(ct, str) else ct for ct in cell_types
            ]
        else:
            self.cell_types = None
        if isinstance(perturbed_cell_type, str):
            self.perturbed_cell_type = CellType[perturbed_cell_type]
        else:
            self.perturbed_cell_type = perturbed_cell_type
        self.positive = positive

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        assert isinstance(m, nn.V1)

        model = nn.V1(
            ["cell_type", "space"],
            cell_types=m.cell_types,
            sigma_symmetry=m.sigma_symmetry,
        )
        # print(model.gW.grad)
        model.gW = m.gW
        model.sigma = m.sigma
        # print(model.gW.grad)
        # nn.load_state_dict(model, nn.state_dict(m), strict=False)

        space = torch.cat([torch.tensor([0]), self.r])  # (N + 1,)
        space = torch.stack(
            [space, *([torch.zeros_like(space)] * (self.d - 1))], dim=-1
        )  # (N + 1, d)
        dh = torch.zeros((model.n, space.shape[0]))  # (n, N + 1)
        dh[m.cell_types.index(self.perturbed_cell_type), 0] = 1.0

        # Note that dV doesn't matter since we are only interested in the response sign
        x = ParameterFrame(
            {
                "cell_type": torch.arange(model.n)[:, None],  # (n, 1)
                "space": space[None, ...],  # (1, N + 1, d)
                "dV": torch.tensor([[1.0]]),  # (1, 1)
                "dh": dh,  # (n, N + 1)
            },
            ndim=2,
        )  # (n, N + 1)

        x = x.to(model.gW.device)
        model.to(model.gW.device)

        out = model(
            x, ndim=x.ndim, check_circulant=False, to_dataframe=False
        )  # (n, N + 1)
        dr = out["dr"][:, 1:]  # (n, N)

        if not self.positive:
            dr = -dr

        if self.cell_types is None:
            out = dr.min()
        else:
            indices = [m.cell_types.index(ct) for ct in self.cell_types]
            out = dr[torch.tensor(indices, device=dr.device)].min()

        return out - self.eps


class LinearResponseSpace2dCon(Constraint):
    def __init__(
        self,
        a: float,
        b: float,
        min_r: float = 0.0,
        dr: float = 10.0,
        eps: float = 0.0,
        cell_types: Iterable[CellType | str] | None = None,
        perturbed_cell_type: CellType | str = CellType.PYR,
        positive: bool = True,
    ):
        super().__init__(is_equality=False)
        r = torch.arange(min_r, (a**2 + b**2) ** 0.5, dr)
        self.r = (r[1:] + r[:-1]) / 2
        self.pdf = RectLinePicking(a, b).log_prob(self.r).exp() * dr  # (N,)
        self.eps = eps
        if cell_types is not None:
            self.cell_types = [
                CellType[ct] if isinstance(ct, str) else ct for ct in cell_types
            ]
        else:
            self.cell_types = None
        if isinstance(perturbed_cell_type, str):
            self.perturbed_cell_type = CellType[perturbed_cell_type]
        else:
            self.perturbed_cell_type = perturbed_cell_type
        self.positive = positive

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        assert isinstance(m, nn.V1)

        if "space" not in m.variables:
            raise ValueError("model must have 'space' as one of its variables.")

        model = nn.V1(
            [v for v in m.variables if v in {"cell_type", "space"}],
            cell_types=m.cell_types,
            sigma_symmetry=m.sigma_symmetry,
        )
        model.gW = m.gW
        model.sigma = m.sigma

        space = torch.cat([torch.tensor([0]), self.r])  # (N + 1,)
        space = torch.stack([space, torch.zeros_like(space)], dim=-1)  # (N + 1, d)
        dh = torch.zeros((model.n, space.shape[0]))  # (n, N + 1)
        dh[m.cell_types.index(self.perturbed_cell_type), 0] = 1.0

        # Note that dV doesn't matter since we are only interested in the response sign
        x = ParameterFrame(
            {
                "cell_type": torch.arange(model.n)[:, None],  # (n, 1)
                "space": space[None, ...],  # (1, N + 1, d)
                "dV": torch.tensor([[1.0]]),  # (1, 1)
                "dh": dh,  # (n, N + 1)
            },
            ndim=2,
        )  # (n, N + 1)

        x = x.to(model.gW.device)
        model.to(model.gW.device)

        out = model(x, ndim=x.ndim, check_circulant=False, to_dataframe=False)
        dr = out["dr"][:, 1:]  # (n, N)
        dr = (dr * self.pdf.to(dr.device)).sum(dim=1)  # (n,)

        if self.cell_types is None:
            out = dr.min()
        else:
            indices = [m.cell_types.index(ct) for ct in self.cell_types]
            out = dr[torch.tensor(indices, device=dr.device)].min()

        if not self.positive:
            out = -out
        return out - self.eps


class LinearResponseOriCon(Constraint):
    def __init__(
        self,
        theta: float,
        eps: float = 1e-5,
        cell_types: Iterable[CellType | str] | None = None,
        positive: bool = True,
    ):
        super().__init__(is_equality=False)
        self.theta = theta
        self.eps = eps
        if cell_types is not None:
            self.cell_types = [
                CellType[ct] if isinstance(ct, str) else ct for ct in cell_types
            ]
        else:
            self.cell_types = None
        self.positive = positive

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        assert isinstance(m, nn.V1)

        variables = [v for v in m.variables if v != "space"]
        if "ori" not in variables:
            raise ValueError("model must have 'ori' as one of its variables.")

        model = nn.V1(
            variables,
            cell_types=m.cell_types,
            osi_func=m._osi_func,
            osi_prob=m.osi_prob,
        )
        model.gW = m.gW
        model.kappa = m.kappa

        expected = model.osi_scale(p=1)
        expected = (
            expected.cpu() if isinstance(expected, Tensor) else torch.tensor(expected)
        )
        osi = elementwise.bisect(
            lambda x: m.osi_func(x).cpu() - expected,
            torch.zeros_like(expected),
            torch.ones_like(expected),
        )  # osi_prob.batch_shape, which should be either () or (n,)
        osi = osi.broadcast_to((model.n,))  # (n,)

        ori = periodic.tensor(
            [[0.0], [self.theta / 2], [self.theta]], extents=[(-90.0, 90.0)]
        )  # (3, 1)
        dh = torch.zeros((model.n, 3))  # (n, 3)
        dh[m.cell_types.index(CellType.PYR), 0] = 1.0  # Excitatory neuron perturbation

        # Note that dV doesn't matter since we are only interested in the response sign
        x = ParameterFrame(
            {
                "cell_type": torch.arange(model.n)[:, None],  # (n, 1)
                "ori": ori[None, ...],  # (1, 3, 1)
                "osi": osi[:, None],  # (n, 1)
                "dV": torch.tensor([[1.0]]),  # (1, 1)
                "dh": dh,  # (n, 3)
            },
            ndim=2,
        )  # (n, 3)

        x = x.to(model.gW.device)
        model.to(model.gW.device)

        out = model(x, ndim=x.ndim, check_circulant=False, to_dataframe=False)  # (n, 3)
        dr = out["dr"][:, 1:]  # (n, 2)

        if self.cell_types is None:
            out = dr.min()
        else:
            indices = [m.cell_types.index(ct) for ct in self.cell_types]
            out = dr[torch.tensor(indices, device=dr.device)].min()

        if not self.positive:
            out = -out
        return out - self.eps


class LinearResponseSpace2dOriCon(Constraint):
    """
    In mode "min", constrain that the response averaged over space and OSI is greater
    than (or less than, if positive is False) 0 with margin eps.
    In mode "diff", constrain that the response averaged over space and OSI has kappa
    greater than (or less than, if positive is False) 0 with margin eps.
    """

    def __init__(
        self,
        a: float,
        b: float,
        mode: str = "min",
        theta: float = 90.0,
        min_r: float = 0.0,
        dr: float = 10.0,
        eps: float = 1e-5,
        cell_types: Sequence[CellType | str] | None = None,
        perturbed_cell_type: CellType | str = CellType.PYR,
        positive: bool = True,
    ):
        if mode not in {"min", "diff"}:
            raise ValueError()

        if mode == "diff" and (theta != 90.0 or not cell_types or len(cell_types) > 1):
            raise ValueError()

        super().__init__(is_equality=False)
        r = torch.arange(min_r, (a**2 + b**2) ** 0.5, dr)
        self.a = a
        self.b = b
        self.r = (r[1:] + r[:-1]) / 2
        self.pdf = RectLinePicking(a, b).log_prob(self.r).exp() * dr  # (N,)
        self.mode = mode
        self.theta = theta
        self.eps = eps
        if cell_types is not None:
            self.cell_types = [
                CellType[ct] if isinstance(ct, str) else ct for ct in cell_types
            ]
        else:
            self.cell_types = None
        if isinstance(perturbed_cell_type, str):
            self.perturbed_cell_type = CellType[perturbed_cell_type]
        else:
            self.perturbed_cell_type = perturbed_cell_type
        self.positive = positive

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        assert isinstance(m, nn.V1)

        if "ori" not in m.variables or "space" not in m.variables:
            raise ValueError(
                "model must have 'space' and 'ori' as one of its variables."
            )

        model = nn.V1(
            m.variables,
            cell_types=m.cell_types,
            osi_func=m._osi_func,
            osi_prob=m.osi_prob,
            sigma_symmetry=m.sigma_symmetry,
        )
        model.gW = m.gW
        model.sigma = m.sigma
        model.kappa = m.kappa

        expected = model.osi_scale(p=1)
        expected = (
            expected.cpu() if isinstance(expected, Tensor) else torch.tensor(expected)
        )
        osi = elementwise.bisect(
            lambda x: m.osi_func(x).cpu() - expected,
            torch.zeros_like(expected),
            torch.ones_like(expected),
        )  # osi_prob.batch_shape, which should be either () or (n,)
        osi = osi.broadcast_to((model.n,))  # (n,)

        ori = periodic.tensor(
            [[0.0], [self.theta / 2], [self.theta]], extents=[(-90.0, 90.0)]
        )  # (3, 1)
        space = torch.cat([torch.tensor([0]), self.r])  # (N + 1,)
        space = torch.stack([space, torch.zeros_like(space)], dim=-1)  # (N + 1, d)
        dh = torch.zeros((model.n, space.shape[0], 3))  # (n, N + 1, 3)
        dh[m.cell_types.index(self.perturbed_cell_type), 0, 0] = 1.0

        # Note that dV doesn't matter since we are only interested in the response sign
        x = ParameterFrame(
            {
                "cell_type": torch.arange(model.n)[:, None, None],  # (n, 1, 1)
                "space": space[None, :, None, :],  # (1, N + 1, 1, d)
                "ori": ori[None, None, ...],  # (1, 1, 3, 1)
                "osi": osi[:, None, None],  # (n, 1, 1)
                "dV": torch.tensor([[[180.0 * self.a * self.b]]]),  # (1, 1)
                "dh": dh,  # (n, N + 1, 3)
            },
            ndim=3,
        )  # (n, N + 1, 3)

        x = x.to(model.gW.device)
        model.to(model.gW.device)

        out = model(x, ndim=x.ndim, check_circulant=False, to_dataframe=False)
        dr = out["dr"][:, 1:, :]  # (n, N, 3)
        dr = (dr * self.pdf.to(dr.device)[None, :, None]).sum(dim=1)  # (n, 3)

        if self.mode == "min":
            if self.cell_types is None:
                out = dr.min()
            else:
                indices = [m.cell_types.index(ct) for ct in self.cell_types]
                out = dr[torch.tensor(indices, device=dr.device)].min()

            if not self.positive:
                out = -out
            return out - self.eps

        assert self.cell_types and len(self.cell_types) == 1
        dr = dr[m.cell_types.index(self.cell_types[0])]  # (3,)
        out = (dr[0] - dr[-1]) / 4
        if not self.positive:
            out = -out
        return out - self.eps * dr[1].abs()


class EISigmaDiagCon(Constraint):
    def __init__(self, eps: float = 0.0):
        super().__init__(is_equality=False)
        self.eps = eps

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        if len(m.cell_types) != 2:
            raise ValueError(
                f"model must have exactly 2 cell types, but got {m.cell_types=}."
            )

        S0 = torch.minimum(m.S[..., 0, 0], m.S[..., 1, 1])
        S1 = torch.maximum(m.S[..., 0, 1], m.S[..., 1, 0])
        return S0 - S1 - self.eps


class L4CenterRespCon(Constraint):
    """
    Constrain stim_1 L4 center resp to be less than or equal to
    frac * (stim_0 L4 center resp), i.e. frac * dr0 - dr1 >= eps.
    """

    def __init__(
        self, stim_0: str, stim_1: str, frac: float, radius: float, eps: float = 0.0
    ):
        super().__init__(is_equality=False)
        self.stim_0 = stim_0
        self.stim_1 = stim_1
        self.frac = frac
        self.radius = radius
        self.eps = eps

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        vis_modules = list(
            filter(lambda m: isinstance(m, nn.VisInput), model.modules())
        )
        if len(vis_modules) != 1:
            raise ValueError(
                f"model must have exactly one Vis module, but got {len(vis_modules)=}."
            )
        m = vis_modules[0]
        i = m.stim_types.categories.index(self.stim_0)
        j = m.stim_types.categories.index(self.stim_1)
        # Integral of a * exp(-r^2 / (2 * s^2)) over a disk with radius R is given by
        # 2 * pi * a * s^2 * (1 - exp(-R^2 / (2 * s^2))), thus average L4 activity
        # within this disk is 2 * a * (s / R)^2 * (1 - exp(-(R / s)^2 / 2)). Letting
        # r = R / s, we get 2 * a / r^2 * (1 - exp(-r^2 / 2)). We ignore the factor of 2
        # since we just care about the relative L4 activity.
        r = self.radius / m.s  # (2, n_stims)
        dr = m.a[0] / r[0] ** 2 * (1 - torch.exp(-(r[0] ** 2) / 2)) - m.a[1] / r[
            1
        ] ** 2 * (1 - torch.exp(-(r[1] ** 2) / 2))  # (n_stims,)
        return self.frac * dr[i] - dr[j] - self.eps


class ResponseSpaceOriCon(Constraint):
    def __init__(
        self,
        theta: float,
        dtheta: float = 10.0,
        min_r: float = 0.0,
        space_extent: Sequence[float] = (),
        min_osi: float = 0.0,
        max_osi: float = 1.0,
        N: int | None = None,
        cell_type: CellType | str | None = None,
        positive: bool = True,
        eps: float = 0.0,
    ):
        super().__init__(is_equality=False, module_name="model", requires_graph=True)
        self.min_theta = theta - dtheta / 2
        self.max_theta = theta + dtheta / 2
        self.min_r = min_r
        self.space_extent = space_extent
        self.min_osi = min_osi
        self.max_osi = max_osi
        self.N = N
        if cell_type:
            cell_type = cell_type if isinstance(cell_type, str) else cell_type.name
        self.cell_type = cell_type
        self.positive = positive
        self.eps = eps

    def __call__(self, m: torch.nn.Module, _: Any, out: Any) -> Tensor:
        assert isinstance(m, nn.V1)
        assert isinstance(out, tdfl.DataFrame)

        mask = (out["rel_ori"] > self.min_theta) & (out["rel_ori"] < self.max_theta)
        mask &= out["distance"] > self.min_r
        mask &= (out["osi"] >= self.min_osi) & (out["osi"] <= self.max_osi)
        for i, extent in enumerate(self.space_extent):
            mask &= out[f"space[{i}]"].abs() < extent / 2
        if self.N is not None:
            mask &= out["N"] == self.N
        if self.cell_type:
            mask &= out["cell_type"] == self.cell_type
        dr = out["dr"][mask].mean()
        return dr - self.eps


class L23CenterRespCon(Constraint):
    """
    Constrain stim_1 L2/3 center resp to be less than or equal to
    frac * (stim_0 L2/3 center resp), i.e. frac * dr0 - dr1 >= eps.
    """

    def __init__(
        self,
        cell_type: CellType | str,
        stim_0: str,
        stim_1: str,
        frac: float,
        radius: float,
        eps: float = 0.0,
    ):
        super().__init__(is_equality=False, module_name="model", requires_graph=True)
        self.cell_type = (
            cell_type.name if isinstance(cell_type, CellType) else cell_type
        )
        self.stim_0 = stim_0
        self.stim_1 = stim_1
        self.frac = frac
        self.radius = radius
        self.eps = eps

    def __call__(self, m: torch.nn.Module, _: Any, out: Any) -> Tensor:
        assert isinstance(m, nn.V1)
        assert isinstance(out, ParameterFrame)
        mask = (out["cell_type"] == self.cell_type) & (out["distance"] < self.radius)
        dr0 = out["dr"][(out["stim_type"] == self.stim_0) & mask].mean()
        dr1 = out["dr"][(out["stim_type"] == self.stim_1) & mask].mean()
        return self.frac * dr0 - dr1 - self.eps


class L4SigmaCon(Constraint):
    """
    Constrain L4 activity to be a typical difference of gaussians, where the negative
    component is broader than the positive component.
    """

    def __init__(self, stim: str, eps: float = 0.0):
        super().__init__(is_equality=False)
        self.stim = stim
        self.eps = eps

    def __call__(self, model: torch.nn.Module, *args) -> Tensor:
        vis_modules = list(
            filter(lambda m: isinstance(m, nn.VisInput), model.modules())
        )
        if len(vis_modules) != 1:
            raise ValueError(
                f"model must have exactly one Vis module, but got {len(vis_modules)=}."
            )
        m = vis_modules[0]
        i = m.stim_types.categories.index(self.stim)
        # s[1], the negative component, which should be broader than s[0] for every stim
        return m.s[1, i] - m.s[0, i] - self.eps


class RelativeParamCon(Constraint):
    """Impose relative constraint between two model parameters.

    Concretely, require frac * param[idx0] - param[idx1] == eps if is_equality else
    frac * param[idx0] - param[idx1] >= eps, where idx0, idx1 are the indices
    corresponding to cell_types_0, cell_types_1. If eps == 0, then this simplifies to
    the more intuitive constraints param[idx1] / param[idx0] == frac if is_equality else
    param[idx1] / param[idx0] <= frac if param[idx0] >= 0.
    """

    def __init__(
        self,
        param: str,
        cell_types_0: Sequence[CellType | str],
        cell_types_1: Sequence[CellType | str],
        frac: float,
        is_equality: bool = False,
        eps: float = 0.0,
    ):
        super().__init__(is_equality=is_equality, module_name="model")
        self.param = param
        self.cell_types_0 = [
            CellType[ct] if isinstance(ct, str) else ct for ct in cell_types_0
        ]
        self.cell_types_1 = [
            CellType[ct] if isinstance(ct, str) else ct for ct in cell_types_1
        ]
        self.frac = frac
        self.eps = eps

    def __call__(self, m: torch.nn.Module, *args) -> Tensor:
        assert isinstance(m, nn.V1)
        if m.batch_shape != ():
            raise NotImplementedError()

        idx0 = tuple(m.cell_types.index(ct) for ct in self.cell_types_0)
        idx1 = tuple(m.cell_types.index(ct) for ct in self.cell_types_1)
        param = getattr(m, self.param)
        return self.frac * param[idx0] - param[idx1] - self.eps
