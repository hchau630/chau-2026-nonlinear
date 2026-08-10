import functools
import itertools
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from numbers import Number
from typing import NamedTuple

import hyclib as lib
import pandas as pd
import tdfl
import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from xitorch import optimize

from niarb import exceptions, ft, linalg, nn, numerics, random, special, utils, weights
from niarb import gaussian_moments as gm
from niarb.cell_type import CellType
from niarb.nn import functional
from niarb.nn.modules import frame
from niarb.nn.modules.frame import ParameterFrame
from niarb.nn.modules.kernels import Kernel, Radial
from niarb.optimize import elementwise
from niarb.tensors import categorical
from niarb.tensors.circulant import CirculantTensor
from niarb.tensors.periodic import PeriodicTensor
from niarb.utils import cast

logger = logging.getLogger(__name__)

APPROX_MODES = (
    "linear_approx",
    "quasi_linear_approx",
    "second_order_approx",
    "second_order_approx_2-1",
    "second_order_approx_2-2",
    "second_order_approx_naive",
    "second_order_approx_naive_2-1",
    "second_order_approx_naive_2-2",
)
MATRIX_APPROX_MODES = tuple("matrix_" + m for m in APPROX_MODES)
OTHER_MODES = (
    "analytical",
    "matrix",
    "newton",
    "broyden",
    "gd",
    "broyden1",
    "broyden2",
    "linearmixing",
    "numerical",
)
MODES = OTHER_MODES + APPROX_MODES + MATRIX_APPROX_MODES


class SpectralSummary(NamedTuple):
    abscissa: Tensor
    radius: Tensor


def compute_osi_scale(
    osi_prob: Distribution,
    osi_func: float | Callable[[Tensor], Tensor] = 1.0,
    p: int = 2,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""Compute $\mathbb{E}[f^p]$, where $f$ is osi_func and expectation is over osi_prob.

    Args:
        osi_prob: Distribution of OSI.
        osi_func (optional): Amplitude of of cosine-tuned weights as a function of OSI.
          f should map [0, 1] to [0, 1], with f(0) = 0 and f(1) = 1.
          If a float, f is given by CDF(x) ** osi_func, where CDF is the cumulative
          distribution of osi_prob. If osi_prob has batch_shape (n,), CDF is taken
          to be the CDF of the first element of osi_prob.
        p (optional): power of osi_func to calculate.
        device (optional): Device to create the tensor on. If None, uses CPU.
        dtype (optional): Tensor dtype. If None, uses torch.float.

    Returns:
        Tensor with shape osi_prob.batch_shape

    """
    if isinstance(osi_func, float):
        # we need to integrate P(x)F(x)^(pa) from 0 to 1
        # where P(x) is the probability density of x and F(x) is the CDF of x
        # so let u = F(x) so that du = P(x)dx. Also, since F(0) = 0 and F(1) = 1
        # by assumption, the integration limits remain the same.
        # Thus we have int_0^1 u^(pa) du = 1 / (pa + 1)
        if math.prod(osi_prob.batch_shape) == 1:
            return torch.tensor(1 / (p * osi_func + 1), device=device, dtype=dtype)

        def osi_func(x, alpha=osi_func):
            return osi_prob.cdf(x)[0] ** alpha

    if isinstance(osi_func, torch.nn.Identity):
        if p == 1:
            return torch.as_tensor(osi_prob.mean, device=device, dtype=dtype)

        if p == 2:
            return torch.as_tensor(
                osi_prob.mean**2 + osi_prob.variance, device=device, dtype=dtype
            )

        raise NotImplementedError("Only p = 1 or 2 implemented for osi_func(x) = x.")

    integrand = lambda x: osi_func(x) ** p * osi_prob.log_prob(x).exp()
    # do integral on CPU, since osi_func and osi_prob may have CPU parameters
    lb, ub = torch.zeros(osi_prob.batch_shape, dtype=dtype), 1
    out = lib.pt.integrate.fixed_quad(integrand, lb, ub, n=10)[0].to(device, dtype)
    return out


def UV_decomposition(W: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Decompose W*sigma**-2 into U*V.

    Decomposition is based on the shape of sigma, taking advantage of the implicit
    symmetries in sigma defined by its shape.

    Args:
        W: Tensor with shape (*, n, n)
        sigma: Tensor with shape broadcastable to (*, n, n).

    Returns:
        Tuple of tensors (U, V, S) with shapes (*, n, m), (*, m, n), (*, m, m),
        where S is a diagonal matrix with elements sigma**2.

    """
    n = W.shape[-1]
    A = W * sigma**-2  # (*, n, n)

    if sigma.shape[-2:] == (n, n):
        U = torch.zeros((*A.shape[:-2], n, n**2), dtype=A.dtype, device=A.device)
        V = torch.zeros((n**2, n), dtype=A.dtype, device=A.device)
        for i, j in itertools.product(range(n), repeat=2):
            U[..., i, i * n + j] = A[..., i, j]
            V[i * n + j, j] = 1
    else:
        eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        U, V = (A, eye) if sigma.shape[-2] == 1 else (eye, A)

    shape, m = sigma.shape[:-2], U.shape[-1]
    S = (sigma**2).reshape(*shape, -1).broadcast_to(*shape, m).diag_embed()  # (*, m, m)
    return U, V, S


def resolvent(
    l: Number,
    x: ParameterFrame,
    y: ParameterFrame,
    W: Tensor,
    sigma: Tensor,
    kappa: Tensor | float,
    osi_func: Callable[[Tensor], Tensor],
    osi_scale: Tensor | float,
    autapse: bool = False,
    order: int = 1,
    mode: str = "parallel",
    checkpoint: bool = False,
) -> Tensor:
    """Computes the linear response and connectivity function of the V1 model.

    Plugging in l = -1 yields the linear response (minus a delta function), while
    plugging in l = 0 yeidls the connectivity function. For other values of l,
    this returns l^{-1}(I - (I + lW)^{-1})

    Args:
        l: regular value of resolvent.
        x, y:
            ParameterFrames with any combination of columns ['space', 'cell_type', 'ori', 'osi'], with shapes Bx, By.
            x and y must share the same columns. If "osi" is present, "ori" must also be present.
        W: Tensor with shape (*BW, n, n)
        sigma: Shape must be broadcastable to (*BW, n, n).
        kappa: If a tensor, shape must be broadcastable to (k, *BW, n, n).
        osi_func: Amplitude of of cosine-tuned weights as a function of OSI.
        osi_scale:
            Expectation of osi_func ** 2 over some distribution of OSI. If a tensor, shape must be
            broadcastable to (n,).
        order (optional): Optional argument passed to functional.wrapped.
        mode (optional): {'parallel', 'sequential'}. Optional argument passed to functional.wrapped.
        checkpoint (optional): See niarb.special.resolvent.mixture

    Returns:
        Tensor with shape BWxy = broadcast(BW, Bx, By).

    """
    if set(x.keys()) != set(y.keys()):
        raise ValueError(
            f"Expected x and y to have the same keys, but got {x.keys()=} and"
            f" {y.keys()=}."
        )

    if "osi" in x and "ori" not in x:
        raise ValueError("If 'osi' is present in x, 'ori' must also be present.")

    batch_shape = W.shape[:-2]

    if isinstance(osi_scale, float):
        osi_scale = torch.tensor(osi_scale, device=W.device, dtype=W.dtype)
    osi_scale = osi_scale.broadcast_to((kappa.shape[0], W.shape[-1])).clone()  # (k, n)
    osi_scale[0] = 1.0

    if "cell_type" in x:
        i, j = x.data["cell_type"], y.data["cell_type"]  # Bx, By
    else:
        i = torch.zeros(x.shape, device=x.device, dtype=torch.long)  # Bx
        j = torch.zeros(y.shape, device=y.device, dtype=torch.long)  # By

    if "ori" in x:
        W = (W * kappa).movedim(0, -3)  # (*BW, k, n, n)
        sigma = sigma.unsqueeze(-3)  # (*BW, 1, n, n)
        i, j = i[..., None], j[..., None]  # (*Bx, 1), (*By, 1)

    if "osi" in x:
        l = l * osi_scale[..., None]  # (k, n, 1)

    if "space" in x:
        # U: (*BW, [k,], n, m), V: (*BW, [k,], m, n), S: (*BW, [k,] m, m)
        U, V, S = UV_decomposition(W, sigma)
        r = functional.diff(x.data["space"], y.data["space"])  # (*Bxy, d)
        if autapse:
            dr = special.ball_radius(r.shape[-1], y.data["space_dV"])  # (*By)
        else:
            dr = 0.0
        if "ori" in x:
            r = r[..., None, :]  # (*Bxy, 1, d)
            if isinstance(dr, Tensor):
                dr = dr[..., None]  # (*By, 1)

        if order == 1:
            # more optimized, memory-efficient path
            func = functools.partial(
                special.laplace_r, r.shape[-1], dr=dr, validate_args=False
            )
            r = r.norm(dim=-1)
        else:
            func = functools.partial(special.laplace, dr=dr, validate_args=False)
        func = functools.partial(
            special.mixture, S, U, V, func, l, i, j, checkpoint=checkpoint, real=True
        )
        if order > 1:
            func = functional.wrapped(func, order=order, mode=mode)
        out = func(r)  # (*BWxy, [k])
    else:
        eye = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)  # (n,)
        out = W @ torch.linalg.inv(l * W + eye)  # (*BW, [k,] n, n)
        out = utils.take_along_dims(
            out, i[..., None, None], j[..., None, None]
        )  # (*BWxy, [k])

    if "osi" in x:
        osi_factor = osi_func(x.data["osi"]) * osi_func(y.data["osi"])  # BWxy
        out, osi_factor = torch.broadcast_tensors(out, osi_factor[..., None])
        osi_factor = osi_factor.clone()  # (*BWxy, k)
        osi_factor[..., 0] = 1.0
        out = out * osi_factor  # (*BWxy, k)

    if "ori" in x:
        theta = functional.diff(x.data["ori"], y.data["ori"])  # (*BWxy, 1)
        out = ft.torus.irftn(
            out, theta, dim=(-1,), period=theta.period.tolist()
        )  # BWxy

    return out.broadcast_to(torch.broadcast_shapes(batch_shape, x.shape, y.shape))


def spectrum(
    W: Tensor,
    S: Tensor | None = None,
    tau: Tensor | None = None,
    kmax: float = 100.0,
    ksteps: int = 1000,
    cell_types: Iterable[int] | None = None,
) -> Tensor:
    """Compute a finite subset of the spectrum of the connectivity or jacobian operator.

    Args:
        W: Tensor with shape (*, n, n)
        S (optional): Tensor with shape (*, n, n). Connectivity width squared.
        tau (optional): Tensor with shape (*, n). Time constants of the model. If None,
          computes spectrum of the connectivity operator W. Otherwise, computes the
          spectrum of the jacobian T^{-1}(W - I)
        kmax (optional): Maximum Fourier frequency.
        ksteps (optional): Number of steps in Fourier frequency.
        cell_types (optional): Spectrum of the subcircuit composed of only the
          specified cell-types. If None, computes spectrum of the full circuit.

    Returns:
        Tensor with shape (*, ksteps, m), where m = len(cell_types) if cell_types else n.

    """
    if S is not None:
        # normalize by mean of S since the spectrum should be invariant to the scaling of S
        S = S / S.mean(dim=(-2, -1), keepdim=True)  # (*, n, n)

        k = torch.linspace(0, kmax, steps=ksteps, device=S.device, dtype=S.dtype)
        k = k[:, None, None]  # (ksteps, 1, 1)
        W = W[..., None, :, :]  # (*, 1, n, n)
        S = S[..., None, :, :]  # (*, 1, n, n)

        op = W * (1 + S * k**2).reciprocal()  # (*, ksteps, n, n)
    else:
        op = W[..., None, :, :]  # (*, 1, n, n)

    if tau is not None:
        eye = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)
        op = tau[..., None, :, None].reciprocal() * (op - eye)  # (*, ksteps/1, n, n)

    if cell_types:
        cell_types = list(cell_types)
        op = op[..., cell_types, :][..., :, cell_types]

    return cast(torch.linalg.eigvals, op)  # (*, ksteps, m)


def spectral_summary(
    W: Tensor,
    S: Tensor | None = None,
    kappa: Tensor | float | None = None,
    osi_scale: Tensor | float | None = None,
    tau: Tensor | None = None,
    return_eigvals: bool = False,
    **kwargs,
) -> SpectralSummary | tuple[SpectralSummary, Tensor]:
    """Compute spectral abscissa and radius of the connectivity or jacobian operator.

    Args:
        W: Tensor with shape (*, n, n).
        S (optional): If a tensor, shape must be broadcastable to (*, n, n).
        kappa (optional): If a tensor, shape must be broadcastable to (k, *, n, n).
        osi_scale (optional): Expectation of osi_func ** 2 over some distribution of
          OSI. If a tensor, shape must be broadcastable to (n,).
        tau (optional): Tensor with shape (*, n). Time constants of the model. If None,
          computes spectrum of the connectivity operator W. Otherwise, computes the
          spectrum of the jacobian T^{-1}(W - I)
        return_eigvals (optional): If True, also return the eigenvalues.
        **kwargs: Optional arguments passed to spectrum.

    Returns:
        Namedtuple with fields 'abscissa' and 'radius', both of which are Tensors with shape (*)

    """
    if osi_scale is not None:
        kappa[1:] = kappa[1:] * osi_scale

    if kappa is not None:
        W = W * kappa  # (k, *, n, n)

    eigvals = spectrum(W, S=S, tau=tau, **kwargs)  # ([k,] *, ksteps, m)
    eigvals = eigvals.reshape(*eigvals.shape[:-2], -1)  # ([k,] *, ksteps*m)

    abscissa = eigvals.real.max(dim=-1).values  # ([k,] *)
    radius = eigvals.abs().max(dim=-1).values  # ([k,] *)

    if S is not None and tau is None:
        # we have a lower bound for abscissa by taking frequency to infinity
        abscissa = torch.clip(abscissa, 0, torch.inf)  # ([k,] *)

    if kappa is not None:
        abscissa = abscissa.max(dim=0).values  # (*)
        radius = radius.max(dim=0).values  # (*)

    out = SpectralSummary(abscissa=abscissa, radius=radius)
    if return_eigvals:
        if kappa is not None:
            eigvals = eigvals.movedim(0, -1)  # (*, ksteps*m, k)
            eigvals = eigvals.reshape(*eigvals.shape[:-2], -1)  # (*, ksteps*m*k)
        return out, eigvals

    return out


def spectral_norm(
    W: Tensor,
    sigma: Tensor | None = None,
    kappa: Tensor | float = 0.0,
    osi_scale: Tensor | float = 1.0,
    a: Tensor | float = 1.0,
    b: Tensor | float = 1.0,
    kmax: float = 100.0,
    ksteps: int = 1000,
) -> Tensor:
    r"""Compute spectral norm of the operator diag(a)\tilde{L}diag(b).

    Args:
        W: Tensor with shape (*, n, n).
        sigma (optional): Tensor with shape broadcastable to (*, n, n).
        kappa (optional): If a tensor, shape must be broadcastable to (*, n, n).
        osi_scale (optional): Expectation of osi_func ** 2 over some distribution of
          OSI. If a tensor, shape must be broadcastable to (n,).
        a (optional): Tensor with shape broadcastable to (n,).
        b (optional): Tensor with shape broadcastable to (n,).

    Returns:
        Tensor with shape (*) representing the spectral norm.

    """
    n = W.shape[-1]

    if sigma is None:
        sigma = torch.ones(1, 1, device=W.device, dtype=W.dtype)
    else:
        # spectrum should be invariant to the scaling of S
        sigma = sigma / sigma.mean(dim=(-2, -1), keepdim=True)  # (*, n, n)

    W = torch.stack([W, W * kappa])  # (*, 2, n, n)
    K = torch.ones((n,), device=W.device, dtype=W.dtype)  # (n,)
    K = torch.stack([K, K * osi_scale]).sqrt()  # (2, n)
    KA, BK = (K * a).diag_embed(), (b * K).diag_embed()  # (*, 2, n, n)
    U, V, S = UV_decomposition(W, sigma)  # (*, 2, n, m), (*, 2, m, n), (*, m, m)
    Sinv = torch.linalg.inv(S)[..., None, :, :]  # (*, 1, m, m)
    L, Q = torch.linalg.eig(Sinv - V @ K.diag_embed() @ U)  # (*, 2, m), (*, 2, m, m)
    k = torch.linspace(0, kmax, steps=ksteps, device=L.device, dtype=L.dtype)[..., None]
    U, V, L, Q = U.unsqueeze(-3), V.unsqueeze(-3), L.unsqueeze(-2), Q.unsqueeze(-3)
    KA, BK = KA.unsqueeze(-3), BK.unsqueeze(-3)
    KA, BK, U, V = KA.to(L.dtype), BK.to(L.dtype), U.to(L.dtype), V.to(L.dtype)

    # shape of C: (*, 2, ksteps, n, n)
    C = KA @ U @ Q @ (L + k**2).reciprocal().diag_embed() @ torch.linalg.inv(Q) @ V @ BK
    # since output of eigvalsh is sorted, largest eigenvalue is the last element
    out = torch.linalg.eigvalsh(C.mH @ C)[..., -1]  # (*, 2, ksteps)
    return out.amax(dim=(-2, -1)).sqrt()  # (*)


class V1(torch.nn.Module):
    r"""Firing rate model of perturbation response of a single layer of mouse V1.

    Model connectivity function is given by
    \[
        W_{\alpha\beta}(x, y, \theta, \phi, \mu, \nu)
        = \frac{w_{\alpha\beta}}{2\pi\sigma_{\alpha\beta}^2}
          G_d(r;\sigma_{\alpha\beta}^{-2})
          (1 + 2\kappa_{\alpha\beta}f(\mu)f(\nu)\cos(\theta - \phi))
    \]
    where $d$ is the number of spatial dimensions, and $G_d(r;\sigma^{-2})$ is defined by
    \[
        G_d(r;\sigma^{-2}) = (2\pi)^{-\frac{d}{2}}(\sigma r)^-\nu K_\nu(\frac{r}{\sigma})
    \]
    where $\nu = \frac{d}{2} - 1$, and $K_\nu$ is the modified Bessel function of the
    second kind of order $\nu$. Distribution of OSI $\mu \in [0, 1]$ is allowed to be
    non-uniform and dependent on cell type with probability distribution $P_\alpha$.
    Number of spatial dimensions is determined by the shape of the model input x, with
    d = x["space"].shape[-1].

    """

    def __init__(
        self,
        variables: Sequence[str],
        *,
        cell_types: Sequence[CellType | str] = tuple(CellType),
        tau: Sequence[float] | float = 1.0,
        ori_func: str | Callable[[int, Tensor], Tensor] = "cosine",
        ori_order: int | None = None,
        osi_func: (
            float | Distribution | Callable[[Tensor], Tensor] | str | Sequence
        ) = "Identity",
        osi_prob: Distribution | Sequence = ("Uniform", 0.0, 1.0),
        f: float | Callable[[Tensor], Tensor] | str | Sequence = "Identity",
        sigma_symmetry: str | Sequence[Sequence[int]] | Tensor | None = None,
        vf_symmetry: bool = True,
        null_connections: Iterable[Sequence[CellType | str]] | None = None,
        autapse: bool = False,
        gW_optim: bool = True,
        sigma_optim: bool | Sequence[bool | Sequence] | Tensor | None = None,
        kappa_optim: bool | Sequence[bool | Sequence] | Tensor | None = None,
        vf_optim: bool | Sequence[bool | Sequence] | Tensor | None = None,
        gW_bounds: Sequence[float | Sequence] | Tensor = (1e-5, 1e3),
        sigma_bounds: Sequence[float | Sequence] | Tensor = (1e0, 1e3),
        kappa_bounds: Sequence[float | Sequence] | Tensor = (-0.5, 0.5),
        vf_bounds: Sequence[float] | Tensor = (1.0e-5, 1e3),
        init_gW_std: float = 0.5,
        init_gW_bounds: Sequence[float | Sequence] | Tensor | None = None,
        init_sigma_bounds: Sequence[float | Sequence] | Tensor | None = None,
        init_kappa_bounds: Sequence[float | Sequence] | Tensor | None = None,
        init_vf: float | Sequence[float] | Tensor = 1.0,
        rf_cv: float | Sequence[float] | Tensor = 0.0,
        init_stable: bool = False,
        mode: str = "analytical",
        space_strength_kernel: str | type[Kernel] | None = None,
        prob_kernel: dict[str, Kernel] | None = None,
        monotonic_strength: bool = False,
        keep_monotonic_norm: bool = False,
        monotonic_norm_ord: int | float = 1,
        dense: bool = False,
        N_synapses: float | int | None = None,
        W_std: float = 0.0,
        seed: int | None = None,
        sparsify_kwargs: dict[str] | None = None,
        nonlinear_kwargs: dict[str] | None = None,
        simulation_kwargs: dict[str] | None = None,
        monotonic_kwargs: dict[str] | None = None,
        wrapped_kwargs: dict[str] | None = None,
        batch_shape: Sequence[int] = (),
    ):
        """Initialize V1 model.

        Args:
            variables:
                {"cell_type", "space", "ori", "osi"}. Dependent variables of the connectivity function.
            cell_types (optional): Cell types in the model.
            tau (optional): Time constants of the model.
            ori_func (optional): Orientation tuning kernel. If a string, must be one of
                {"cosine", "von_mises"}. If a callable, must return Fourier coefficients.
            ori_order (optional): Number of Fourier modes for orientation tuning.
                If None, include all Fouriers modes, but raises error if ori_func is
                callable. Set to 2 if None and ori_func is "cosine".
            osi_func (optional):
                A function f(x) that determines the scaling of cosine-tuned weights
                as a function of OSI. If a float, f is given by CDF(x) ** osi_func,
                where CDF is the cumulative distribution of osi_prob. If a Distribution,
                f is given by osi_func.cdf. If osi_prob has batch_shape (n,), CDF is
                taken to be that of the first element. Defaults to f(x) = x.
            osi_prob (optional):
                Distribution of OSI. If a tuple, the first argument is the name of the distribution,
                and the rest are the distribution parameters. batch_shape must be either () or (n,).
                Ignored if "osi" not in variables.
            f (optional): Model nonlinearity. If None, model is linear.
            sigma_symmetry (optional): Symmetries in sigma. If a string, must be one of
                {"pre", "post", "full"}. If tensor-like, must have shape (n, n)
                consisting of consecutive integers starting from 0. If None, no symmetry
                is assumed.
            vf_symmetry (optional): If True, vf is the same for different cell types.
            null_connections (optional):
                Sequence of (cell_type_i, cell_type_j) pairs where the connectivity from
                cell_type_j to cell_type_i is fixed to zero. If None, defaults to the connectivity
                specified by the 'targets' field of each cell type in cell_types.
            autapse (optional): Whether or not to allow autapses. Ignored if space_strength_kernel is
                not None.
            gW_optim (optional): If False, don't optimize gW at all.
            sigma_optim (optional): Whether or not to optimize sigma. Defaults to True if "space" in variables.
            kappa_optim (optional): Whether or not to optimize kappa. Defaults to True if "ori" in variables.
            vf_optim (optional): Whether or not to optimize vf. Defaults to True if f is not Identity.
            gW_bounds (optional): Lower and upper bound for all elements or elementwise
                lower and upper bounds of gW.
            sigma_bounds (optional):
                Lower and upper bound for all elements or elementwise lower and upper bounds of sigma.
            kappa_bounds (optional):
                Lower and upper bound for all elements or elementwise lower and upper bounds of kappa.
            vf_bounds (optional):
                Lower and upper bound for all elements or elementwise lower and upper bounds of vf.
                Defaults to (1.0e-5, inf), since for most typical choices of nonlinearity a negative vf
                would typically result in 0 gain due to rectification.
            init_gW_std (optional): Standard deviation of initial gW.
            init_gW_bounds (optional): Initialization lower and upper bound for all
                elements or elementwise initialization lower and upper bounds of gW.
            init_sigma_bounds (optional):
                Initialization lower and upper bound for all elements or elementwise initialization
                lower and upper bounds of sigma.
            init_kappa_bounds (optional):
                Initialization lower and upper bound for all elements or elementwise initialization
                lower and upper bounds of kappa.
            init_vf (optional): Initial baseline voltage ('v'oltage 'f'ixed-point).
            rf_cv (optional): Coefficient of variation of baseline firing rate across
                neurons. If nonzero, the value of vf used in the forward pass is sampled
                per-neuron from a Gaussian distribution chosen (via gaussian_pullback) so
                that the variance of rf is (rf_cv * f(vf)) ** 2. Only supported when
                mode == 'numerical' or is one of the matrix modes; other modes raise
                NotImplementedError if rf_cv != 0.
            init_stable (optional): If True, resample initial parameters until the model is stable.
            mode (optional): {'analytical', 'matrix', 'newton', 'broyden', 'gd',
                'broyden1', 'broyden2', 'linearmixing', 'numerical', 'linear_approx',
                'quasi_linear_approx', 'second_order_approx', 'second_order_approx_2-1',
                'second_order_approx_2-2', 'second_order_approx_naive',
                'second_order_approx_naive_2-1', 'second_order_approx_naive_2-2',
                'matrix_linear_approx', 'matrix_quasi_linear_approx',
                'matrix_second_order_approx', 'matrix_second_order_approx_2-1',
                'matrix_second_order_approx_2-2', 'matrix_second_order_approx_naive',
                'matrix_second_order_approx_naive_2-1', 'matrix_second_order_approx_naive_2-2'}.
                Method for computing the perturbation response.
            wrapped_kwargs (optional): keyword arguments passed to functional.wrapped.
            batch_shape (optional): Shape of model batch dimensions.
            space_strength_kernel (optional): Kernel class for computing spatial connectivity strength.
                Assumed to be translationally invariant. If None, this is the Laplace kernel divided by
                prob_kernel["space"]. If not None, `mode` must be "numerical" or "matrix".
            The remaining options are ignored if `mode` == 'analytical':
            prob_kernel (optional): Dict of kernel functions for computing connection probabilities.
                space and ori kernels are assumed to be translationally invariant. Keys must be ones of
                a subset of {"cell_type"} | set(variables). If "cell_type" is not in variables, use
                "cell_type" to specify the overall probability amplitude.
            monotonic_strength (optional): If True, connectivity strength is modified to be
                monotonically decreasing with distance, and prob_kernel["space"] must be a Radial kernel
                if provided.
            keep_monotonic_norm (optional): If True, the monotonic kernel is scaled such that the
                norm of the product of the monotonic strength kernel and the probability kernel
                is equal to the norm of the product of the non-monotonic strength kernel and the
                probability kernel. Ignored if monotonic_strength is False.
            monotonic_norm_ord (optional): Order of the vector norm used for `keep_monotonic_norm`.
            dense (optional): If True, connectivity is dense, and `N_synapses` must be None.
            N_synapses (optional):
                Expected number of synapses per neuron, must be non-negative. If not None, `dense` must be False.
            W_std (optional): Standard deviation (as a fraction of the mean) of connection weight distribution.
            seed (optional):
                Random seed for generating connection weight matrix, only relevant if N_synapses is not None or W_std > 0.
            sparsify_kwargs (optional): keyword arguments passed to weights.sparsify.
            nonlinear_kwargs (optional): keyword arguments passed to numerics.perturbed_steady_state_approx.
            simulation_kwargs (optional): keyword arguments passed to numerics.perturbed_response.
            wrapped_kwargs (optional): keyword arguments passed to functional.wrapped.
            batch_shape (optional): Shape of model batch dimensions.

        """
        # validate inputs
        if any(v not in {"cell_type", "space", "ori", "osi"} for v in variables):
            raise ValueError(
                "variables must be a subset of ['cell_type', 'space', 'ori', 'osi'],"
                f" but got {variables=}."
            )

        if "osi" in variables and "ori" not in variables:
            raise ValueError(
                "If 'osi' is present in variables, 'ori' must also be present."
            )

        if ori_func not in {"cosine", "von_mises"} and not callable(ori_func):
            raise ValueError(
                "ori_func must be 'cosine', 'von_mises', or a callable, but got "
                f"{ori_func=}."
            )

        if ori_order is None and callable(ori_func):
            raise ValueError("If ori_func is callable, ori_order must be specified.")

        if not (
            isinstance(sigma_symmetry, (Tensor, Sequence))
            or sigma_symmetry in {"pre", "post", "full", None}
        ):
            raise ValueError(
                "sigma_symmetry must be a Tensor, Sequence, 'pre', 'post', 'full', or"
                f"None, but got {sigma_symmetry=}."
            )

        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, but {mode=}.")

        if W_std < 0.0:
            raise ValueError(f"W_std must be non-negative, but got {W_std=}.")

        if (torch.as_tensor(rf_cv) < 0.0).any():
            raise ValueError(f"rf_cv must be non-negative, but got {rf_cv=}.")

        if space_strength_kernel is not None and mode not in {"numerical", "matrix"}:
            raise ValueError(
                "space_strength_kernel must be None if mode is not 'numerical' or"
                f" 'matrix', but got {space_strength_kernel=}, {mode=}."
            )

        if prob_kernel and not set(prob_kernel.keys()).issubset(
            ["cell_type"] + variables
        ):
            raise ValueError(
                f"prob_kernel keys must be a subset of {'cell_type'} | set(variables), "
                f"but got {prob_kernel.keys()=}."
            )

        if (
            monotonic_strength
            and "space" in prob_kernel
            and not isinstance(prob_kernel["space"], Radial)
        ):
            raise ValueError(
                "If monotonic_strength is True and prob_kernel['space'] is provided, "
                "prob_kernel['space'] must be a Radial kernel, but "
                f"{type(prob_kernel['space'])=}."
            )

        if dense and N_synapses is not None:
            raise ValueError("`N_synapses` must be none if `dense` is True.")

        if wrapped_kwargs is not None:
            raise NotImplementedError("wrapped_kwargs is not currently implemented.")

        cell_types = tuple(
            CellType[ct] if isinstance(ct, str) else ct for ct in cell_types
        )

        # initialize defaults
        if "cell_type" not in variables:
            cell_types = [cell_types[0]]
        n = len(cell_types)

        if isinstance(tau, float):
            tau = [tau] * n

        if isinstance(osi_func, Distribution):
            if not (
                isinstance(osi_func.support, constraints.interval)
                and osi_func.support.lower_bound == 0.0
                and osi_func.support.upper_bound == 1.0
            ):
                raise ValueError(
                    "osi_func must be a Distribution with support [0, 1], "
                    f"but got {osi_func.support=}"
                )
            osi_func = osi_func.cdf
        elif not isinstance(osi_func, float) and not callable(osi_func):
            osi_func = utils.call(nn, osi_func)

        if isinstance(osi_prob, Sequence):
            osi_prob = getattr(torch.distributions, osi_prob[0])(
                *[torch.as_tensor(v) for v in osi_prob[1:]]
            )
        if len(osi_prob.batch_shape) not in {0, 1}:
            raise ValueError(
                f"osi_prob must have 0/1D batch_shape, but got {osi_prob.batch_shape=}."
            )

        if not callable(f):
            f = utils.call(nn, f)

        if isinstance(f, nn.Match) and "cell_type" not in variables:
            raise ValueError("nn.Match nonlinearity requires 'cell_type' variable.")

        if isinstance(sigma_symmetry, Sequence) and not isinstance(sigma_symmetry, str):
            sigma_symmetry = torch.tensor(sigma_symmetry)

        if isinstance(sigma_symmetry, Tensor):
            if sigma_symmetry.shape != (n, n):
                raise ValueError(
                    "sigma_symmetry must have shape (n, n), but got"
                    f" {sigma_symmetry.shape=}."
                )
            if sigma_symmetry.dtype != torch.long:
                raise ValueError(
                    "sigma_symmetry must have dtype torch.long, but got"
                    f" {sigma_symmetry.dtype=}."
                )
            m = sigma_symmetry.max().item() + 1
            if set(sigma_symmetry.reshape(-1).tolist()) != set(range(m)):
                raise ValueError(
                    "sigma_symmetry must contain consecutive integers starting from 0."
                )

        if mode.endswith("approx") and isinstance(f, nn.Identity):
            raise ValueError("f must be nonlinear when using approximation modes.")

        if null_connections is None:
            null_connections = filter(
                lambda ct: ct[0] not in ct[1].targets,
                itertools.product(cell_types, cell_types),
            )
        else:
            null_connections = (
                (
                    CellType[cti] if isinstance(cti, str) else cti,
                    CellType[ctj] if isinstance(ctj, str) else ctj,
                )
                for cti, ctj in null_connections
            )

        if sigma_optim is None:
            sigma_optim = "space" in variables

        if kappa_optim is None:
            kappa_optim = "ori" in variables

        if vf_optim is None:
            vf_optim = not isinstance(f, nn.Identity)

        if prob_kernel is None:
            prob_kernel = {}

        if monotonic_kwargs is None:
            monotonic_kwargs = {}

        super().__init__()

        self.variables = list(variables)
        self.cell_types = cell_types
        self.ori_func = ori_func
        self.ori_order = 2 if ori_order is None and ori_func == "cosine" else ori_order
        self._osi_func = osi_func
        self.osi_prob = osi_prob
        self.f = f
        self.autapse = autapse
        self.mode = mode
        self.N_synapses = N_synapses
        self.dense = dense
        self.W_std = W_std
        self.seed = seed
        self.sparsify_kwargs = sparsify_kwargs if sparsify_kwargs else {}
        self.nonlinear_kwargs = nonlinear_kwargs if nonlinear_kwargs else {}
        self.simulation_kwargs = simulation_kwargs if simulation_kwargs else {}
        self.wrapped_kwargs = wrapped_kwargs if wrapped_kwargs else {}

        # define model parameters
        # note that gW refers to the matrix GW, where G = diag(gain). The lowercase
        # g is due to the fact that in older versions of this code, gain is a scalar.
        self.gW = nn.Parameter(
            torch.empty((*batch_shape, n, n)),
            requires_optim=gW_optim,
            bounds=gW_bounds,
            tag="gW",
        )  # (*, n, n)
        for cti, ctj in null_connections:
            self.gW.requires_optim[
                ..., cell_types.index(cti), cell_types.index(ctj)
            ] = False

        sigma_shape = {"pre": (1, n), "post": (n, 1), "full": (1, 1), None: (n, n)}
        sigma_shape = (
            (m,) if isinstance(sigma_symmetry, Tensor) else sigma_shape[sigma_symmetry]
        )
        self.sigma = nn.Parameter(
            torch.empty((*batch_shape, *sigma_shape)),
            requires_optim=sigma_optim,
            bounds=sigma_bounds,
            tag="sigma",
        )  # (*, 1 or n, 1 or n) or (*, m)

        self.kappa = nn.Parameter(
            torch.empty((*batch_shape, n, n)),
            requires_optim=kappa_optim,
            bounds=kappa_bounds,
            tag="kappa",
        )  # (*, n, n)

        self.vf = nn.Parameter(
            torch.empty(() if vf_symmetry or n == 1 else (n,)),
            requires_optim=vf_optim,
            bounds=vf_bounds,
            tag="vf",
        )

        self.register_buffer("tau", torch.tensor(tau), persistent=False)  # (n,)
        self.register_buffer(
            "sign",
            torch.tensor([ct.sign for ct in cell_types]).float(),
            persistent=False,
        )  # (*, n)
        if isinstance(sigma_symmetry, Tensor):
            self.register_buffer("sigma_symmetry", sigma_symmetry, persistent=False)
        else:
            self.sigma_symmetry = sigma_symmetry

        # define model kernels
        if isinstance(space_strength_kernel, str):
            space_strength_kernel = getattr(nn, space_strength_kernel)

        if space_strength_kernel is not None and not issubclass(
            space_strength_kernel, Kernel
        ):
            raise TypeError("space_strength_kernel must be a Kernel subclass or None.")

        has_ct = "cell_type" in self.variables

        space_prob_kernel = prob_kernel.get("space", 1)
        sqrtS = nn.Matrix(self.sqrtS, "cell_type") if has_ct else nn.Scalar(self.sqrtS)
        if space_strength_kernel is None:
            if autapse:
                k = nn.AutapsedLaplace(
                    sqrtS, ["space", "space_dV"], normalize="integral"
                )
            else:
                k = nn.Laplace(sqrtS, "space", normalize="integral")
            if monotonic_strength:
                space_strength_kernel = nn.Monotonic(
                    k / space_prob_kernel, "space", **monotonic_kwargs
                )
                if keep_monotonic_norm:
                    space_strength_kernel = (
                        space_strength_kernel
                        * nn.Norm(k, "cell_type", ord=monotonic_norm_ord)
                        / nn.Norm(
                            space_strength_kernel * space_prob_kernel,
                            "cell_type",
                            ord=monotonic_norm_ord,
                        )
                    )
                space_product_kernel = space_strength_kernel * space_prob_kernel
            else:
                space_product_kernel = k
                space_strength_kernel = space_product_kernel / space_prob_kernel
        else:
            space_strength_kernel = space_strength_kernel(
                sqrtS, "space", normalize="integral"
            )
            space_product_kernel = space_strength_kernel * space_prob_kernel

        if self.ori_func == "cosine" or (
            self.ori_func == "von_mises" and self.ori_order is None
        ):
            func = self.kappa_
        else:
            func = self.fourier_coefs
        tuning_kernel = nn.Matrix(func, "cell_type") if has_ct else nn.Scalar(func)

        if self.ori_func == "cosine":
            tuning_kernel = nn.Cosine(tuning_kernel, "ori")
        elif self.ori_func == "von_mises" and self.ori_order is None:
            tuning_kernel = nn.VonMises(tuning_kernel, "ori")
        else:
            tuning_kernel = nn.Tuning(tuning_kernel, "ori")

        if "osi" in variables:
            tuning_kernel = nn.Scaled(
                tuning_kernel, nn.RankOne(self.osi_func, x_keys="osi"), "ori"
            )

        product_kernel = {
            "cell_type": (
                nn.Matrix(self.W, "cell_type") if has_ct else nn.Scalar(self.W)
            ),
            "space": space_product_kernel,
            "ori": tuning_kernel,
        }
        strength_kernel = {
            "cell_type": product_kernel["cell_type"] / prob_kernel.get("cell_type", 1),
            "space": space_strength_kernel,
            "ori": product_kernel["ori"] / prob_kernel.get("ori", 1),
        }

        def filt(x):
            return x[0] in {"cell_type"} | set(self.variables)

        self.product_kernel = nn.Prod(dict(filter(filt, product_kernel.items())))
        self.prob_kernel = nn.Prod(dict(filter(filt, prob_kernel.items())))
        self.strength_kernel = nn.Prod(dict(filter(filt, strength_kernel.items())))

        # initialize model parameters
        self.init_gW_std = init_gW_std
        self.init_gW_bounds = gW_bounds if init_gW_bounds is None else init_gW_bounds
        self.init_sigma_bounds = (
            sigma_bounds if init_sigma_bounds is None else init_sigma_bounds
        )
        self.init_kappa_bounds = (
            kappa_bounds if init_kappa_bounds is None else init_kappa_bounds
        )
        self.register_buffer("init_vf", torch.as_tensor(init_vf), persistent=False)
        self.register_buffer("rf_cv", torch.as_tensor(rf_cv), persistent=False)
        self.init_stable = init_stable
        self.reset_parameters()

    @property
    def n(self):
        return self.gW.shape[-1]

    @property
    def batch_shape(self):
        return self.gW.shape[:-2]

    @property
    def batch_ndim(self):
        return len(self.batch_shape)

    @property
    def sigma_(self):
        return (
            self.sigma[self.sigma_symmetry]
            if isinstance(self.sigma_symmetry, Tensor)
            else self.sigma
        )

    @property
    def S(self):
        batch_shape, n = self.batch_shape, self.n
        return (self.sigma_**2).broadcast_to(*batch_shape, n, n)

    def osi_scale(self, p=2):
        if "osi" not in self.variables:
            return 1.0

        return compute_osi_scale(
            self.osi_prob,
            osi_func=self._osi_func,
            p=p,
            device=self.gW.device,
            dtype=self.gW.dtype,
        )  # osi_prob.batch_shape

    @property
    def osi_func(self):
        if isinstance(self._osi_func, float):
            if self.osi_prob.batch_shape != ():
                if not isinstance(self.osi_prob, torch.distributions.Beta):
                    raise NotImplementedError(
                        "Only supports Beta distribution for now."
                    )
                osi_prob = type(self.osi_prob)(
                    self.osi_prob.concentration1[0], self.osi_prob.concentration0[0]
                )
            else:
                osi_prob = self.osi_prob
            return lambda x: osi_prob.cdf(x) ** self._osi_func
        return self._osi_func

    def kappa_(self) -> Tensor:
        return self.kappa

    def fourier_coefs(self) -> Tensor:
        if self.ori_order is None:
            raise ValueError(
                "ori_order must be an integer to compute Fourier coefficients."
            )

        # Return Fourier series coefficients
        if self.ori_func == "cosine":
            assert self.ori_order == 2
            out = torch.stack([torch.ones_like(self.kappa), self.kappa])
        elif self.ori_func == "von_mises":
            out = torch.stack(
                [special.iv(i, self.kappa) for i in range(self.ori_order)]
            ) / torch.special.modified_bessel_i0(self.kappa)
        else:
            out = torch.stack(
                [self.ori_func(i, self.kappa) for i in range(self.ori_order)]
            )
        return out  # (2 | ori_order, *batch_shape, n, n)

    def W(self, with_gain: bool = False, **kwargs) -> Tensor:
        W = self.gW * self.sign[..., None, :]  # (*batch_shape, n, n)
        if not with_gain:
            W = W / self.gain(**kwargs)[..., None]  # (*batch_shape, n, n)
        return W

    def sqrtS(self) -> Tensor:
        batch_shape, n = self.batch_shape, self.n
        return self.sigma_.broadcast_to(*batch_shape, n, n)

    def reset_parameters(self):
        nn.init.copy_(self.vf, self.init_vf)
        nn.init.W_(
            self.gW, self.init_gW_std, self.init_gW_bounds, self.gW.requires_optim
        )
        nn.init.uniform_(self.sigma, self.init_sigma_bounds, self.sigma.requires_optim)
        nn.init.uniform_(self.kappa, self.init_kappa_bounds, self.kappa.requires_optim)

        if self.init_stable and self.spectral_summary().abscissa.item() >= 1.0:
            self.reset_parameters()

    def gain(self, dh: Tensor | float = 0.0, create_graph=None, **kwargs) -> Tensor:
        """Compute gain of the model.

        Args:
            dh: Evaluate gain at self.vf + dh. If Tensor, must have shape () or (self.n,)
            **kwargs: keyword arguments to torch.autograd.functional.jacobian

        Returns:
            torch.Tensor: Tensor with shape () or (n,)

        """
        f, vf = self.f, self.vf
        if isinstance(f, nn.Match):
            f = functools.partial(
                f.forward,
                key=categorical.tensor(
                    list(range(self.n)),
                    categories=[ct.name for ct in self.cell_types],
                    device=self.vf.device,
                ),
            )
            vf = vf.broadcast_to((self.n,))
        create_graph = vf.requires_grad if create_graph is None else False
        return numerics.compute_gain(f, vf + dh, create_graph=create_graph, **kwargs)

    def kth_deriv(self, k: int) -> Tensor:
        """Compute kth-derivative of the nonlinearity.

        Should be equivalent to, though slower than, self.gain(dh=0.0) when k == 1.

        Args:
            k: Order of the derivative to compute.

        Returns:
            Tensor with shape () or (n,)

        """
        f, vf, f_args = self.f, self.vf, ()
        if isinstance(f, nn.Match):
            f_args = (
                categorical.tensor(
                    list(range(self.n)),
                    categories=[ct.name for ct in self.cell_types],
                    device=self.vf.device,
                ),
            )
            vf = vf.broadcast_to((self.n,))
        return numerics.compute_nth_deriv(f, vf, f_args, n=k)  # () or (n,)

    def resolvent(
        self,
        l: Number,
        x: ParameterFrame,
        y: ParameterFrame,
        with_gain: bool = True,
        checkpoint: bool = False,
        **kwargs,
    ) -> Tensor:
        W = self.W(with_gain=with_gain, **kwargs)  # (*batch_shape, n, n)
        sigma = self.sigma_  # (*batch_shape, n, n)
        kappa = self.fourier_coefs()  # (ori_order, *batch_shape, n, n)

        if self.batch_ndim > 0:
            idx = (slice(None),) * self.batch_ndim + (None,) * x.ndim
            W, sigma, kappa = W[idx], sigma[idx], kappa[(slice(None),) + idx]

        return resolvent(
            l,
            x,
            y,
            W,
            sigma,
            kappa,
            self.osi_func,
            self.osi_scale(),
            autapse=self.autapse,
            checkpoint=checkpoint,
            **self.wrapped_kwargs,
        )  # (*self.batch_shape, *broadcast_shapes(x.shape, y.shape))

    def spectral_summary(
        self,
        cell_types: Iterable[CellType | str] | None = None,
        kind: str = "W",
        dh: Tensor | float = 0.0,
        **kwargs,
    ) -> SpectralSummary:
        """Compute spectral abscissa and radius of the model's connectivity or jacobian operator.

        Args:
            cell_types (optional): If not None, restrict to subcircuit composed of only
              the specified cell-types.
            kind (optional): {'W', 'J'}. Whether to compute the spectrum of the
              connectivity or jacobian operator.
            dh (optional): If specified, compute the spectrum at self.vf + dh.
            **kwargs: Optional arguments passed to v1.spectral_summary

        Returns:
            Named tuple with fields 'abscissa' and 'radius':
              abscissa: Tensor with shape self.batch_shape
              radius: Tensor with shape self.batch_shape

        """
        if kind not in {"W", "J"}:
            raise ValueError(f"kind must be either 'W' or 'J', but got {kind=}.")

        if (self.rf_cv > 0).any():
            raise NotImplementedError(
                "spectral summary is not implemented for rf_cv > 0."
            )

        if cell_types:
            cell_types = (
                CellType[ct] if isinstance(ct, str) else ct for ct in cell_types
            )
            cell_types = (self.cell_types.index(ct) for ct in cell_types)

        W = self.gW * self.sign[..., None, :]  # (*batch_shape, n, n)
        if not (isinstance(dh, float) and dh == 0.0):
            W = W / self.gain(**kwargs)[..., None]  # (*batch_shape, n, n)
            W = W * self.gain(dh=dh, **kwargs)[..., None]  # (*batch_shape, n, n)

        return spectral_summary(
            W,
            S=(self.S if "space" in self.variables else None),
            kappa=(self.fourier_coefs() if "ori" in self.variables else None),
            osi_scale=(self.osi_scale() if "osi" in self.variables else None),
            tau=(self.tau if kind == "J" else None),
            cell_types=cell_types,
            **kwargs,
        )

    def spectral_norm(
        self,
        cell_types: Iterable[CellType | str] | None = None,
        Ginv: bool = True,
        H: bool = True,
        **kwargs,
    ) -> Tensor:
        r"""Compute spectral norm of the linear response operator $\tilde{L}$.

        Args:
            cell_types (optional): If not None, right-multiply $\tilde{L}$ by $D$, where
              $D$ is a diagonal matrix with value 1 if it corresponds to a neuron with
              cell type within cell_types, and 0 otherwise.
            Ginv (optional): If True, left-multiply $\tilde{L}$ by $G^{-1}$, where $G$
              is the diagonal matrix of neuronal gains.
            H (optional): If True, right-multiply $\tilde{L}$ by $H$, where $H$ is the
              diagonal matrix of f''(vf)/2, where f is the nonlinearity.
            **kwargs: Optional arguments passed to v1.spectral_norm

        Returns:
            Tensor with shape self.batch_shape

        """
        if self.ori_func == "von_mises" or callable(self.ori_func):
            raise NotImplementedError(
                f"spectral norm is not implemented for {self.ori_func}."
            )

        if (self.rf_cv > 0).any():
            raise NotImplementedError("spectral norm is not implemented for rf_cv > 0.")

        b = self.kth_deriv(2) / 2 if H else 1.0
        if cell_types:
            cell_types = {
                CellType[ct] if isinstance(ct, str) else ct for ct in cell_types
            }
            b = b * torch.tensor(
                [1.0 if ct in cell_types else 0.0 for ct in self.cell_types],
                device=self.gW.device,
            )

        return spectral_norm(
            self.gW * self.sign[..., None, :],
            sigma=(self.sigma_ if "space" in self.variables else None),
            kappa=(self.kappa if "ori" in self.variables else 0.0),
            osi_scale=(self.osi_scale() if "osi" in self.variables else 1.0),
            a=(self.kth_deriv(1).reciprocal() if Ginv else 1.0),
            b=b,
            **kwargs,
        )

    def forward(
        self,
        x: ParameterFrame,
        t: Sequence[float] = (),
        avg_t: bool = False,
        output: str = "response",
        ndim: int = 1,
        in_var: str = "dh",
        out_var: str = "dr",
        dv_var: str = "dv",
        time_var: str = "t",
        mask_var: str = "mask",
        vf_var: str = "vf",
        rf_var: str = "rf",
        drr_var: str = "drf/rf",
        check_circulant: bool = True,
        assert_finite: bool = True,
        to_dataframe: str | bool = True,
        checkpoint: bool = False,
        return_dv: bool = False,
        newton_kwargs: dict | None = None,
    ) -> ParameterFrame | tdfl.DataFrame | pd.DataFrame | Tensor:
        """Compute perturbation response or connectivity weights of the model.

        Output is computed for each set of model parameters, i.e. it is equivalent
        to computing output for each param[*idx] where idx in np.ndindex(self.batch_shape)
        and param is a model parameter ("gW", "sigma", "kappa")

        Args:
            x: ParameterFrame containing zero-dimensional columns {in_var, 'dV'}
            t (optional): Sequence of time points at which to compute the response.
                If empty, computes the steady-state response. Only valid if
                output == 'response' and mode == 'numerical'.
            avg_t: (optional): If True, the average of all time points specified by `t`
                is computed.
            output (optional): {'response', 'weight'}. Whether to compute perturbation
                response or connectivity weights.
            ndim (optional): Number of non-batch dimensions (assumed to be trailing).
            in_var (optional): Column name of input. Ignored if output == 'weight'.
            out_var (optional): Column name of output. Ignored if output == 'weight'.
            time_var (optional): Column name of time. Ignored if t is empty.
            mask_var (optional):
                Column name of output mask. Ignored if output == 'weight'. If mask_var in x,
                only returns neurons where x[mask] == True. This improves efficiency of computing
                analytic linear response when only the responses of a subset of neurons are desired.
            vf_var (optional): Column name of baseline synaptic inputs.
            rf_var (optional): Column name of baseline firing rates.
            drr_var (optional): Ignored if output == "weight". Not allowed if drr_var in x
                and output == "response" and self.mode == "analytical" or contains
                "approx". Otherwise, if drr_var in x, calculate response of a network with
                W = self.W() and vf = finv(rf + rf * x[drr_var]), where rf = f(self.vf).
            check_circulant (optional):
                Check if the connectivity is circulant and use circulant matrix optimizations if so.
                Set to False if it is known to not be circulant to improve performance.
            assert_finite (optional): Raise error if Infs or NaNs are present in model output.
                Ignored if output == 'weight'.
            to_dataframe (optional): If True, returns a tdfl.DataFrame. If a str, must
                be either 'tdfl' or 'pandas'. If 'pandas', returns a pandas.DataFrame.
            checkpoint (optional): If True, uses activation checkpointing to save GPU
                memory if gradient is required.
            return_dv (optional): If True, additionally return a column with name dv_var
                containing the change in total recurrent input due to the perturbation,
                W @ out["out"].
            newton_kwargs (optional): Keyword arguments passed to elementwise.newton.

        Returns:
            If output == 'response', a DataFrame with column out_var added. If
            to_dataframe is False, then output is instead a ParamaterFrame.
            Otherwise, a DataFrame with columns ['W', 'presynaptic_cell_type',
            'postsynaptic_cell_type', 'distance', 'rel_ori', 'presynaptic_osi',
            'postsynaptic_osi'] (excluding columns incompatible with model
            variables, e.g. if 'ori' is not in self.variables, then 'rel_ori'
            is not an output column). If to_dataframe is False, then output
            is instead a Tensor of connectivity weights.

        """
        if output not in {"response", "weight", "jacobian"}:
            raise ValueError(
                "output must be either 'response', 'weight', or 'jacobian', but got "
                f"{output=}."
            )

        if len(t) > 0 and (output != "response" or self.mode != "numerical"):
            raise ValueError(
                "t must be empty if output != 'response' or mode != 'numerical', "
                f"but {t=}."
            )

        if isinstance(to_dataframe, str) and to_dataframe not in {"tdfl", "pandas"}:
            raise ValueError(
                "to_dataframe must be either 'tdfl', 'pandas', or bool, but got "
                f"{to_dataframe=}."
            )
        # don't keep indices if output is connectivity weights to save memory
        framelike_kwargs = {
            "cls": pd.DataFrame if to_dataframe == "pandas" else tdfl.DataFrame,
            "keep_indices": to_dataframe == "pandas" and output == "response",
            "to_numpy": to_dataframe == "pandas",
        }

        if (
            output == "response"
            and self.mode != "numerical"
            and (self.N_synapses or self.W_std)
        ):
            raise ValueError(
                "N_synapses must be None and W_std must be 0 when output == 'response'"
                " and mode != 'numerical'."
            )

        keys = self.variables + ["dV"]
        if "space" in self.variables and "space_dV" in x:
            keys += ["space_dV"]

        if check_circulant:
            cdim = _cdim(x.data[self.variables], ndim, rtol=1.0e-2, atol=1.0e-8)
        else:
            cdim = ()
        logger.debug(f"Assuming circulant dimensions are {cdim}, with {x.shape=}")

        x_ = x.data[keys]  # (*, *shape)

        f, f_, f_args = self.f, self.f, ()
        if isinstance(f, nn.Match):
            # ensure nn.Match nonlinearities gets called with cell type information
            f_ = functools.partial(f, key=x_["cell_type"])
            f_args = (x_["cell_type"],)

        # expand vf to shape of x_ if vf is not a scalar
        vf = self.vf[x_["cell_type"]] if self.vf.ndim > 0 else self.vf
        if drr_var in x:
            rf = f(vf, *f_args) * (1 + x.data[drr_var])
            if hasattr(f, "inv"):
                vf = f.inv()(rf)
            else:
                vf = vf.broadcast_to(rf.shape)

                def _func(vf, rf, *f_args):
                    return f(vf, *f_args) - rf

                vf = elementwise.newton(_func, vf, args=(rf, *f_args), **newton_kwargs)

        if (self.rf_cv > 0).any():
            # Sample vf per-neuron from a Gaussian N(mu, sigma^2) chosen so that the
            # baseline firing rate has mean f(vf) and variance (rf_cv * f(vf))**2.
            allowed_modes = {"numerical", "matrix", *MATRIX_APPROX_MODES}
            if output == "response" and self.mode not in allowed_modes:
                raise NotImplementedError(
                    f"rf_cv > 0 is only supported when mode is one of {allowed_modes}, "
                    f"but got {self.mode=}."
                )
            rf_cv = self.rf_cv[x_["cell_type"]] if self.rf_cv.ndim > 0 else self.rf_cv
            rf_mean = f(vf, *f_args)  # target mean of f(vf)
            rf_variance = (rf_cv * rf_mean) ** 2  # target variance of f(vf)
            mu, sigma = gm.gaussian_pullback(
                f, rf_mean, rf_variance, args=f_args, mu_init=vf
            )
            noise = torch.randn(x.shape, dtype=sigma.dtype, device=sigma.device)
            vf = mu + sigma * noise
        x[vf_var] = vf[(None,) * (x.ndim - vf.ndim) + (...,)]
        rf = f(vf, *f_args)
        x[rf_var] = rf[(None,) * (x.ndim - rf.ndim) + (...,)]

        if output == "response" and self.mode in {"analytical", *APPROX_MODES}:
            if drr_var in x:
                raise ValueError(
                    f"Passing {drr_var} is not allowed when output == 'response' and "
                    f"self.mode == {self.mode}."
                )

            linear = isinstance(f, nn.Identity) or self.mode.endswith("linear_approx")
            if not linear:
                # note that this is only approximate
                nonlinear_mask = numerics.compute_nth_deriv(f, vf, f_args, n=2) != 0

            if linear or not nonlinear_mask.all():
                bcastdim = [
                    d
                    for d, (n0, n1) in enumerate(zip(x_.shape[:-ndim], x.shape[:-ndim]))
                    if n0 != n1
                ]

                dh = x.data[in_var]  # (*, *shape)
                assert isinstance(dh, Tensor)
                in_mask = (dh != 0).any(dim=bcastdim, keepdim=True)  # (*, *shape)
                if mask_var in x:
                    out_mask = x.data[mask_var]
                    out_mask = out_mask.any(dim=bcastdim, keepdim=True)  # (*, *shape)
                else:
                    out_mask = torch.ones_like(in_mask)
                out_mask = out_mask | in_mask

                if not linear:
                    in_mask = in_mask | nonlinear_mask
                    out_mask = out_mask | nonlinear_mask

                # topk requires integer or float dtypes
                in_mask = in_mask.to(torch.int8)
                out_mask = out_mask.to(torch.int8)

                x_ = x_.reshape(*x_.shape[:-ndim], -1)  # (*, N)
                dh = dh.reshape(*dh.shape[:-ndim], -1)  # (*, N)
                in_mask = in_mask.reshape(*in_mask.shape[:-ndim], -1)  # (*, N)
                out_mask = out_mask.reshape(*out_mask.shape[:-ndim], -1)  # (*, N)
                vf = vf if vf.ndim == 0 else vf.reshape(*vf.shape[:-ndim], -1)
                f_args = (arg.reshape(*arg.shape[:-ndim], -1) for arg in f_args)

                N_out = out_mask.sum(dim=-1).max().item()
                _, out_idx = out_mask.topk(N_out, dim=-1, sorted=False)  # (*, N_out)
                x_post = x_.take_along_dim(out_idx, dim=-1)  # (*, N_out)
                dh = dh.take_along_dim(out_idx, dim=-1)  # (*, N_out)
                in_mask = in_mask.take_along_dim(out_idx, dim=-1)  # (*, N_out)
                vf = vf if vf.ndim == 0 else vf.take_along_dim(out_idx, dim=-1)
                f_args = tuple(arg.take_along_dim(out_idx, dim=-1) for arg in f_args)

                N_in = in_mask.sum(dim=-1).max().item()
                _, idx = in_mask.topk(N_in, dim=-1, sorted=False)  # (*, N_in)
                x_pre = x_post.take_along_dim(idx, dim=-1)  # (*, N_in)

                mdims = [(-1, None)] * 2
                x_post, x_pre = frame.meshgrid(x_post, x_pre, sparse=True, dims=mdims)
                logger.debug(f"{x_post.shape=}, {x_pre.shape=}")
                tL = self.resolvent(-1, x_post, x_pre, checkpoint=checkpoint)
                tL = tL * x_pre.data["dV"]  # (*bshape, *, N_out, N_in)

                dr = torch.empty(
                    (*self.batch_shape, *x.shape[:-ndim], x_.shape[-1]),
                    dtype=dh.dtype,
                    device=dh.device,
                )
                if return_dv:
                    dv = torch.empty_like(dr)
                kwargs = {
                    "f_args": f_args,
                    "indices": idx,
                    "mode": self.mode,
                    "return_dv": return_dv,
                }
                _dr = numerics.perturbed_steady_state_approx(
                    vf, tL, f, dh, **kwargs, **self.nonlinear_kwargs
                )  # (*bshape, *, N_out)
                if return_dv:
                    _dr, _dv = _dr
                # Perform broadcasting manually since torch.Tensor.scatter_ does not
                # perform broadcasting between index and src tensors.
                out_idx = out_idx.broadcast_to(_dr.shape)
                dr.scatter_(-1, out_idx, _dr)
                dr = dr.reshape((*dr.shape[:-1], *x.shape[-ndim:]))
                if return_dv:
                    dv.scatter_(-1, out_idx, _dv)
                    dv = dv.reshape((*dv.shape[:-1], *x.shape[-ndim:]))

            else:
                kernel = functools.partial(self.resolvent, -1, checkpoint=checkpoint)
                tL = weights.discretize(
                    kernel, x_, ndim=ndim, dim=cdim
                )  # (I - GW)^{-1} - I, (*bshape, *, *shape, *shape)
                dh = x.data[in_var]  # (*, *shape)
                kwargs = {"f_args": f_args, "mode": self.mode, "return_dv": return_dv}
                dr = numerics.perturbed_steady_state_approx(
                    vf, tL, f, dh, **kwargs, **self.nonlinear_kwargs
                )  # (*bshape, *, *shape)
                if return_dv:
                    dr, dv = dr

        elif output == "response" and self.mode in {"matrix", *MATRIX_APPROX_MODES}:
            W = self.forward(
                x,
                output="weight",
                ndim=ndim,
                check_circulant=check_circulant,
                checkpoint=checkpoint,
                to_dataframe=False,
            )
            I = linalg.eye_like(W)  # (*bshape, *, *shape, *shape)
            dh = x.data[in_var]  # (*, *shape)

            if isinstance(f, nn.Identity):
                # (*bshape, *, *shape)
                dr = torch.linalg.solve(I - W, dh.unsqueeze(-1)).squeeze(-1)
            else:
                if isinstance(W, CirculantTensor):
                    raise NotImplementedError()
                else:
                    G = numerics.compute_nth_deriv(f, vf, f_args)  # () or (*, N)
                    tL = torch.linalg.inv(I - G[..., None] * W) - I
                mode = "analytical" if self.mode == "matrix" else self.mode[7:]
                dr = numerics.perturbed_steady_state_approx(
                    vf, tL, f, dh, f_args=f_args, mode=mode, **self.nonlinear_kwargs
                )  # (*bshape, *, *shape)
            if return_dv:
                dv = linalg.bmv(W, dr)

        elif output == "response" and self.mode in {"newton", "broyden", "gd"}:
            # Newton's method to solve for steady-state response
            W = self.forward(
                x,
                output="weight",
                ndim=ndim,
                check_circulant=check_circulant,
                checkpoint=checkpoint,
                to_dataframe=False,
            )

            if isinstance(W, CirculantTensor):
                raise ValueError(
                    "Circulant weights not supported since there is no speedup "
                    "afforded by the circulant structure when mode in {'newton', "
                    "'broyden', 'gd'}."
                )

            dh = x.data[in_var]  # (*, N)
            dr = numerics.solve_perturbed_steady_state(
                vf, W, f, dh, f_args=f_args, method=self.mode, **self.nonlinear_kwargs
            )  # (*bshape, *, N)
            if return_dv:
                dv = linalg.bmv(W, dr)

        elif output == "response" and self.mode in {
            "broyden1",
            "broyden2",
            "linearmixing",
        }:
            # shape of W: (*bshape, *, *shape, *shape)
            W = self.forward(
                x,
                output="weight",
                ndim=ndim,
                check_circulant=check_circulant,
                checkpoint=checkpoint,
                to_dataframe=False,
            )
            dh = x.data[in_var]  # (*, *shape)
            rf = f(vf, *f_args)  # () or (*, *shape)

            def func(dr):
                return rf + dr - f(vf + linalg.bmv(W, dr) + dh, *f_args)

            shape = (*self.batch_shape, *x.shape)
            dr = torch.zeros(shape, dtype=dh.dtype, device=dh.device)
            dr = optimize.rootfinder(
                func, dr, method=self.mode, **self.nonlinear_kwargs
            )  # (*bshape, *, *shape)
            if return_dv:
                dv = linalg.bmv(W, dr)

        else:
            if self.dense or len(self.prob_kernel.funcs) == 0:
                if checkpoint:
                    W = torch.utils.checkpoint.checkpoint(
                        weights.discretize,
                        self.product_kernel,
                        x_,
                        ndim=ndim,
                        dim=cdim,
                        use_reentrant=False,
                    )
                else:
                    W = weights.discretize(
                        self.product_kernel, x_, ndim=ndim, dim=cdim
                    )  # (*bshape, *, *shape, *shape)
            else:
                # Note: this is non-differentiable.
                prob = weights.discretize(
                    self.prob_kernel, x_, ndim=ndim, dim=cdim, mul_dV=False
                )
                W = weights.discretize(
                    self.strength_kernel, x_, ndim=ndim, dim=cdim
                )  # (*bshape, *, *shape, *shape)

                if isinstance(prob, CirculantTensor):
                    prob = prob.dense()
                if isinstance(W, CirculantTensor):
                    W = W.dense()

                if (prob < -1e-5).any() or (prob > 1 + 1e-5).any():
                    raise ValueError("Connection probability must be in [0, 1].")
                prob = prob.clip(min=0.0, max=1.0)

                W = W * torch.bernoulli(prob)  # (*bshape, *, *shape, *shape)

            if self.N_synapses is not None:
                # Note: this is currently non-differentiable.
                mean_p = self.N_synapses / math.prod(x.shape[-ndim:])
                with random.set_seed(self.seed):
                    W = weights.sparsify(W, mean_p=mean_p, **self.sparsify_kwargs)
                logger.debug(
                    f"{W.count_nonzero().item()}/{W.numel()} ="
                    f" {W.count_nonzero().item() / W.numel()} fraction of connected"
                    " neurons"
                )

            if self.W_std > 0.0:
                with random.set_seed(self.seed):
                    W = weights.sample_log_normal(W, self.W_std)

            if output == "weight":
                if not to_dataframe:
                    return W

                dims = [(-ndim, None), (-ndim, None)]
                x = x.data[self.variables + ([mask_var] if mask_var in x else [])]
                x_post, x_pre = frame.meshgrid(x, x, dims=dims, sparse=True)

                # TODO: Think about how to make this more flexible, allowing for
                # arbitrary columns instead of hardcoding everything.
                x = {}
                for k in ["cell_type", "osi"]:
                    if k in self.variables:
                        x[f"presynaptic_{k}"] = x_pre[k]
                        x[f"postsynaptic_{k}"] = x_post[k]
                for k, v in [("space", "distance"), ("ori", "rel_ori")]:
                    if k in self.variables:
                        x[v] = functional.diff(x_post[k], x_pre[k]).norm(dim=-1)
                if mask_var in x_pre:
                    x[mask_var] = x_post[mask_var] & x_pre[mask_var]
                x = frame.ParameterFrame(x, ndim=x_pre.ndim)

                if isinstance(W, CirculantTensor):
                    W = W.dense()
                x = x.datailoc[(None,) * self.batch_ndim] | {"W": W}
                logger.debug(f"x:\n{x}")

                if mask_var in x:
                    x = x.iloc[x[mask_var]]
                    del x[mask_var]
                logger.debug(f"x:\n{x}")
                x = x.to_framelike(**framelike_kwargs)

                return x

            if "cell_type" in self.variables:
                tau = self.tau[x.data["cell_type"]]  # (*, *shape)
            else:
                tau = 1.0

            if output == "jacobian":
                if not to_dataframe and not isinstance(W, CirculantTensor):
                    if isinstance(tau, Tensor):
                        tau = tau[..., :, None]  # (*, N, 1)
                    return (W - linalg.eye_like(W)) / tau
                raise NotImplementedError()

            if ndim > 1 and not isinstance(W, CirculantTensor):
                raise NotImplementedError(
                    "Currently only supports 1D inputs for non-circulant tensors."
                )

            dh = x.data[in_var]  # (*, *shape)
            # dtype doesn't matter, torchdiffeq always converts to double
            # squeeze to turn length-1 argument to 0-dim tensor, see numerics.integrate
            t = torch.tensor(t, device=dh.device).squeeze() if len(t) > 0 else None
            if isinstance(t, Tensor) and t.ndim == 1:
                # indicate that we start the stimulation at t = 0
                t = torch.cat([torch.tensor([0.0], device=dh.device), t])
            dr = numerics.perturbed_response(
                vf, W, f_, dh, t=t, tau=tau, **self.simulation_kwargs
            ).x  # (*bshape, *, *shape) or (len(t) + 1, *bshape, *, *shape)
            if isinstance(t, Tensor) and t.ndim == 1:
                # remove the initial time point that we just added
                t, dr = t[1:], dr[1:]
                if avg_t:
                    t, dr = t.mean(), dr.mean(dim=0)
            if return_dv:
                dv = linalg.bmv(W, dr)

        if assert_finite and not dr.isfinite().all():
            N_finite = dr.isfinite().count_nonzero()
            N = dr.numel()
            raise exceptions.SimulationError(
                f"{N - N_finite}/{N} non-finite values (NaN, +inf, or -inf) "
                "detected in V1 response."
            )

        n_expanded_dims = self.batch_ndim + (t.ndim if isinstance(t, Tensor) else 0)
        x = x.datailoc[(None,) * n_expanded_dims] | {out_var: dr}
        if return_dv:
            x[dv_var] = dv
        if isinstance(t, Tensor):
            x[time_var] = t[(...,) + (None,) * (x.ndim - t.ndim)]

        if mask_var in x:
            x = x.iloc[x[mask_var]]

        if to_dataframe:
            x = x.to_framelike(**framelike_kwargs)

        return x


def _allconst(x, dim, **kwargs):
    n = x.shape[dim]
    if n == 1:
        return True
    return torch.allclose(x.narrow(dim, 0, n - 1), x.narrow(dim, 1, n - 1), **kwargs)


def _cdim(x, ndim, **kwargs):
    cdim = []
    for dim in range(x.ndim - ndim, x.ndim):
        if all(
            _allconst(v, dim, **kwargs)
            or (
                (k in {"space", "ori"})
                and isinstance(v, PeriodicTensor)
                and _allconst(v.diff(dim=dim), dim, **kwargs)
                # note: this is currently not exhaustive since it does not check
                # that the circular difference between the first and last element is
                # the same as that between all other consecutive pairs of elements
            )
            for k, v in x._items()
        ):
            cdim.append(dim)
    return utils.normalize_dim(cdim, x.ndim, neg=True)
