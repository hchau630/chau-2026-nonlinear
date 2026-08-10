import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from niarb import exceptions, integrate, linalg, nn
from niarb.tensors.circulant import CirculantTensor
from niarb.utils import take_along_dims

logger = logging.getLogger(__name__)


def simulate(
    W: Tensor,
    f: Callable[[Tensor], Tensor],
    h: Tensor,
    t: Tensor | None = None,
    tau: Tensor | float = 1.0,
    x0: float | Tensor = 0.0,
    kind: str = "rate",
    **kwargs,
) -> integrate.OdeResult:
    x0 = torch.as_tensor(x0, dtype=h.dtype, device=h.device)
    x0, h = torch.broadcast_tensors(x0, h)  # (*, N)

    if kind == "rate":

        def func(_, x, W=W, f=f, h=h, tau=tau):
            return (f(linalg.bmv(W, x) + h) - x) / tau

    elif kind == "voltage":

        def func(_, x, W=W, f=f, h=h, tau=tau):
            return (linalg.bmv(W, f(x)) + h - x) / tau

    else:
        raise ValueError(f"kind must be 'rate' or 'voltage', but got {kind}.")

    logger.debug(f"{x0.shape=}, {W.shape=}, {h.shape=}")

    if t is None:
        out = integrate.odeint_ss(func, x0, **kwargs)
    else:
        out = integrate.odeint(func, x0, t, **kwargs)

    return out


def fixed_point(
    vf: Tensor, W: Tensor, f: Callable[[Tensor], Tensor]
) -> tuple[Tensor, Tensor]:
    """Compute fixed point of the dynamical system.

    Args:
        vf: Baseline voltage tensor with shape (*, N)
        W: Connection weight tensor with shape (*, N, N).
        f: Nonlinearity.

    Returns:
        A tuple of tensors (rf, hf), where rf is the fixed point firing rate and
        hf is the fixed point input. Both have shape (*, N).

    """
    rf = f(vf)  # (*, N)
    hf = vf - linalg.bmv(W, rf)  # (*, N)

    return rf, hf  # (*, N), (*, N)


def perturbed_response(
    vf: Tensor,
    W: Tensor,
    f: Callable[[Tensor], Tensor],
    dh: Tensor,
    dx0: float | Tensor = 0.0,
    kind: str = "rate",
    **kwargs,
) -> integrate.OdeResult:
    """Compute perturbed response by simulating the dynamical system.

    Args:
        vf: Baseline voltage tensor with shape (), (*, 1), or (*, N).
        W: Connection weight tensor with shape (*, N, N).
        f: Nonlinearity.
        dh: Perturbation tensor with shape (*, N).
        dx0 (optional): Initial condition of the perturbation response. If a tensor,
          must have shape (), (*, 1), or (*, N).
        kind (optional): 'rate' or 'voltage'.
        **kwargs: Optional arguments passed to integrate.odeint_ss.

    Returns:
        integrate.OdeResult

    Raises:
        ValueError: If kind is not 'rate' or 'voltage'.

    """
    if kind not in {"rate", "voltage"}:
        raise ValueError(f"kind must be 'rate' or 'voltage', but got {kind}.")

    vf, dh = torch.broadcast_tensors(vf, dh)  # (*, N), (*, N)
    rf, hf = fixed_point(vf, W, f)  # (*, N), (*, N)
    xf = rf if kind == "rate" else vf

    out = simulate(W, f, hf + dh, x0=xf + dx0, kind=kind, **kwargs)

    return integrate.OdeResult(x=out.x - xf, t=out.t, dxdt=out.dxdt)


# As of torch 2.3.1, torch.autograd.functional.jacobian outputs all zeros in
# inference_mode. Not sure if this is a bug or intended behavior, but we need
# to set inference_mode(False) to get the correct Jacobian. See github issue #128264.
@torch.inference_mode(False)
def compute_gain(f, vf, create_graph=True, **kwargs):
    if vf.ndim > 1:
        raise ValueError(f"vf must be 0 or 1-dimensional, but {vf=}.")

    vf = vf.clone()  # needed if vf is the output of code that is run in inference mode

    jac = torch.autograd.functional.jacobian(
        f, vf, create_graph=create_graph, **kwargs
    )  # defaults to create_graph=True because we need to backprop through it

    if jac.ndim not in {0, 2} or (jac.ndim == 2 and jac.shape[0] != jac.shape[1]):
        raise ValueError(
            f"output of f must have same shape as its input, but {jac.shape=}"
        )

    if jac.ndim == 2 and not linalg.is_diagonal(jac):
        raise ValueError(
            "f must be an element-wise function, but its Jacobian is not diagonal."
        )

    return jac.diagonal() if jac.ndim == 2 else jac  # (N,) or ()


# Compared to compute_gain, compute_nth_deriv is much faster for large inputs.
# However compute_nth_deriv requires a bit of warmup time.
@torch.inference_mode(False)  # see comment in compute_gain
def compute_nth_deriv(
    f: Callable[[Tensor, *tuple[Tensor, ...]], Tensor],
    vf: Tensor,
    args: tuple[Tensor, ...] = (),
    kwargs: dict[str, Any] | None = None,
    n: int = 1,
) -> Tensor:
    """Compute the nth derivative of a scalar function.

    Args:
        f: A scalar function.
        vf: Input tensors at which the derivative is evaluated.
        args (optional): Additional tensors to pass to `f`. Each tensor must be
          broadcastable with `vf`. These tensors are vmapped over.
        kwargs (optional): Optional arguments passed to `f`. Note that these arguments
          are NOT vmapped over, unlike `args`.
        n (optional): Order of the derivative.

    Returns:
        Derivative tensor with shape broadcast(vf.shape, *[arg.shape for arg in args]).

    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, but {n=}.")

    if kwargs is None:
        kwargs = {}

    shape = torch.broadcast_shapes(vf.shape, *[arg.shape for arg in args])
    vf = vf.broadcast_to(shape)
    args = [arg.broadcast_to(shape) for arg in args]

    if hasattr(f, "nth_deriv"):
        return f.nth_deriv(n, vf, *args, **kwargs)

    for _ in range(n):
        f = torch.func.grad(f)

    for _ in range(vf.ndim):
        f = torch.func.vmap(f)

    return f(vf, *args, **kwargs)


def perturbed_steady_state_approx(
    vf: float | Tensor,
    tL: Tensor,
    f: Callable[[Tensor], Tensor],
    dh: Tensor,
    f_args: Sequence[Tensor] = (),
    indices: Tensor | None = None,
    mode: str = "analytical",
    sparse_nonlinear: bool = True,
    max_num_steps: int = 20,
    check_finite: bool = True,
    assert_convergence: bool = True,
    assert_bounds: bool = True,
    max_dr_frac: float = 1e3,
    min_dv_frac: float | None = -1e3,
    max_final_delta_dv_norm: float = torch.inf,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    return_dv: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Compute approximate perturbed steady state with a recursive algorithm.

    Args:
        vf: Baseline voltage. If Tensor, must have shape () or (*, *shape).
        tL: In terms of the weights W and gain G matrices, tL := (I - GW)^{-1} - I,
          with shape (*, *shape, *shape').
        f: Nonlinearity.
        dh: Perturbation tensor with shape (**, *shape).
        f_args (optional): Additional Tensor arguments to pass to f.
        indices (optional): Tensor with shape (**, *shape') that maps the output dimensions
          to the input dimensions. This is required if and only if shape' != shape.
        mode (optional): One of {'analytical', 'linear_approx', 'quasi_linear_approx',
          'second_order_approx', 'second_order_approx_2-1', 'second_order_approx_2-2',
          'second_order_approx_naive', 'second_order_approx_naive_2-1',
          'second_order_approx_naive_2-2'}.
          If not 'analytical', the following arguments are ignored.
        sparse_nonlinear (optional): If True, use an optimization for sparse nonlinearities.
          If nonlinearity is dense, this option is less efficient especially when (*) and (**)
          have different shapes.
        max_num_steps (optional): Maximum number of steps for the recursive algorithm.
        check_finite (optional): If True, return as soon as any element of dr is non-finite.
        assert_convergence (optional): Whether to raise an error if the algorithm does not converge.
        max_dr_frac (optional): Maximum change in firing rate as a fraction of f(vf) elementwise.
          This is important for preventing numerical issues (which may show up as an
          linalg_eig_backward error because pytorch is dumb).
        min_dv_frac (optional): Minimum change in voltage as a fraction of vf elementwise.
          Also important for preventing numerical issues. If None, no minimum is imposed.
        max_final_delta_dv_norm (optional): Maximum norm of the vector dv_n - dv_{n-1}
          for the last iteration n.
        assert_bounds (optional): If either max_dr_frac or min_dv_frac is violated,
          raise SimulationError. Otherwise, simply return the final bounded values.
        rtol, atol (optional): tolerances for convergence criterion.
        return_dv (optional): If True, also return dv.

    Returns:
        Perturbation response tensor with shape (***, *shape), where *** = broadcast(*, **).

    Raises:
        ValueError: If max_num_steps is not a positive integer.
        SimulationError: If assert_convergence is True and the algorithm fails to converge due
          to one of the following reasons: the perturbation response vector has NaN values,
          the norm of the perturbation voltage vector exceeds max_dv_norm, or the convergence
          criterion is not satisfied within max_num_steps.

    """
    if not isinstance(max_num_steps, int) or max_num_steps < 0:
        raise ValueError(
            f"max_num_steps must be a non-negative integer, but {max_num_steps=}."
        )

    if mode not in {
        "analytical",
        "linear_approx",
        "quasi_linear_approx",
        "second_order_approx",
        "second_order_approx_2-1",
        "second_order_approx_2-2",
        "second_order_approx_naive",
        "second_order_approx_naive_2-1",
        "second_order_approx_naive_2-2",
    }:
        raise ValueError(f"Invalid mode: {mode}.")

    if (
        not isinstance(tL, CirculantTensor)
        and tL.shape[-2] != tL.shape[-1]
        and indices is None
    ):
        raise ValueError(
            "indices must be provided if tL is not a circulant tensor and not square."
        )

    if isinstance(tL, CirculantTensor) and indices is not None:
        raise ValueError("indices must be None if tL is a circulant tensor.")

    if mode == "linear_approx":
        dh = compute_nth_deriv(f, vf, f_args) * dh
    elif mode == "quasi_linear_approx":
        dh = f(vf + dh, *f_args) - f(vf, *f_args)

    if isinstance(f, nn.Identity) or mode in {"linear_approx", "quasi_linear_approx"}:
        # fast path for identity function
        dv = dh if indices is None else dh.take_along_dim(indices, dim=-1)
        dv = linalg.bmv(tL, dv)
        dr = dh + dv
        if return_dv:
            G = compute_nth_deriv(f, vf, f_args)  # () or (*, *shape)
            dv = dv / G
            return dr, dv
        return dr

    if isinstance(vf, float):
        vf = torch.tensor(vf, dtype=dh.dtype, device=dh.device)  # ()

    is_circulant = isinstance(tL, CirculantTensor)
    ndim = tL.vec_ndim if is_circulant else 1
    bshape = torch.broadcast_shapes(
        tL.batchshape if is_circulant else tL.shape[: -2 * ndim], dh.shape[:-ndim]
    )  # (***)
    N = math.prod(tL.vec_shape) if is_circulant else dh.shape[-1]
    if indices is not None:
        indices = indices[(None,) * (len(bshape) + ndim - indices.ndim) + (...,)]

    G = compute_nth_deriv(f, vf, f_args)  # () or (*, *shape)
    Gp = compute_nth_deriv(f, vf + dh, f_args)  # (**/***, *shape)
    Gp = Gp.broadcast_to(*bshape, *Gp.shape[-ndim:])  # (***, *shape)
    dG = Gp - G  # (***, *shape)
    dG = dG.reshape(*bshape, -1)  # (***, N)
    if mode.startswith("second_order_approx"):
        H = 0.5 * compute_nth_deriv(f, vf + dh, f_args, n=2)  # (**/***, *shape)

    if mode.startswith("second_order_approx_naive"):
        # Redefine f such that the second order approximation will become equiavlent
        # to the naive second order approximation.
        # H = 0.5 * compute_nth_deriv(f, vf + dh, f_args, n=2)  # (**/***, *shape)
        def f(x, *_):
            return H * (x - vf) ** 2 + G * (x - vf)

    if sparse_nonlinear:
        M = dG.count_nonzero(dim=-1).max().item()
        dGp, idx = dG.topk(M, dim=-1, sorted=False)  # (***, M), (***, M)
        if G.ndim == 0:
            F = G.unsqueeze(0)
        else:
            F = take_along_dims(
                G.reshape(*G.shape[:-ndim], -1), idx, dims=-1, keep_dims=True
            )
        F = (dGp / F).diag_embed()  # (***, M, M)
        eye = torch.eye(M, dtype=F.dtype, device=F.device)
        K = torch.zeros(*bshape, N, M, dtype=dG.dtype, device=dG.device)  # (***, N, M)
        K.scatter_(-2, idx[..., None, :], 1.0)  # (***, N, M)

        if is_circulant:
            if mode == "analytical":
                Kp = K.reshape(*bshape, *tL.vec_shape, M)  # (***, *shape, M)
                tLpp = tL @ Kp  # (***, *shape, M)
                tLpp = tLpp.reshape(*bshape, -1, M)  # (***, N, M)
                tLpp = K.t() @ tLpp  # (***, M, M)
                tLppK = torch.linalg.inv(eye - F @ tLpp) @ F @ K.t()  # (***, M, N)
            else:  # mode.startswith("second_order_approx")
                tLppK = F @ K.t()  # (***, M, N)
        else:
            tLp = torch.einsum("...ji,...jk->...ik", K, tL)  # (***, M, N')
            if mode == "analytical":
                # QK shape: (***, N', M)
                QK = (
                    K
                    if indices is None
                    else K.take_along_dim(indices[..., None], dim=-2)
                )
                tLpp = torch.einsum("...ij,...jk->...ik", tLp, QK)  # (***, M, M)
                tLpp = torch.linalg.inv(eye - F @ tLpp) @ F  # (***, M, M)
            else:  # mode.startswith("second_order_approx")
                tLpp = F  # (***, M, M)
            tLp = torch.einsum("...ij,...jk->...ik", tLpp, tLp)  # (***, M, N')
    else:
        pass

    rf = f(vf, *f_args)  # () or (*, *shape)
    dv = torch.zeros_like(dh)  # (**, *shape)
    dr = f(vf + dh + dv, *f_args) - rf  # (**/***, *shape)
    prev_dr = dr

    logger.debug(f"{dv.shape=}, {dr.shape=}, {G.shape=}, {Gp.shape=}")

    if mode.startswith("second_order_approx"):
        dr = dr[(None,) * (len(bshape) + ndim - dr.ndim) + (...,)]  # (***, *shape)
        dh = dr if indices is None else dr.take_along_dim(indices, dim=-1)
        if is_circulant:
            dr1 = linalg.bmv(
                K, linalg.bmv(tLppK, linalg.bmv(tL, dh).reshape(*bshape, -1))
            ).reshape(*bshape, *tL.vec_shape)  # (***, *shape)
        else:
            dr1 = linalg.bmv(K, linalg.bmv(tLp, dh))  # (***, N)
        if mode.endswith("2-1"):
            dr = dr1
        elif mode.endswith("2-2"):
            dr = H * (linalg.bmv(tL, dh) / G) ** 2
            # Ldh = linalg.bmv(tL, dh)
            # dr = f(vf + Ldh / G, *f_args) - rf - Ldh
        else:
            dr = dr + dr1 + H * (linalg.bmv(tL, dh) / G) ** 2  # (***, N)
            # Ldh = linalg.bmv(tL, dh)
            # dr = dr + dr1 + f(vf + Ldh / G, *f_args) - rf - Ldh
        dh = dr if indices is None else dr.take_along_dim(indices, dim=-1)
        dh = linalg.bmv(tL, dh)
        dr = dr + dh  # (***, *shape)
        if return_dv:
            return dr, dh / G
        return dr

    max_dr = max_dr_frac * rf  # () or (*, *shape)
    min_dv = min_dv_frac * vf.abs() if min_dv_frac else vf - torch.inf
    converged, n = False, 0
    while (
        n < max_num_steps
        and not converged
        and (not check_finite or dr.isfinite().all())
        and (dr <= max_dr).all()
        and (dv >= min_dv).all()
    ):
        prev_dv, prev_dr = dv, dr  # (**/***, *shape)
        dv = dr - Gp * dv  # (***, *shape)
        if is_circulant:
            dv = dv + linalg.bmv(
                K, linalg.bmv(tLppK, linalg.bmv(tL, dv).reshape(*bshape, -1))
            ).reshape(*bshape, *tL.vec_shape)  # (***, *shape)
        else:
            if indices is not None:
                dv = dv.take_along_dim(indices, dim=-1)  # (***, N')
            dv = dv + linalg.bmv(QK, linalg.bmv(tLp, dv))  # (***, N')
        dv = linalg.bmv(tL, dv) / G  # (***, N)
        dr = f(vf + dh + dv, *f_args) - rf  # (***, N)

        if logger.isEnabledFor(logging.DEBUG):
            # prevent unnecessary GPU-CPU sync which may impact performance
            # by only executing the debug statement if DEBUG level is enabled
            logger.debug(
                f"n = {n}: {(dr / rf).max().item()}, {(dv / vf).min().item()}, "
                f"{(dv - prev_dv).norm().item()}, {dr.norm().item()}"
            )
        converged = torch.allclose(dv, prev_dv, rtol=rtol, atol=atol)

        n += 1

    if assert_convergence and not converged:
        raise exceptions.SimulationError(
            f"Failed to converge with {rtol=}, {atol=} within {max_num_steps=} steps."
        )

    if assert_convergence and (dv - prev_dv).norm().item() > max_final_delta_dv_norm:
        raise exceptions.SimulationError(
            f"Failed to converge since {(dv - prev_dv).norm().item()=} is greater than "
            f"{max_final_delta_dv_norm=}."
        )

    if assert_bounds and (dr > max_dr).any():
        raise exceptions.SimulationError(f"{(dr / rf).max()=} exceeded {max_dr_frac=}.")

    if assert_bounds and (dv < min_dv).any():
        raise exceptions.SimulationError(f"{(dv / vf).min()=} exceeded {min_dv_frac=}.")

    if (dr > max_dr).any() or (dv < min_dv).any():
        return (prev_dr, prev_dv) if return_dv else prev_dr

    return (dr, dv) if return_dv else dr


class LowRankMatrix:
    """Matrix of the form αI + UV."""

    def __init__(
        self,
        n: int,
        k: int,
        alpha: float = 1.0,
        batch_shape: Sequence[int] = (),
        **kwargs,
    ):
        self.alpha = alpha
        self.U = torch.zeros((*batch_shape, n, k), **kwargs)
        self.V = torch.zeros((*batch_shape, k, n), **kwargs)

    def update(self, u: Tensor, v: Tensor, idx: int) -> Tensor:
        self.U[..., :, idx] = u
        self.V[..., idx, :] = v
        return self

    def bmv(self, x: Tensor) -> Tensor:
        return self.alpha * x + linalg.bmv(self.U, linalg.bmv(self.V, x))

    def bvm(self, x: Tensor) -> Tensor:
        return self.alpha * x + linalg.bvm(linalg.bvm(x, self.U), self.V)


def solve_perturbed_steady_state_forward(
    vf: float | Tensor,
    W: Tensor,
    dh: Tensor,
    f: Callable[[Tensor], Tensor],
    f_args: Sequence[Tensor],
    method: str = "newton",
    step_method: str = "line_search",
    init_J_exact: bool = False,
    low_rank_rep: bool | None = None,
    max_num_steps: int | None = None,
    check_finite: bool = True,
    assert_convergence: bool = True,
    assert_bounds: bool = True,
    max_dr_frac: float = 1e3,
    min_dv_frac: float | None = -1e3,
    max_final_delta_dv_norm: float = torch.inf,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    F_norm_max: float = 1.0,
    eps: float = 1e-12,
    alpha: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    **kwargs,
) -> Tensor:
    if method not in {"newton", "broyden", "gd"}:
        raise ValueError(
            f"`method` must be'newton', 'broyden', or 'gd, but got {method=}."
        )

    if step_method not in {"line_search", "adam", "fixed"}:
        raise ValueError(
            "`step_method` must be 'line_search', 'adam', or 'fixed', but got "
            f"{step_method=}."
        )

    if max_num_steps is None:
        # Broyden/GD is much faster per step but also requires many more steps
        max_num_steps = {"newton": 20, "broyden": 1000, "gd": 5000}[method]

    if low_rank_rep is None:
        low_rank_rep = max_num_steps < W.shape[-1]

    I = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)  # (N, N)
    rf = f(vf, *f_args)  # (*, N)

    def _func(x, dv=None):
        if dv is None:
            dv = linalg.bmv(W, x) + dh
        return rf + x - f(vf + dv, *f_args)

    dr, dv = torch.zeros_like(dh), dh  # (**, N)
    Fx = _func(dr, dv=dv)  # (***, N)
    prev_dr, prev_dv, prev_Fx, J = None, None, None, None  # make LSP happy
    max_dr = max_dr_frac * rf  # (*, N)
    min_dv = min_dv_frac * vf.abs() if min_dv_frac else vf - torch.inf
    converged, n = False, 0
    _m, _v = 0, 0  # Adam moments

    while (
        n < max_num_steps
        and not converged
        and (not check_finite or dr.isfinite().all())
        and (dr <= max_dr).all()
        and (dv >= min_dv).all()
    ):
        # Update the Jacobian or the inverse Jacobian
        gain = compute_nth_deriv(f, vf + dv, f_args)  # (***, N)
        if method == "broyden" and n > 0:
            dx, dF = dr - prev_dr, Fx - prev_Fx
            JdF = linalg.bmv(J, dF) if isinstance(J, Tensor) else J.bmv(dF)  # (***, N)
            denom = torch.linalg.vecdot(dx, JdF, dim=-1)  # (***)
            denom = torch.copysign(torch.clip(denom.abs(), min=eps), denom)  # (***)
            u = (dx - JdF) / denom[..., None]  # (***, N)
            v = linalg.bvm(dx, J) if isinstance(J, Tensor) else J.bvm(dx)  # (***, N)
            # update inverse Jacobian
            if isinstance(J, Tensor):
                J = torch.einsum("...i,...j->...ij", u, v).add_(J)
            else:
                J.update(u, v, n - 1)
        elif method == "broyden" and not init_J_exact:
            # initialize inverse Jacobian
            if low_rank_rep:
                N, K = Fx.shape[-1], max_num_steps
                J = LowRankMatrix(
                    N, K, batch_shape=Fx.shape[:-1], dtype=Fx.dtype, device=Fx.device
                )
            else:
                J = I  # (N, N)
        elif method != "gd":  # "newton" or ("broyden", init_J_exact == True, n == 0)
            del J  # save memory, kind of hacky but works
            J = (gain[..., None] * W).sub_(I).neg_()  # (***, N, N)
            if method == "broyden":
                torch.linalg.inv(J, out=J)  # inverse Jacobian

        # Compute update step direction dx
        if method == "broyden":
            dx = -linalg.bmv(J, Fx) if isinstance(J, Tensor) else -J.bmv(Fx)  # (***, N)
        elif method == "gd":
            # gradient descent on ||F(x)||^2, so dx = -2F^T J = -2(F^T - F^T G W)
            dx = 2 * (linalg.bvm(Fx * gain, W) - Fx)  # (***, N)
        else:  # method == "newton"
            dx = -torch.linalg.solve(J, Fx)  # (***, N)

        # Update dr, dv, Fx with or without line search
        prev_dv, prev_dr, prev_Fx = dv, dr, Fx
        if step_method == "line_search":
            # here phi0 = ||F||^2, derphi0 = 2F^T J dx
            alpha0, phi0 = alpha, torch.linalg.vector_norm(Fx, dim=-1) ** 2
            if method == "broyden":
                Jdx = dx - gain * linalg.bmv(W, dx)
                derphi0 = 2.0 * torch.linalg.vecdot(Fx, Jdx, dim=-1)
            elif method == "gd":
                # dx = -2J^T F, so derphi0 = 2F^T Jdx = -4F^T JJ^T F = -||dx||^2
                derphi0 = -(torch.linalg.vector_norm(dx, dim=-1) ** 2)
                # heuristic: if slope is too steep, reduce step size
                alpha0 = torch.clip(alpha / derphi0.abs(), max=alpha)
            else:  # method == "newton"
                # dx = -J^{-1}F, so derphi0 = 2F^T Jdx = -2F^T J J^{-1} F = -2||F||^2
                derphi0 = -2.0 * phi0
            s, dr = _nonlin_line_search(
                _func, dr, dx, phi0, derphi0, alpha0=alpha0, **kwargs
            )
        elif step_method == "adam":
            # see the Adam paper (Kingma and Ba, 2014)
            _m = betas[0] * _m + (1 - betas[0]) * -dx
            _v = betas[1] * _v + (1 - betas[1]) * dx**2
            num = math.sqrt(1 - betas[1] ** (n + 1))
            alpha_t = alpha * num / (1 - betas[0] ** (n + 1))
            s, dr = alpha_t, dr - alpha_t * _m / (_v.sqrt() + num * adam_eps)
        else:
            s, dr = alpha, dr + alpha * dx
        dv = linalg.bmv(W, dr) + dh
        Fx = _func(dr, dv=dv)  # (***, N)
        Fn_max = Fx.norm(dim=-1).max().item()

        if logger.isEnabledFor(logging.DEBUG):
            # prevent unnecessary GPU-CPU sync which may impact performance
            # by only executing the debug statement if DEBUG level is enabled
            s_mean = s if isinstance(s, float) else s.mean().item()
            dr_frac_max = (dr / rf).max().item()
            dv_frac_min = (dv / vf).min().item()
            delta_dv_norm = (dv - prev_dv).norm(dim=-1).mean().item()
            F_norm = Fx.norm(dim=-1).mean().item()
            logger.debug(
                f"n = {n}: {s_mean=:.5g}, {dr_frac_max=:.5g}, {dv_frac_min=:.5g}, "
                f"{delta_dv_norm=:.5g}, {F_norm=:.5g}, F_norm_max={Fn_max:.5g}"
            )
        converged = (
            torch.allclose(dv, prev_dv, rtol=rtol, atol=atol) and Fn_max < F_norm_max
        )

        n += 1

    if assert_convergence and not converged:
        raise exceptions.SimulationError(
            f"Failed to converge with {rtol=}, {atol=}, {F_norm_max=} within "
            f"{max_num_steps=} steps."
        )

    if assert_bounds and (dv - prev_dv).norm().item() > max_final_delta_dv_norm:
        raise exceptions.SimulationError(
            f"Failed to converge since {(dv - prev_dv).norm().item()=} is greater than "
            f"{max_final_delta_dv_norm=}."
        )

    if assert_bounds and (dr > max_dr).any():
        raise exceptions.SimulationError(f"{(dr / rf).max()=} exceeded {max_dr_frac=}.")

    if assert_bounds and (dv < min_dv).any():
        raise exceptions.SimulationError(f"{(dv / vf).min()=} exceeded {min_dv_frac=}.")

    if (dr > max_dr).any() or (dv < min_dv).any():
        return prev_dr

    return dr


def _safe_norm(v):
    out = torch.linalg.vector_norm(v, dim=-1)
    nonfinite = ~torch.isfinite(v)
    out[nonfinite.any(dim=-1)] = torch.inf
    return out


# code adapted from scipy/optimize/_nonlin.py
def _nonlin_line_search(func, x, dx, phi0, derphi0, search_type="armijo", **kwargs):
    def phi(s):
        xt = x + s[..., None] * dx
        v = func(xt)
        p = _safe_norm(v) ** 2
        return p

    if search_type == "wolfe":
        raise NotImplementedError()
    elif search_type == "armijo":
        s = scalar_search_armijo(phi, phi0, derphi0, **kwargs)

    x = x + s[..., None] * dx
    return s, x


# code adapted from scipy/optimize/_linesearch.py
def scalar_search_armijo(
    phi,
    phi0,
    derphi0,
    c1=1e-2,  # 1e-4 can result in a different solution than simulation
    alpha0=1,
    amin=1e-5,
    afail=1e-5,
    tmin=0.1,
    tmax=0.9,
    tau=0.5,
):
    """Minimize over alpha, the function ``phi(alpha)``.

    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    alpha > 0 is assumed to be a descent direction.

    """
    alpha0 = torch.full_like(phi0, alpha0) if isinstance(alpha0, float) else alpha0
    alpha = torch.full_like(phi0, afail)
    satisfied = derphi0 >= 0  # if not descent direction, return afail

    if satisfied.any():
        logger.debug(
            f"{satisfied.count_nonzero()}/{alpha.numel()} non-descent directions."
        )

    if satisfied.all():
        return alpha

    phi_a0 = phi(alpha0)
    satisfied0 = (phi_a0 <= phi0 + c1 * alpha0 * derphi0) & (alpha0 > amin)
    alpha = torch.where(~satisfied & satisfied0, alpha0, alpha)
    satisfied = satisfied | satisfied0
    if satisfied.all():
        return alpha

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    mask = alpha1.isnan() | (alpha1 < tmin * alpha0) | (alpha1 > tmax * alpha0)
    alpha1 = torch.where(mask, tau * alpha0, alpha1)
    phi_a1 = phi(alpha1)

    satisfied1 = (phi_a1 <= phi0 + c1 * alpha1 * derphi0) & (alpha1 > amin)
    alpha = torch.where(~satisfied & satisfied1, alpha1, alpha)
    satisfied = satisfied | satisfied1
    if satisfied.all():
        return alpha

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while (alpha1 > amin).any():  # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1 - alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1**2 * (
            phi_a0 - phi0 - derphi0 * alpha0
        )
        a = a / factor
        b = -(alpha0**3) * (phi_a1 - phi0 - derphi0 * alpha1) + alpha1**3 * (
            phi_a0 - phi0 - derphi0 * alpha0
        )
        b = b / factor

        alpha2 = (-b + torch.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0 * a)
        mask = alpha2.isnan() | (alpha2 < tmin * alpha1) | (alpha2 > tmax * alpha1)
        alpha2 = torch.where(mask, tau * alpha1, alpha2)
        phi_a2 = phi(alpha2)

        satisfied2 = (phi_a2 <= phi0 + c1 * alpha2 * derphi0) & (alpha2 > amin)
        alpha = torch.where(~satisfied & satisfied2, alpha2, alpha)
        satisfied = satisfied | satisfied2
        # logger.debug(
        #     f"{satisfied.squeeze()}, alpha={alpha.squeeze()}, alpha2={alpha2.squeeze()}"
        # )
        if satisfied.all():
            return alpha

        # mask = ((alpha1 - alpha2) > alpha1 / 2.0) | ((1 - alpha2 / alpha1) < 0.96)
        # alpha2 = torch.where(mask, tau * alpha1, alpha2)

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    logger.debug(f"{(~satisfied).count_nonzero()}/{alpha.numel()} line search failed.")
    return alpha


class SolvePerturbedSteadyState(torch.autograd.Function):
    @staticmethod
    def forward(vf, W, dh, f, f_args, kwargs):
        return solve_perturbed_steady_state_forward(vf, W, dh, f, f_args, **kwargs)

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs, output):
        if any(ctx.needs_input_grad[:3]):
            ctx.save_for_backward(*inputs[:3], output, *inputs[4])
            ctx.f = inputs[3]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor | None):
        if grad_output is None:
            return None, None, None, None, None, None

        vf, W, dh, dr, *f_args = ctx.saved_tensors
        f = ctx.f
        grad_vf, grad_W, grad_dh = None, None, None

        dv = linalg.bmv(W, dr) + dh  # (***, N)
        I = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)  # (N, N)
        G = compute_nth_deriv(f, vf + dv, f_args)  # (***, N)
        J = (G[..., None] * W).sub_(I).neg_()  # (***, N, N)
        grad_output = torch.linalg.solve(J.mT, grad_output)  # (***, N)

        if ctx.needs_input_grad[0]:
            G0 = compute_nth_deriv(f, vf, f_args)  # (***, N)
            grad_vf = grad_output * (G - G0)[..., None, :]
        if ctx.needs_input_grad[1]:
            grad_W = torch.einsum("...i,...j->...ij", grad_output * G[..., None, :], dr)
        if ctx.needs_input_grad[2]:
            grad_dh = grad_output * G[..., None, :]

        return grad_vf, grad_W, grad_dh, None, None, None


def solve_perturbed_steady_state(
    vf: float | Tensor,
    W: Tensor,
    f: Callable[[Tensor], Tensor],
    dh: Tensor,
    f_args: Sequence[Tensor] = (),
    **kwargs,
) -> Tensor:
    r"""Solve for perturbed steady state response using Newton or quasi-Newton methods.

    Args:
        vf: Baseline voltage. If Tensor, must have shape () or (*, N).
        W: Connectivity matrix with shape (*, N, N).
        f: Nonlinearity.
        dh: Perturbation tensor with shape (**, N).
        f_args (optional): Additional Tensor arguments to pass to f.
        method (optional): Currently only 'newton' is supported.
        line_search (optional): If True, uses Armijo line search to compute step size.
        init_J_exact (optional): If True, initialize the Jacobian exactly at the first
          step. This only applies when method is 'broyden'.
        low_rank_rep (optional): If True, use a low-rank representation of the inverse
          Jacobian. This only applies when method is 'broyden'. If None, use a low-rank
          representation if and only if the inverse Jacobian is truly low-rank.
        max_num_steps (optional): Maximum number of steps for the recursive algorithm.
        check_finite (optional): If True, return as soon as any element of dr is non-finite.
        assert_convergence (optional): Whether to raise an error if the algorithm does not converge.
        max_dr_frac (optional): Maximum change in firing rate as a fraction of f(vf) elementwise.
          This is important for preventing numerical issues (which may show up as an
          linalg_eig_backward error because pytorch is dumb).
        min_dv_frac (optional): Minimum change in voltage as a fraction of vf elementwise.
          Also important for preventing numerical issues. If None, no minimum is imposed.
        max_final_delta_dv_norm (optional): Maximum norm of the vector dv_n - dv_{n-1}
          for the last iteration n.
        assert_bounds (optional): If either max_dr_frac or min_dv_frac is violated,
          raise SimulationError. Otherwise, simply return the final bounded values.
        rtol, atol (optional): tolerances for convergence criterion.

    Returns:
        Perturbation response tensor with shape (***, N), where *** = broadcast(*, **).

    Raises:
        ValueError: If max_num_steps is not a positive integer.
        SimulationError: If assert_convergence is True and the algorithm fails to converge due
          to one of the following reasons: the perturbation response vector has NaN values,
          the norm of the perturbation voltage vector exceeds max_dv_norm, or the convergence
          criterion is not satisfied within max_num_steps.

    """
    vf = torch.as_tensor(vf, dtype=dh.dtype, device=dh.device)
    if vf.ndim == 0:
        vf = vf.broadcast_to(W.shape[:-1])  # (*, N)
    return SolvePerturbedSteadyState.apply(vf, W, dh, f, f_args, kwargs)
