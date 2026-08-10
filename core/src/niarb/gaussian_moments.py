"""Push a Gaussian N(mu, sigma^2) through a function f and back, by quadrature.

Forward (`gaussian_pushforward`): A = E[f(X)] and V = Var[f(X)] by k-point
Gauss-Hermite quadrature -- k batched evaluations of f, exact for polynomial f
up to degree 2k - 1.

Inverse (`gaussian_pullback`): given (A, V), recovers one (mu, sigma) branch by
damped Newton in (mu, log sigma) with backtracking line search. The Jacobian is
exact for the quadrature map and costs one extra pass over f' (a single n=1
call to compute_nth_deriv per node). Since the moment equations are generally
multi-valued (one branch per root of f(mu) = A, plus e.g. mirror solutions for
symmetric f), `mu_init` selects the branch. Round-trips with the forward
function are exact to Newton tolerance: the residual IS the forward map.

f must support batched elementwise evaluation: f(x, *args, **kwargs) with x of
any shape and args broadcastable against it.

Quadrature nodes are visited sequentially with running (Welford) accumulation,
so peak memory is O(batch) -- one node's worth of evaluations at a time --
rather than O(num_points * batch) for a stacked grid. Nodes and weights enter
as Python floats, so no dtype/device bookkeeping is needed for them.
"""

import math
from collections.abc import Callable
from typing import Any

import torch
from scipy.special import roots_hermite
from torch import Tensor
from torch.autograd.function import once_differentiable

from niarb.numerics import compute_nth_deriv

_SQRT2 = math.sqrt(2.0)
_GH_CACHE: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}


def _gauss_hermite(k: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes t and normalized weights w (sum w = 1) as Python floats, so that
    E[h(X)] ~= sum_i w_i h(mu + sqrt(2) sigma t_i) for X ~ N(mu, sigma^2).
    Exact for polynomial h up to degree 2k - 1.

    scipy's roots_hermite targets the bare integral of e^{-t^2} h(t), so its
    weights sum to sqrt(pi). We want an expectation under the normal density:
    substituting x = mu + sqrt(2) sigma t leaves a 1/sqrt(pi) prefactor, which
    we fold into the weights. Dividing by weights.sum() (= sqrt(pi)) does this
    while making sum w = 1 exact to rounding, so constants integrate exactly.
    Scalars combine with tensors of any dtype/device without promotion, hence
    the cache is keyed by k alone.
    """
    if k not in _GH_CACHE:
        nodes, weights = roots_hermite(k)
        weights = weights / weights.sum()
        _GH_CACHE[k] = (tuple(map(float, nodes)), tuple(map(float, weights)))
    return _GH_CACHE[k]


def _quadrature_pass(
    f: Callable[[Tensor, *tuple[Tensor, ...]], Tensor],
    mu: Tensor,
    sigma: Tensor,
    args: tuple[Tensor, ...],
    kwargs: dict[str, Any],
    num_points: int,
    jacobian: bool,
) -> tuple[Tensor, ...]:
    """Compute the first two moments, analytically where available.

    If ``f.gaussian_moments`` exists, it is asked for E[f(X)] and E[f(X)^2]
    and, when requested, their derivatives.  An output element uses the
    analytical result only if no tensor returned by ``gaussian_moments`` is
    NaN at that element; otherwise all outputs for that element are filled
    by a sequential Gauss-Hermite pass.

    The fallback accumulates (A, V) by the weighted Welford recurrence, which is
    cancellation-safe while requiring only O(batch) peak memory.  With
    ``jacobian=True``, the same pass also accumulates the exact Jacobian of the
    quadrature map with respect to ``mu`` and ``sigma``.

    Returns (A, V) or (A, V, A_mu, A_sig, V_mu, V_sig).
    """
    exact_outputs = None
    fallback = None
    if hasattr(f, "gaussian_moments"):
        exact = f.gaussian_moments([1, 2], mu, sigma, jacobian, *args, **kwargs)
        if jacobian:
            moments, moments_mu, moments_sigma = exact
            A, second = moments
            A_mu, second_mu = moments_mu
            A_sig, second_sig = moments_sigma
            exact_outputs = (
                A,
                second - A.square(),
                A_mu,
                A_sig,
                second_mu - 2.0 * A * A_mu,
                second_sig - 2.0 * A * A_sig,
            )
            exact_tensors = moments + moments_mu + moments_sigma
        else:
            A, second = exact
            exact_outputs = (A, second - A.square())
            exact_tensors = exact

        fallback = torch.zeros_like(mu, dtype=torch.bool)
        for value in exact_tensors:
            fallback |= torch.isnan(value)
        if not bool(fallback.any()):
            return exact_outputs

    nodes, weights = _gauss_hermite(num_points)
    wsum = 0.0
    mean: Tensor | None = None
    m2 = s_fp = s_fpt = s_ffp = s_ffpt = None
    for t, w in zip(nodes, weights):
        if w == 0.0:
            # For large num_points the extreme nodes' weights (~ e^{-t^2},
            # |t| ~ sqrt(2k)) underflow to exactly 0.0 in float64. They
            # contribute nothing to any sum, and visiting them anyway would
            # both waste an evaluation of f and divide by a still-zero wsum
            # in the Welford update if they come first (nodes are ordered, so
            # they do). The number of nonzero-weight nodes grows only like
            # O(sqrt(num_points)), so very large k stays cheap.
            continue
        x = mu + _SQRT2 * t * sigma
        fx = f(x, *args, **kwargs)
        wsum += w
        if mean is None:
            mean, m2 = fx, torch.zeros_like(fx)
        else:
            delta = fx - mean
            mean = mean + (w / wsum) * delta
            m2 = m2 + w * delta * (fx - mean)
        if jacobian:
            fpx = compute_nth_deriv(f, x, args, kwargs, n=1)
            if s_fp is None:
                s_fp, s_fpt = w * fpx, (w * t) * fpx
                s_ffp, s_ffpt = w * fx * fpx, (w * t) * fx * fpx
            else:
                s_fp = s_fp + w * fpx
                s_fpt = s_fpt + (w * t) * fpx
                s_ffp = s_ffp + w * fx * fpx
                s_ffpt = s_ffpt + (w * t) * fx * fpx

    A, V = mean, m2 / wsum
    if jacobian:
        A_mu, A_sig = s_fp, _SQRT2 * s_fpt
        quadrature_outputs = (
            A,
            V,
            A_mu,
            A_sig,
            2.0 * (s_ffp - A * s_fp),
            2.0 * _SQRT2 * (s_ffpt - A * s_fpt),
        )
    else:
        quadrature_outputs = (A, V)

    if exact_outputs is None:
        return quadrature_outputs
    return tuple(
        torch.where(fallback, quadrature, exact)
        for exact, quadrature in zip(exact_outputs, quadrature_outputs)
    )


def gaussian_pushforward(
    f: Callable[[Tensor, *tuple[Tensor, ...]], Tensor],
    mu: Tensor,
    sigma: Tensor,
    args: tuple[Tensor, ...] = (),
    kwargs: dict[str, Any] | None = None,
    num_points: int = 16,
) -> tuple[Tensor, Tensor]:
    """Compute A = E[f(X)] and V = Var[f(X)] for X ~ N(mu, sigma^2).

    Args:
        f: Elementwise function of x; must accept batched inputs.
        mu, sigma: Mean and std of X; broadcastable with each other and args.
        args: Extra tensors passed to f, broadcastable with mu.
        kwargs: Extra keyword arguments passed to f as-is.
        num_points: Number of quadrature nodes k. Exact for polynomial f of
            degree <= 2k - 1; increase for large sigma or sharp features in f.
            Convergence is easily checked by doubling k.

    Returns:
        (A, V), each with shape broadcast(mu, sigma, *args).

    Accumulation details (sequential Welford pass, O(batch) peak memory,
    cancellation-safe variance) are documented on _quadrature_pass.
    """
    if kwargs is None:
        kwargs = {}
    shape = torch.broadcast_shapes(mu.shape, sigma.shape, *[a.shape for a in args])
    mu, sigma = mu.broadcast_to(shape), sigma.broadcast_to(shape)
    args = tuple(a.broadcast_to(shape) for a in args)

    return _quadrature_pass(f, mu, sigma, args, kwargs, num_points, jacobian=False)


# Default (rtol, atol) per dtype, following torch.testing.assert_close
# (torch.testing._comparison._DTYPE_PRECISIONS).
_DTYPE_PRECISIONS: dict[torch.dtype, tuple[float, float]] = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex32: (1e-3, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}


def _resolve_tolerances(
    dtype: torch.dtype, rtol: float | None, atol: float | None
) -> tuple[float, float]:
    """Fill in default (rtol, atol) for `dtype` following the defaults of
    torch.testing.assert_close. As there, both must be given or neither."""
    if (rtol is None) ^ (atol is None):
        raise ValueError(
            "Both 'rtol' and 'atol' must be specified together, or neither "
            f"(got rtol={rtol}, atol={atol})."
        )
    if rtol is None:
        rtol, atol = _DTYPE_PRECISIONS.get(dtype, (0.0, 0.0))
    return rtol, atol


def _newton_1d_init(
    f: Callable[[Tensor, *tuple[Tensor, ...]], Tensor],
    A: Tensor,
    args: tuple[Tensor, ...],
    kwargs: dict[str, Any] | None,
    mu_init: Tensor | None,
    newton_iters: int,
    rtol: float,
    atol: float,
) -> Tensor:
    """Safeguarded 1D Newton for f(mu) = A (the sigma -> 0 initializer).

    Per-element backtracking on |f(mu) - A| prevents the plain Newton step from
    catapulting into flat tails (f' ~ 0) of non-monotone f such as bumps, where
    the raw step (f - A)/f' diverges. mu_init selects the branch of f^{-1}.
    """
    if kwargs is None:
        kwargs = {}
    tiny = torch.finfo(A.dtype).tiny
    if mu_init is None:
        mu = torch.zeros(A.shape, dtype=A.dtype, device=A.device)
    else:
        mu = mu_init.broadcast_to(A.shape).to(A.dtype).clone()
    f0 = f(mu, *args, **kwargs)
    for _ in range(newton_iters):
        if torch.allclose(f0, A, rtol=rtol, atol=atol):
            break
        res = (f0 - A).abs()
        f1 = compute_nth_deriv(f, mu, args, kwargs, n=1)
        f1 = torch.where(f1.abs() < tiny, torch.full_like(f1, tiny), f1)
        step = (f0 - A) / f1
        alpha = torch.ones_like(res)
        for _ in range(12):
            res_t = (f(mu - alpha * step, *args, **kwargs) - A).abs()
            ok = res_t <= res * (1.0 - 1e-4 * alpha)
            if bool(ok.all()):
                break
            alpha = torch.where(ok, alpha, alpha / 2)
        mu = mu - alpha * step
        f0 = f(mu, *args, **kwargs)
    return mu


def _solve_pullback(
    f: Callable[[Tensor, *tuple[Tensor, ...]], Tensor],
    A: Tensor,
    V: Tensor,
    args: tuple[Tensor, ...],
    kwargs: dict[str, Any],
    num_points: int,
    mu_init: Tensor | None,
    newton_iters: int,
    rtol: float | None,
    atol: float | None,
    check_convergence: bool,
) -> tuple[Tensor, Tensor]:
    """Damped Newton in (mu, v = log sigma) with backtracking. See
    gaussian_pullback for the semantics; this is the raw (non-differentiable)
    solver it wraps.

    Convergence is judged on the final residuals only: the 1D initializer is a
    heuristic (it solves the sigma -> 0 limit, whose root need not be close to
    the true mu), so a stall there is not itself a failure -- the 2D solve may
    still recover. Conversely, satisfying torch.allclose(., ., rtol, atol) at
    the end is the contract."""
    shape = torch.broadcast_shapes(A.shape, V.shape, *[a.shape for a in args])
    A, V = A.broadcast_to(shape), V.broadcast_to(shape)
    args = tuple(a.broadcast_to(shape) for a in args)
    tiny = torch.finfo(A.dtype).tiny
    rtol, atol = _resolve_tolerances(A.dtype, rtol, atol)

    # Initialization: sigma -> 0 limit. Solve f(mu) = A by safeguarded 1D
    # Newton, then the delta method gives sigma_0 = sqrt(V) / |f'(mu_0)|.
    mu = _newton_1d_init(f, A, args, kwargs, mu_init, newton_iters, rtol, atol)
    f1 = compute_nth_deriv(f, mu, args, kwargs, n=1)
    v = 0.5 * (V / f1.square().clamp_min(tiny)).clamp_min(tiny).log()  # v = log sigma

    converged = False
    for _ in range(newton_iters):
        sigma = v.exp()
        A_hat, V_hat, A_mu, A_sig, V_mu, V_sig = _quadrature_pass(
            f, mu, sigma, args, kwargs, num_points, jacobian=True
        )
        rA, rV = A_hat - A, V_hat - V
        if torch.allclose(A_hat, A, rtol=rtol, atol=atol) and torch.allclose(
            V_hat, V, rtol=rtol, atol=atol
        ):
            converged = True
            break

        A_v, V_v = sigma * A_sig, sigma * V_sig  # chain rule to v = log sigma
        det = A_mu * V_v - A_v * V_mu
        det = torch.where(det.abs() < tiny, torch.full_like(det, tiny), det)
        d_mu = (V_v * rA - A_v * rV) / det
        d_v = (A_mu * rV - V_mu * rA) / det

        # Per-element backtracking line search on the residual max-norm.
        rnorm = torch.maximum(rA.abs(), rV.abs())
        alpha = torch.ones_like(rnorm)
        for _ in range(12):
            A_t, V_t = gaussian_pushforward(
                f, mu - alpha * d_mu, (v - alpha * d_v).exp(), args, kwargs, num_points
            )
            ok = torch.maximum((A_t - A).abs(), (V_t - V).abs()) <= rnorm * (
                1.0 - 1e-4 * alpha
            )
            if bool(ok.all()):
                break
            alpha = torch.where(ok, alpha, alpha / 2)
        mu = mu - alpha * d_mu
        v = v - alpha * d_v

    if check_convergence and not converged:
        # The loop's last action was an update, so re-evaluate the residuals
        # at the final iterate (one derivative-free quadrature pass).
        A_hat, V_hat = gaussian_pushforward(f, mu, v.exp(), args, kwargs, num_points)
        resid = torch.maximum((A_hat - A).abs(), (V_hat - V).abs())
        # Elementwise version of the torch.allclose convergence test; NaN
        # residuals compare not-close and so count as failures.
        bad = ~(
            torch.isclose(A_hat, A, rtol=rtol, atol=atol)
            & torch.isclose(V_hat, V, rtol=rtol, atol=atol)
        )
        if bool(bad.any()):
            raise RuntimeError(
                f"gaussian_pullback did not converge: {int(bad.sum())}/{bad.numel()} "
                f"element(s) not within rtol={rtol:.1e}, atol={atol:.1e} after "
                f"{newton_iters} iterations "
                f"(worst residual {resid[bad].max().item():.3e}). The targets may be "
                "unattainable (e.g. A outside the range of f, or V too large for a "
                "bounded f), f may have a critical point near the solution, or the "
                "iteration budget may be too small. Consider raising newton_iters or "
                "num_points, trying a different mu_init, or passing "
                "check_convergence=False to get the best iterate anyway."
            )

    return mu, v.exp()


class _GaussianPullback(torch.autograd.Function):
    """Implicit-function-theorem gradients for the pullback.

    The solver returns theta = (mu, sigma) with F(theta; A, V, args) = 0 where
    F = (A_hat(theta, args) - A, V_hat(theta, args) - V). Differentiating the
    identity at the solution:

        dtheta/d(A, V) = J^{-1},        dtheta/dargs = -J^{-1} d(A_hat, V_hat)/dargs,

    with J = d(A_hat, V_hat)/d(mu, sigma) -- the same analytic quadrature-map
    Jacobian the Newton solve uses, evaluated once at the solution. The
    backward therefore costs one Jacobian pass (plus, if args need gradients,
    one VJP through the differentiable forward quadrature), independent of how
    many Newton iterations the solve took, and is exact for the quadrature-
    defined implicit function up to the solve tolerance.
    """

    @staticmethod
    def forward(
        ctx,
        f,
        kwargs,
        num_points,
        mu_init,
        newton_iters,
        rtol,
        atol,
        check_convergence,
        A,
        V,
        *args,
    ):
        mu, sigma = _solve_pullback(
            f,
            A,
            V,
            args,
            kwargs,
            num_points,
            mu_init,
            newton_iters,
            rtol,
            atol,
            check_convergence,
        )
        ctx.save_for_backward(mu, sigma, *args)
        ctx.f, ctx.kwargs, ctx.num_points = f, kwargs, num_points
        return mu, sigma

    @staticmethod
    @once_differentiable
    def backward(ctx, g_mu, g_sigma):
        mu, sigma, *args = ctx.saved_tensors
        args = tuple(args)
        f, kwargs, num_points = ctx.f, ctx.kwargs, ctx.num_points
        needs_A, needs_V = ctx.needs_input_grad[8], ctx.needs_input_grad[9]
        needs_args = ctx.needs_input_grad[10:]

        _, _, A_mu, A_sig, V_mu, V_sig = _quadrature_pass(
            f, mu, sigma, args, kwargs, num_points, jacobian=True
        )
        tiny = torch.finfo(mu.dtype).tiny
        det = A_mu * V_sig - A_sig * V_mu
        det = torch.where(det.abs() < tiny, torch.full_like(det, tiny), det)
        # lambda = J^{-T} [g_mu, g_sigma]; also equals [dL/dA, dL/dV] since
        # dF/d(A, V) = -I  =>  dtheta/d(A, V) = J^{-1}.
        lam_A = (V_sig * g_mu - V_mu * g_sigma) / det
        lam_V = (A_mu * g_sigma - A_sig * g_mu) / det

        grad_args: tuple[Tensor | None, ...] = tuple(None for _ in args)
        if any(needs_args):

            def push(*a: Tensor) -> tuple[Tensor, Tensor]:
                return gaussian_pushforward(f, mu, sigma, a, kwargs, num_points)

            with torch.enable_grad():
                _, vjp_fn = torch.func.vjp(push, *args)
                ga = vjp_fn((-lam_A, -lam_V))
            grad_args = tuple(g if need else None for g, need in zip(ga, needs_args))

        # lam_A / lam_V live in the broadcast shape; the autograd engine
        # sum-reduces grads over broadcast dims back to each input's shape
        # (validate_outputs applies sum_to for expandable-to shapes), so no
        # manual reduction is needed.
        grad_A = lam_A if needs_A else None
        grad_V = lam_V if needs_V else None
        return (
            None,  # f
            None,  # kwargs
            None,  # num_points
            None,  # mu_init
            None,  # newton_iters
            None,  # rtol
            None,  # atol
            None,  # check_convergence
            grad_A,
            grad_V,
            *grad_args,
        )


def gaussian_pullback(
    f: Callable[[Tensor, *tuple[Tensor, ...]], Tensor],
    A: Tensor,
    V: Tensor,
    args: tuple[Tensor, ...] = (),
    kwargs: dict[str, Any] | None = None,
    num_points: int = 16,
    mu_init: Tensor | None = None,
    newton_iters: int = 100,
    rtol: float | None = None,
    atol: float | None = None,
    check_convergence: bool = True,
) -> tuple[Tensor, Tensor]:
    """Recover (mu, sigma) such that E[f(X)] = A and Var[f(X)] = V.

    Solved by damped Newton in (mu, v = log sigma): the log parametrization
    keeps sigma > 0 for free and makes steps scale-invariant; the Jacobian is
    the exact quadrature-map Jacobian (see _quadrature_pass) and each
    step is globalized by a per-element backtracking line search on the
    residual max-norm (trial residuals cost one derivative-free quadrature
    pass; NaN/inf trials compare False and get backtracked automatically).

    Differentiable: gradients w.r.t. A, V, and args flow through an analytic
    implicit-function-theorem backward (see _GaussianPullback) rather than
    through the Newton iterations -- one Jacobian pass at the solution,
    regardless of iteration count. Gradient accuracy is limited by the solve
    tolerances `rtol`/`atol` (the IFT holds at the exact root). kwargs and
    mu_init are not differentiated
    (mu_init only selects the branch; the solution is locally independent of
    it), and double backward is not supported.

    Args:
        f, args, kwargs, num_points: As in gaussian_pushforward.
        A, V: Target mean and variance of f(X); V must be > 0 and attainable.
        mu_init: Starting point for the mean; selects the solution branch.
        newton_iters: Max iterations (for both the 1D init and the 2D solve).
        rtol, atol: Relative and absolute convergence tolerances; the solve
            stops when torch.allclose(A_hat, A, rtol=rtol, atol=atol) and
            torch.allclose(V_hat, V, rtol=rtol, atol=atol) both hold for the
            recovered moments (A_hat, V_hat). Specify both or neither; if
            neither is given, defaults are chosen from A's dtype following
            torch.testing.assert_close (e.g. rtol=1e-7, atol=1e-7 for
            float64; rtol=1.3e-6, atol=1e-5 for float32).
        check_convergence: If True (default), raise RuntimeError when any
            element's final moments fail the allclose test elementwise
            (torch.isclose) -- silent non-convergence
            would otherwise corrupt downstream results, and in particular the
            implicit-function gradients are only valid at a root. If False,
            return the best iterate unconditionally (the previous behavior);
            verify it via a forward round-trip.

    Returns:
        (mu, sigma) with shape broadcast(A, V, *args).

    Failure modes: convergence stalls near critical points of f (f' ~ 0 makes
    the Jacobian singular -- there sigma ~ sqrt(2V)/|f''| rather than
    sqrt(V)/|f'|; gradients blow up there too, as the true sensitivities do),
    and no solution exists if (A, V) is unattainable (e.g. A outside the range
    of f). Both surface as RuntimeError under check_convergence=True.
    """
    return _GaussianPullback.apply(
        f,
        kwargs or {},
        num_points,
        mu_init,
        newton_iters,
        rtol,
        atol,
        check_convergence,
        A,
        V,
        *args,
    )
