"""Tests for gaussian_moments.gaussian_pushforward / gaussian_pullback.

Reference values come from three independent sources: closed forms (lognormal,
Gaussian bump, monomial), Gaussian moment identities (polynomial exactness of
the quadrature), and seeded Monte Carlo. Pullback correctness is defined by
moment residuals -- (mu, sigma) recovering the *target moments* -- since the
inverse problem is multi-valued and parameter recovery is only guaranteed up
to branch.
"""

import pytest
import torch
from torch import Tensor

from niarb import nn
from niarb.gaussian_moments import (
    _quadrature_pass,
    gaussian_pullback,
    gaussian_pushforward,
)


def T(data) -> Tensor:
    return torch.tensor(data, dtype=torch.double)


# ---------------------------------------------------------------------------
# Reference functions and their exact moments
# ---------------------------------------------------------------------------


def f_exp(x: Tensor) -> Tensor:
    return torch.exp(x)


def lognormal_moments(mu: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
    A = torch.exp(mu + sigma.square() / 2)
    return A, A.square() * torch.expm1(sigma.square())


def f_bump(x: Tensor, c: Tensor, ell: Tensor) -> Tensor:
    return torch.exp(-((x - c).square()) / (2 * ell.square()))


def bump_moments(
    mu: Tensor, sigma: Tensor, c: Tensor, ell: Tensor
) -> tuple[Tensor, Tensor]:
    d, s, l2 = mu - c, sigma.square(), ell.square()
    A = ell / (l2 + s).sqrt() * torch.exp(-d.square() / (2 * (l2 + s)))
    E2 = ell / (l2 + 2 * s).sqrt() * torch.exp(-d.square() / (l2 + 2 * s))
    return A, E2 - A.square()


def f_cubic(x: Tensor) -> Tensor:
    return x**3


def cubic_moments(mu: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
    m, s2 = mu, sigma.square()
    A = m**3 + 3 * m * s2
    V = 9 * m**4 * s2 + 36 * m**2 * s2**2 + 15 * s2**3
    return A, V


def f_mixed(x: Tensor) -> Tensor:
    return torch.exp(0.7 * x) + 0.5 * torch.exp(-((x - 1.0).square()))


def assert_moments_match(
    f, mu, sigma, A, V, args=(), kwargs=None, num_points=16, **tol
):
    """The pullback's contract: recovered (mu, sigma) reproduces the targets."""
    A2, V2 = gaussian_pushforward(
        f, mu, sigma, args=args, kwargs=kwargs, num_points=num_points
    )
    torch.testing.assert_close(A2, A, **tol)
    torch.testing.assert_close(V2, V, **tol)


# ---------------------------------------------------------------------------
# gaussian_pushforward
# ---------------------------------------------------------------------------


class TestPushforward:
    def test_analytic_polynomial_moments_do_not_depend_on_node_count(self):
        mu, sigma = T([0.7, -1.2]), T([0.9, 2.0])
        A, V = gaussian_pushforward(nn.Pow(3), mu, sigma, num_points=1)
        A_t, V_t = cubic_moments(mu, sigma)
        torch.testing.assert_close(A, A_t)
        torch.testing.assert_close(V, V_t)

    def test_analytic_jacobian(self):
        mu, sigma = T([0.4, -0.7]), T([0.3, 0.8])
        A, V, A_mu, A_sig, V_mu, V_sig = _quadrature_pass(
            nn.Pow(2), mu, sigma, (), {}, num_points=1, jacobian=True
        )
        torch.testing.assert_close(A, mu.square() + sigma.square())
        torch.testing.assert_close(V, 4 * mu.square() * sigma.square() + 2 * sigma**4)
        torch.testing.assert_close(A_mu, 2 * mu)
        torch.testing.assert_close(A_sig, 2 * sigma)
        torch.testing.assert_close(V_mu, 8 * mu * sigma.square())
        torch.testing.assert_close(V_sig, 8 * sigma * (mu.square() + sigma.square()))

    def test_any_nan_falls_back_all_outputs_for_that_element(self):
        class PartiallyAnalyticSquare(nn.Pow):
            def __init__(self):
                super().__init__(2)

            def gaussian_moments(self, n, mu, sigma, jacobian=False):
                out = super().gaussian_moments(n, mu, sigma, jacobian)
                if not jacobian:
                    return out
                moments, moments_mu, moments_sigma = out
                moments_mu[1] = torch.where(
                    mu < 0,
                    torch.full_like(moments_mu[1], torch.nan),
                    moments_mu[1],
                )
                return moments, moments_mu, moments_sigma

        mu, sigma = T([-0.5, 0.5]), T([0.3, 0.3])
        actual = _quadrature_pass(
            PartiallyAnalyticSquare(),
            mu,
            sigma,
            (),
            {},
            num_points=1,
            jacobian=True,
        )

        exact = (
            mu.square() + sigma.square(),
            4 * mu.square() * sigma.square() + 2 * sigma**4,
            2 * mu,
            2 * sigma,
            8 * mu * sigma.square(),
            8 * sigma * (mu.square() + sigma.square()),
        )
        quadrature = (
            mu.square(),
            torch.zeros_like(mu),
            2 * mu,
            torch.zeros_like(mu),
            torch.zeros_like(mu),
            torch.zeros_like(mu),
        )
        fallback = mu < 0
        for value, exact_value, quadrature_value in zip(actual, exact, quadrature):
            torch.testing.assert_close(
                value, torch.where(fallback, quadrature_value, exact_value)
            )

    def test_match_uses_quadrature_only_for_nan_moments(self):
        f = nn.Match({1: nn.Identity()}, torch.exp)
        mu, sigma = T([0.4, -0.2]), T([0.7, 0.5])
        key = torch.tensor([1, 0])
        A, V = gaussian_pushforward(f, mu, sigma, args=(key,), num_points=1)

        # One-point quadrature evaluates the unsupported exp branch at x=mu,
        # while the Identity branch still receives its exact variance.
        torch.testing.assert_close(A, torch.stack([mu[0], mu[1].exp()]))
        torch.testing.assert_close(V, torch.stack([sigma[0].square(), T(0.0)]))

    @pytest.mark.parametrize("sigma_val", [0.1, 0.5, 1.0, 1.5])
    def test_exp_matches_lognormal(self, sigma_val):
        mu = torch.linspace(-1.0, 1.5, 7, dtype=torch.double)
        sigma = torch.full_like(mu, sigma_val)
        A, V = gaussian_pushforward(f_exp, mu, sigma)
        A_t, V_t = lognormal_moments(mu, sigma)
        torch.testing.assert_close(A, A_t)
        torch.testing.assert_close(V, V_t)

    def test_bump_with_args_matches_closed_form(self):
        c, ell = T(0.3), T(1.5)
        mu = T([[-0.5, 0.2], [0.8, 1.0]])
        sigma = T([[0.2, 0.35], [0.15, 0.4]])
        A, V = gaussian_pushforward(f_bump, mu, sigma, args=(c, ell))
        A_t, V_t = bump_moments(mu, sigma, c, ell)
        torch.testing.assert_close(A, A_t)
        torch.testing.assert_close(V, V_t)

    def test_polynomial_exactness_at_minimal_node_count(self):
        # V of x^3 involves E[X^6]; k = 4 integrates degree <= 7 exactly, so
        # the quadrature error must be pure round-off even at k = 4.
        mu, sigma = T([0.7, -1.2]), T([0.9, 2.0])
        A, V = gaussian_pushforward(f_cubic, mu, sigma, num_points=4)
        A_t, V_t = cubic_moments(mu, sigma)
        torch.testing.assert_close(A, A_t, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(V, V_t, rtol=1e-13, atol=1e-13)

    def test_large_sigma_converges_with_node_count(self):
        c, ell = T(0.3), T(1.5)
        mu, sigma = T([0.5]), T([3.0])
        A_t, V_t = bump_moments(mu, sigma, c, ell)
        _, V64 = gaussian_pushforward(f_bump, mu, sigma, args=(c, ell), num_points=64)
        A128, V128 = gaussian_pushforward(
            f_bump, mu, sigma, args=(c, ell), num_points=128
        )
        assert (V128 - V_t).abs().item() < (V64 - V_t).abs().item()
        torch.testing.assert_close(A128, A_t, rtol=1e-10, atol=0)
        torch.testing.assert_close(V128, V_t, rtol=1e-10, atol=0)

    def test_variance_stable_when_V_much_smaller_than_A_squared(self):
        # V / A^2 ~ 1e-14: the naive E[f^2] - A^2 loses ~half the digits here
        mu, sigma = T([10.0]), T([1e-7])
        A, V = gaussian_pushforward(f_exp, mu, sigma)
        A_t, V_t = lognormal_moments(mu, sigma)
        assert V.item() > 0
        torch.testing.assert_close(A, A_t)
        torch.testing.assert_close(V, V_t)

    def test_matches_monte_carlo_for_nonanalytic_reference(self):
        torch.manual_seed(0)
        mu, sigma = T(0.4), T(0.8)
        A, V = gaussian_pushforward(torch.tanh, mu, sigma)
        x = mu + sigma * torch.randn(4_000_000, dtype=torch.double)
        fx = torch.tanh(x)
        torch.testing.assert_close(A, fx.mean(), rtol=1e-3, atol=1e-7)
        torch.testing.assert_close(V, fx.var(), rtol=1e-3, atol=1e-7)

    def test_broadcasting_and_kwargs(self):
        def f(x: Tensor, a: Tensor, *, p: float = 1.0) -> Tensor:
            return torch.exp(p * a * x)

        mu = torch.zeros(2, 1, dtype=torch.double)
        sigma = T([0.2, 0.3, 0.4])
        a = T(0.5)
        A, V = gaussian_pushforward(f, mu, sigma, args=(a,), kwargs={"p": 2.0})
        assert A.shape == (2, 3) and V.shape == (2, 3)
        A_t, V_t = lognormal_moments(2.0 * a * mu, 2.0 * a * sigma.expand(2, 3))
        torch.testing.assert_close(A, A_t)
        torch.testing.assert_close(V, V_t)

    def test_dtype_preserved(self):
        mu = torch.tensor([0.2], dtype=torch.float32)
        sigma = torch.tensor([0.3], dtype=torch.float32)
        A, V = gaussian_pushforward(f_exp, mu, sigma)
        assert A.dtype == torch.float32 and V.dtype == torch.float32
        A_t, V_t = lognormal_moments(mu.double(), sigma.double())
        torch.testing.assert_close(A, A_t.float())
        torch.testing.assert_close(V, V_t.float())

    def test_zero_sigma_is_deterministic_limit(self):
        mu, sigma = T([0.7]), T([0.0])
        A, V = gaussian_pushforward(f_exp, mu, sigma)
        torch.testing.assert_close(A, mu.exp(), rtol=0, atol=1e-20)
        torch.testing.assert_close(V, torch.zeros_like(V), rtol=0, atol=1e-20)


# ---------------------------------------------------------------------------
# gaussian_pullback
# ---------------------------------------------------------------------------


class TestPullback:
    def test_exp_matches_exact_lognormal_inverse(self):
        # For f = exp the inverse is unique and closed-form.
        mu_t = T([-1.0, 0.0, 0.5, 1.5])
        sigma_t = T([0.1, 0.5, 1.0, 1.5])
        A, V = lognormal_moments(mu_t, sigma_t)
        mu, sigma = gaussian_pullback(f_exp, A, V)
        s2 = torch.log1p(V / A.square())
        torch.testing.assert_close(mu, A.log() - s2 / 2)
        torch.testing.assert_close(sigma, s2.sqrt())

    def test_roundtrip_recovers_parameters(self):
        mu_t = T([-0.8, 0.1, 1.2])
        sigma_t = T([0.25, 0.4, 0.3])
        A, V = gaussian_pushforward(f_mixed, mu_t, sigma_t)
        mu, sigma = gaussian_pullback(
            f_mixed, A, V, mu_init=torch.zeros(3, dtype=torch.double)
        )
        torch.testing.assert_close(mu, mu_t)
        torch.testing.assert_close(sigma, sigma_t)

    def test_roundtrip_with_args_and_kwargs(self):
        def f(x: Tensor, a: Tensor, *, p: float = 1.0) -> Tensor:
            return torch.exp(p * a * x) + x.square() * 0.1

        a = T(0.5)
        mu_t, sigma_t = T([0.3, -0.2]), T([0.2, 0.35])
        mu_init = torch.zeros(2, dtype=torch.double)
        A, V = gaussian_pushforward(f, mu_t, sigma_t, args=(a,), kwargs={"p": 2.0})
        mu, sigma = gaussian_pullback(
            f, A, V, args=(a,), kwargs={"p": 2.0}, mu_init=mu_init
        )
        torch.testing.assert_close(mu, mu_t)
        torch.testing.assert_close(sigma, sigma_t)

    def test_large_sigma_bump_satisfies_moment_equations(self):
        # The initializer's sigma -> 0 basin differs from the true solution;
        # the globalized Newton must still drive the residuals to ~0. The
        # solution found may be a different valid branch (the bump is
        # symmetric about c), so we assert on moments, not parameters.
        c, ell = T(0.3), T(1.5)
        A, V = gaussian_pushforward(f_bump, T([0.5]), T([2.5]), args=(c, ell))
        mu, sigma = gaussian_pullback(f_bump, A, V, args=(c, ell), mu_init=T([0.0]))
        assert torch.isfinite(mu).all() and torch.isfinite(sigma).all()
        assert_moments_match(f_bump, mu, sigma, A, V, args=(c, ell))

    def test_mu_init_selects_branch(self):
        # f(mu) = A has two roots for a bump; both must be reachable and both
        # must reproduce the target moments.
        c, ell = T(0.0), T(1.0)
        A, V = gaussian_pushforward(f_bump, T([0.6]), T([0.2]), args=(c, ell))
        mu_r, sg_r = gaussian_pullback(f_bump, A, V, args=(c, ell), mu_init=T([1.0]))
        mu_l, sg_l = gaussian_pullback(f_bump, A, V, args=(c, ell), mu_init=T([-1.0]))
        torch.testing.assert_close(mu_r, T([0.6]))
        torch.testing.assert_close(mu_l, T([-0.6]))
        torch.testing.assert_close(sg_r, sg_l)
        assert_moments_match(f_bump, mu_l, sg_l, A, V, args=(c, ell))

    def test_batched_targets_solved_elementwise(self):
        mu_t = T([[-0.5, 0.2], [0.8, 1.0]])
        sigma_t = T([[0.2, 0.35], [0.15, 0.4]])
        A, V = gaussian_pushforward(f_exp, mu_t, sigma_t)
        mu, sigma = gaussian_pullback(f_exp, A, V)
        assert mu.shape == (2, 2) and sigma.shape == (2, 2)
        torch.testing.assert_close(mu, mu_t)
        torch.testing.assert_close(sigma, sigma_t)

    def test_sigma_always_positive(self):
        A, V = lognormal_moments(T([0.0]), T([1e-3]))
        _, sigma = gaussian_pullback(f_exp, A, V)
        assert (sigma > 0).all()
        torch.testing.assert_close(sigma, T([1e-3]))

    def test_raises_on_unattainable_target(self):
        # The bump's range is (0, 1], so E[f(X)] = 2 is impossible for any
        # (mu, sigma); the residual stalls at its infimum and must raise.
        c, ell = T(0.0), T(1.0)
        with pytest.raises(RuntimeError, match="did not converge"):
            gaussian_pullback(
                f_bump, T([2.0]), T([0.1]), args=(c, ell), newton_iters=15
            )

    def test_check_convergence_false_returns_best_iterate(self):
        c, ell = T(0.0), T(1.0)
        mu, sigma = gaussian_pullback(
            f_bump,
            T([2.0]),
            T([0.1]),
            args=(c, ell),
            newton_iters=15,
            check_convergence=False,
        )
        assert mu.shape == (1,) and sigma.shape == (1,)
        assert (sigma > 0).all()

    def test_partial_batch_failure_raises_and_reports_count(self):
        # One attainable target alongside one unattainable: still an error,
        # and the message must attribute it to exactly 1 of 2 elements.
        c, ell = T(0.0), T(1.0)
        A_ok, V_ok = gaussian_pushforward(f_bump, T([0.4]), T([0.3]), args=(c, ell))
        A = torch.cat([A_ok, T([2.0])])
        V = torch.cat([V_ok, T([0.1])])
        with pytest.raises(RuntimeError, match=r"1/2"):
            gaussian_pullback(
                f_bump, A, V, args=(c, ell), newton_iters=15, mu_init=T([1.0, 1.0])
            )

    def test_tol_controls_residuals(self):
        mu_t, sigma_t = T([0.3]), T([0.4])
        A, V = gaussian_pushforward(f_mixed, mu_t, sigma_t)
        for rtol, atol in [(1e-7, 1e-7), (1e-12, 1e-12)]:
            mu, sigma = gaussian_pullback(f_mixed, A, V, rtol=rtol, atol=atol)
            A2, V2 = gaussian_pushforward(f_mixed, mu, sigma)
            torch.testing.assert_close(A2, A, rtol=rtol, atol=atol)
            torch.testing.assert_close(V2, V, rtol=rtol, atol=atol)


class TestPullbackBackward:
    def test_grads_match_exact_lognormal_inverse(self):
        # For f = exp the inverse is closed-form; autograd through that closed
        # form gives reference gradients for the IFT backward.
        A = T([1.3, 2.0]).requires_grad_()
        V = T([0.2, 0.9]).requires_grad_()
        mu, sigma = gaussian_pullback(f_exp, A, V)
        g_mu = T([0.7, -1.1])
        g_sigma = T([0.3, 2.0])
        gA, gV = torch.autograd.grad((mu * g_mu + sigma * g_sigma).sum(), (A, V))

        A2 = A.detach().requires_grad_()
        V2 = V.detach().requires_grad_()
        s2 = torch.log1p(V2 / A2.square())
        mu_c, sigma_c = A2.log() - s2 / 2, s2.sqrt()
        gA_c, gV_c = torch.autograd.grad(
            (mu_c * g_mu + sigma_c * g_sigma).sum(), (A2, V2)
        )
        torch.testing.assert_close(gA, gA_c)
        torch.testing.assert_close(gV, gV_c)

    def test_gradcheck_wrt_targets(self):
        mu_t, sigma_t = T([0.1, 0.6]), T([0.3, 0.2])
        A, V = gaussian_pushforward(f_mixed, mu_t, sigma_t)
        A = A.detach().requires_grad_()
        V = V.detach().requires_grad_()

        def fn(A: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
            return gaussian_pullback(
                f_mixed, A, V, mu_init=torch.zeros(2, dtype=torch.double)
            )

        assert torch.autograd.gradcheck(fn, (A, V))

    def test_gradcheck_wrt_args(self):
        c, ell = T(0.0), T(1.0)
        A, V = gaussian_pushforward(f_bump, T([0.6]), T([0.25]), args=(c, ell))
        A = A.detach().requires_grad_()
        V = V.detach().requires_grad_()
        c = c.detach().requires_grad_()

        def fn(A: Tensor, V: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
            return gaussian_pullback(f_bump, A, V, args=(c, ell), mu_init=T([1.0]))

        assert torch.autograd.gradcheck(fn, (A, V, c))

    def test_broadcast_targets_get_reduced_grads(self):
        # A of shape (2, 1) against V of shape (3,): grads must come back in
        # the original (unbroadcast) shapes.
        mu_t = T([[0.1], [0.4]]).expand(2, 3)
        sigma_t = T([0.2, 0.3, 0.25]).expand(2, 3)
        A_full, V_full = gaussian_pushforward(f_exp, mu_t, sigma_t)
        A = (
            A_full[:, :1].detach().clone().requires_grad_()
        )  # (2, 1) broadcastable slice
        V = V_full.detach().clone().requires_grad_()  # (2, 3)
        mu, sigma = gaussian_pullback(f_exp, A, V)
        (mu.sum() + sigma.sum()).backward()
        assert A.grad.shape == (2, 1) and V.grad.shape == (2, 3)
        assert torch.isfinite(A.grad).all() and torch.isfinite(V.grad).all()

    def test_no_grad_inputs_still_work(self):
        A, V = lognormal_moments(T([0.2]), T([0.4]))
        mu, sigma = gaussian_pullback(f_exp, A, V)
        assert not mu.requires_grad and not sigma.requires_grad
