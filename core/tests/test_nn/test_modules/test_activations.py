import math

import pytest
import torch

from niarb import nn
from niarb.gaussian_moments import gaussian_pushforward


def _quadrature_moments(f, n, mu, sigma, *, num_points):
    """Compute E[f(X)**k] and its mu/sigma derivatives by quadrature."""
    mu_quad = mu.detach().clone().requires_grad_()
    sigma_quad = sigma.detach().clone().requires_grad_()
    moments, moments_mu, moments_sigma = [], [], []

    for k in n:
        # The lambda deliberately has no gaussian_moments attribute, so
        # gaussian_pushforward must use its Gauss-Hermite implementation.
        moment, _ = gaussian_pushforward(
            lambda x, k=k: f(x) ** k,
            mu_quad,
            sigma_quad,
            num_points=num_points,
        )
        moment_mu, moment_sigma = torch.autograd.grad(
            moment.sum(), (mu_quad, sigma_quad)
        )
        moments.append(moment.detach())
        moments_mu.append(moment_mu.detach())
        moments_sigma.append(moment_sigma.detach())

    return moments, moments_mu, moments_sigma


def test_ssn2_gaussian_moments_matches_quadrature():
    f = nn.SSN(2)
    n = [0, 1, 2, 3]
    mu = torch.tensor([0.8, 1.0, 2.0], dtype=torch.double)
    sigma = torch.tensor([0.4, 0.4, 1.0], dtype=torch.double)

    actual = f.gaussian_moments(n, mu, sigma, jacobian=True)
    expected = _quadrature_moments(f, n, mu, sigma, num_points=40000)

    for actual_group, expected_group in zip(actual, expected):
        for i, (value, target) in enumerate(zip(actual_group, expected_group)):
            if i > 0:
                assert (value != target).any()
            torch.testing.assert_close(value, target, rtol=1e-6, atol=1e-6)


def test_rectified_gaussian_moments():
    mu = torch.tensor([-0.8, 0.3, 1.2], dtype=torch.double)
    sigma = torch.tensor([0.4, 0.7, 0.2], dtype=torch.double)
    z = mu / sigma
    cdf = torch.special.ndtr(z)
    pdf = torch.exp(-z.square() / 2) / math.sqrt(2 * math.pi)

    m1 = mu * cdf + sigma * pdf
    m2 = mu * m1 + sigma.square() * cdf
    m3 = mu * m2 + 2 * sigma.square() * m1
    moments, moments_mu, moments_sigma = nn.Rectified().gaussian_moments(
        [0, 1, 2, 3], mu, sigma, jacobian=True
    )

    expected = [torch.ones_like(mu), m1, m2, m3]
    expected_mu = [torch.zeros_like(mu), cdf, 2 * m1, 3 * m2]
    expected_sigma = [
        torch.zeros_like(mu),
        pdf,
        2 * sigma * cdf,
        6 * sigma * m1,
    ]
    for actual, target in zip(moments, expected):
        torch.testing.assert_close(actual, target)
    for actual, target in zip(moments_mu, expected_mu):
        torch.testing.assert_close(actual, target)
    for actual, target in zip(moments_sigma, expected_sigma):
        torch.testing.assert_close(actual, target)


def test_rectified_gaussian_moments_zero_sigma_limit():
    mu = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.double)
    sigma = torch.zeros_like(mu)
    moments = nn.Rectified().gaussian_moments([0, 1, 2], mu, sigma)
    torch.testing.assert_close(moments[0], torch.ones_like(mu))
    torch.testing.assert_close(moments[1], mu.relu())
    torch.testing.assert_close(moments[2], mu.relu().square())


def test_rectified_gaussian_moments_compose_with_pow():
    mu = torch.tensor([-0.4, 0.6], dtype=torch.double)
    sigma = torch.tensor([0.3, 0.8], dtype=torch.double)
    actual = nn.SSN(3).gaussian_moments([1, 2], mu, sigma, jacobian=True)
    expected = nn.Rectified().gaussian_moments([3, 6], mu, sigma, jacobian=True)
    for actual_group, expected_group in zip(actual, expected):
        for value, target in zip(actual_group, expected_group):
            torch.testing.assert_close(value, target)


def test_rectified_gaussian_moments_nonzero_threshold_raises():
    with pytest.raises(NotImplementedError):
        nn.Rectified(threshold=1.0).gaussian_moments(
            [1, 2], torch.tensor(0.0), torch.tensor(1.0)
        )
