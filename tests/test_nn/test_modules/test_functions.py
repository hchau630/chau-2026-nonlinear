import pytest
import torch

from niarb import nn
from niarb.gaussian_moments import gaussian_pushforward
from niarb.tensors import categorical


@pytest.fixture
def x():
    return torch.linspace(-5.0, 5.0, steps=50)


@pytest.mark.parametrize(
    "f, xlim",
    [
        ("nn.Identity()", (-5.0, 5.0)),
        ("nn.Pow(2.5,)", (0.0, 5.0)),
        ("nn.Threshold(-1.0, -2.0)", (-1.0 + 1.0e-5, 5.0)),
        ("nn.Rectified(-1.0,)", (-1.0 + 1.0e-5, 5.0)),
        ("nn.Rectified() ** 2", (1.0e-5, 5.0)),
    ],
)
def test_inv(f, xlim):
    f = eval(f)
    x = torch.linspace(*xlim, steps=50)
    torch.testing.assert_close(f.inv()(f(x)), x)
    torch.testing.assert_close(f(f.inv()(x)), x)


def test_identity(x):
    torch.testing.assert_close(nn.Identity()(x), x)


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


def test_identity_gaussian_moments_matches_quadrature():
    f = nn.Identity()
    n = [0, 1, 2, 3, 4, 6]
    mu = torch.tensor([-0.7, 0.4, 1.3], dtype=torch.double)
    sigma = torch.tensor([0.3, 1.2, 0.8], dtype=torch.double)

    actual = f.gaussian_moments(n, mu, sigma, jacobian=True)
    expected = _quadrature_moments(f, n, mu, sigma, num_points=8)

    for actual_group, expected_group in zip(actual, expected):
        for i, (value, target) in enumerate(zip(actual_group, expected_group)):
            if i > 1:
                assert (value != target).any()
            torch.testing.assert_close(value, target, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("pow", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_pow(pow, x):
    torch.testing.assert_close(nn.Pow(pow)(x), x**pow, equal_nan=True)


def test_pow_gaussian_moments():
    mu = torch.tensor([0.2, -0.5], dtype=torch.double)
    sigma = torch.tensor([0.7, 0.4], dtype=torch.double)
    actual = nn.Pow(3).gaussian_moments([1, 2], mu, sigma, jacobian=True)
    expected = nn.Identity().gaussian_moments([3, 6], mu, sigma, jacobian=True)
    for actual_group, expected_group in zip(actual, expected):
        for value, target in zip(actual_group, expected_group):
            torch.testing.assert_close(value, target)


@pytest.mark.parametrize("p", [-1, 0.5, 2.0])
def test_pow_gaussian_moments_unsupported(p):
    with pytest.raises(NotImplementedError):
        nn.Pow(p).gaussian_moments([1, 2], torch.tensor(0.0), torch.tensor(1.0))


@pytest.mark.parametrize("f", ["nn.Identity()", "nn.Rectified(threshold=1.0)"])
@pytest.mark.parametrize("pow", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_functional_pow(f, pow, x):
    f = eval(f)
    torch.testing.assert_close((f**pow)(x), f(x) ** pow, equal_nan=True)


def test_compose_gaussian_moments():
    mu = torch.tensor([-0.3, 0.8], dtype=torch.double)
    sigma = torch.tensor([0.2, 0.6], dtype=torch.double)
    actual = (nn.Identity() ** 3).gaussian_moments([1, 2], mu, sigma, jacobian=True)
    expected = nn.Identity().gaussian_moments([3, 6], mu, sigma, jacobian=True)
    for actual_group, expected_group in zip(actual, expected):
        for value, target in zip(actual_group, expected_group):
            torch.testing.assert_close(value, target)


@pytest.mark.parametrize(
    "f",
    [
        "nn.Compose(nn.Identity(), nn.Identity())",
        "nn.Compose(nn.Pow(2))",
        "nn.Compose(nn.Pow(2), nn.Identity(), nn.Identity())",
        "nn.Compose(nn.Pow(2), torch.nn.Sigmoid())",
        "nn.Compose(nn.Pow(0.5), nn.Identity())",
        "nn.Compose(nn.Pow(2), x=nn.Identity())",
    ],
)
def test_compose_gaussian_moments_unsupported(f):
    f = eval(f)
    result = f.gaussian_moments(
        [1, 2], torch.tensor([0.0, 1.0]), torch.tensor(0.5), jacobian=True
    )
    assert all(torch.isnan(value).all() for group in result for value in group)


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_threshold(threshold, value, x):
    torch.testing.assert_close(
        nn.Threshold(threshold, value)(x), torch.where(x > threshold, x, value)
    )


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0])
def test_rectified(threshold, x):
    torch.testing.assert_close(
        nn.Rectified(threshold=threshold)(x), torch.where(x > threshold, x, threshold)
    )


def test_match():
    f = nn.Match({"PV": nn.SSN(3)}, nn.SSN(2))
    x = torch.randn(10)
    key = categorical.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2], categories=["PYR", "PV", "SST"]
    )
    out = f(x, key)
    expected = nn.SSN(2)(x)
    expected[3:6] = nn.SSN(3)(x[3:6])
    torch.testing.assert_close(out, expected)


def test_match_gaussian_moments_are_selected_elementwise():
    f = nn.Match({1: nn.Pow(2)}, nn.Identity())
    mu = torch.tensor([-0.5, 0.2, 1.0], dtype=torch.double)
    sigma = torch.tensor([0.3, 0.4, 0.7], dtype=torch.double)
    key = torch.tensor([0, 1, 0])

    actual = f.gaussian_moments([1, 2], mu, sigma, True, key)
    identity = nn.Identity().gaussian_moments([1, 2], mu, sigma, True)
    square = nn.Pow(2).gaussian_moments([1, 2], mu, sigma, True)
    mask = key == 1
    for actual_group, identity_group, square_group in zip(actual, identity, square):
        for value, identity_value, square_value in zip(
            actual_group, identity_group, square_group
        ):
            torch.testing.assert_close(
                value, torch.where(mask, square_value, identity_value)
            )


def test_match_gaussian_moments_marks_unsupported_branches_nan():
    f = nn.Match({1: nn.Identity()}, torch.exp)
    mu = torch.tensor([0.0, 0.5], dtype=torch.double)
    sigma = torch.ones_like(mu)
    moments = f.gaussian_moments([1, 2], mu, sigma, False, torch.tensor([1, 0]))
    assert all(
        torch.isfinite(moment[0]) and torch.isnan(moment[1]) for moment in moments
    )
