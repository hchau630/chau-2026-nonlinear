from math import sqrt

import pytest
import torch
from scipy import integrate, stats

# We need to import niarb to inject cdf function to torch.distributions.Beta
import niarb  # noqa
from niarb import random
from niarb.distributions import RectLinePicking, UniformEllipsoid


def test_beta_cdf():
    a, b = 2.0, 3.0
    value = torch.rand(10)
    beta = torch.distributions.Beta(a, b)

    torch.testing.assert_close(
        beta.cdf(value), torch.from_numpy(stats.beta(a, b).cdf(value)).float()
    )


def test_beta_cdf_grad():
    value = torch.rand(10, requires_grad=True)
    beta = torch.distributions.Beta(2.0, 3.0)
    with pytest.raises(NotImplementedError):
        beta.cdf(value / 2)


@pytest.mark.parametrize("a, b", [(1.5, 2.5), (1.5, 0.5)])
class TestRectLinePicking:
    def test_support(self, a, b):
        dist = RectLinePicking(a, b)
        torch.testing.assert_close(dist.support.lower_bound, 0.0)
        torch.testing.assert_close(
            dist.support.upper_bound, torch.tensor((a**2 + b**2) ** 0.5)
        )

    def test_unity(self, a, b):
        dist = RectLinePicking(a, b)
        torch.testing.assert_close(
            dist.cdf(dist.support.upper_bound), torch.tensor(1.0)
        )

    def test_rsample_cdf(self, a, b):
        dist = RectLinePicking(a, b)
        with random.set_seed(0):
            samples = dist.rsample((10000,))
        cdf = lambda x: dist.cdf(torch.from_numpy(x)).numpy()
        res = stats.kstest(samples, cdf)
        assert res.pvalue > 0.05

    def test_pdf_cdf(self, a, b):
        dist = RectLinePicking(a, b)
        r = torch.linspace(dist.support.lower_bound, dist.support.upper_bound, steps=50)
        pdf = lambda x: dist.log_prob(torch.tensor(x)).exp().numpy()
        out = torch.tensor([integrate.quad(pdf, 0.0, ri.item())[0] for ri in r])
        expected = dist.cdf(r)
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("r", [(0.5, 1.5)])
class TestUniformEllipsoid:
    def test_support(self, r):
        r = torch.tensor(r)
        dist = UniformEllipsoid(r)
        eps = 1e-5
        value = torch.tensor([[sqrt(2) - eps, sqrt(2)], [sqrt(2) + eps, sqrt(2)]]) / 2
        value = value * r
        out = dist.support.check(value)
        assert (out == torch.tensor([True, False])).all()

    def test_rsample_norm_cdf(self, r):
        r = torch.tensor(r)
        dist = UniformEllipsoid(r)
        with random.set_seed(0):
            samples = dist.rsample((10000,))
        samples = samples / r
        cdf = lambda x: x**2
        res = stats.kstest(torch.linalg.vector_norm(samples, dim=-1), cdf)
        assert res.pvalue > 0.05

    def test_rsample_angle_cdf(self, r):
        r = torch.tensor(r)
        dist = UniformEllipsoid(r)
        with random.set_seed(0):
            samples = dist.rsample((10000,))
        samples = samples / r
        cdf = lambda x: (x + torch.pi) / (2 * torch.pi)
        res = stats.kstest(torch.atan2(samples[:, 1], samples[:, 0]), cdf)
        assert res.pvalue > 0.05

    def test_log_prob(self, r):
        r = torch.tensor(r)
        dist = UniformEllipsoid(r)
        value = torch.tensor([[0.1, 0.2], [0.05, 0.01], [0.02, 0.03]])
        out = dist.log_prob(value)
        assert out.shape == (3,)
        assert (out == out[0]).all()
        torch.testing.assert_close(out[0], torch.log(1 / dist.volume()))

    def test_volume(self, r):
        r = torch.tensor(r)
        dist = UniformEllipsoid(r)
        out = dist.volume()
        expected = torch.pi * r[0] * r[1]
        torch.testing.assert_close(out, expected)
