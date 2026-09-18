import math
from contextlib import nullcontext

import pytest
import torch
from scipy import stats

from niarb import random


@pytest.mark.parametrize("validate_args", [True, False])
def test_log_normal_strict(validate_args):
    mean = torch.tensor([1.5, 0.1, 2.0, 1.0])
    std = torch.tensor([0.5, 0.2, 1.0, 0.7])

    with random.set_seed(0):
        out = random.log_normal_strict(
            mean, std, validate_args=validate_args, size=(1000,)
        )
    assert out.shape == (1000, 4)

    # Check if output has correct log-normal distribution
    for i in range(4):
        mean_i, std_i = mean[i].item(), std[i].item()

        loc = math.log(mean_i**2 / math.sqrt(mean_i**2 + std_i**2))
        scale = math.sqrt(math.log(1 + std_i**2 / mean_i**2))

        m = stats.lognorm(s=scale, scale=math.exp(loc))
        assert math.isclose(m.mean(), mean_i, rel_tol=1.3e-6)
        assert math.isclose(m.std(), std_i, rel_tol=1.3e-6)

        res = stats.kstest(out[:, i], m.cdf)
        assert res.pvalue > 0.05


@pytest.mark.parametrize("no_grad", [True, False])
@pytest.mark.parametrize("validate_args", [True, False])
def test_log_normal(no_grad, validate_args):
    func = random.log_normal_no_grad if no_grad else random.log_normal
    mean = torch.tensor([-1.5, 0.0, -2.0, 1.0])
    mean = mean.broadcast_to(1000, 4)
    std = torch.tensor([0.0, 0.0, 1.0, -0.5])

    if validate_args:
        mean, std = mean.abs(), std.abs()

    with random.set_seed(0):
        out = func(mean, std, validate_args=validate_args)
    mean, std = mean.abs(), std.abs()  # negative mean and std are treated as positive

    # Check if output has correct log-normal distribution when std > 0
    for i in [2, 3]:
        mean_i, std_i = mean[0, i].item(), std[i].item()

        loc = math.log(mean_i**2 / math.sqrt(mean_i**2 + std_i**2))
        scale = math.sqrt(math.log(1 + std_i**2 / mean_i**2))

        m = stats.lognorm(s=scale, scale=math.exp(loc))
        assert math.isclose(m.mean(), mean_i, rel_tol=1.3e-6)
        assert math.isclose(m.std(), std_i, rel_tol=1.3e-6)

        res = stats.kstest(out[:, i], m.cdf)
        assert res.pvalue > 0.05

    # Check if output is equal to mean when std = 0
    assert (out[:, :2] == mean[:, :2]).all()


@pytest.mark.parametrize("no_grad", [True, False])
def test_log_normal_edge_cases(no_grad):
    func = random.log_normal_no_grad if no_grad else random.log_normal
    nan, inf = float("nan"), float("inf")
    mean = torch.tensor(
        [
            [0.0, 2.0, 0.1, 1e-20, 1e20, -1.0, -1e20],
            [nan, nan, 2.0, 0.0, nan, nan, inf],
            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1e20],
        ],
        dtype=torch.float64,
    )
    std = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, nan, nan, nan, inf, nan],
            [inf, -inf, inf, -inf, 1.0, -1.0, inf],
        ],
        dtype=torch.float64,
    )

    samples = func(mean, std, validate_args=False)

    assert samples.shape == mean.shape
    assert samples.dtype == mean.dtype
    assert samples.device == mean.device
    print(samples)
    assert torch.equal(samples[0], mean[0].abs())
    assert torch.isnan(samples[1:]).all()


@pytest.mark.parametrize(
    "mean, std, valid",
    [
        (-1.0, -1.0, False),
        (0.0, -1.0, False),
        (1.0, -1.0, False),
        (-1.0, 0.0, False),
        (0.0, 0.0, True),
        (1.0, 0.0, True),
        (-1.0, 1.0, False),
        (0.0, 1.0, False),
        (1.0, 1.0, True),
    ],
)
def test_log_normal_validate_args(mean, std, valid):
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    with pytest.raises(ValueError) if not valid else nullcontext():
        random.log_normal(mean, std, validate_args=True)


def test_log_normal_grad():
    mean = torch.tensor(
        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.double, requires_grad=True
    )
    std = torch.tensor(
        [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.double, requires_grad=True
    )

    def func(mean, std):
        with random.set_seed(0):
            return random.log_normal(mean, std, validate_args=False)

    torch.autograd.gradcheck(func, (mean, std))


def test_zero_mean_zero_std_log_normal_grad():
    """Test gradient of log-normal w.r.t std at (0, 0).

    The gradient of log-normal w.r.t std is undefined at (0, 0), but here we define the
    gradient to be 0 for numerical convenience. We test this behavior here.
    """
    mean = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    std = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    sample = random.log_normal(mean, std)
    grad_mean, grad_std = torch.autograd.grad(sample, (mean, std))

    assert sample.item() == 0.0
    assert grad_mean.item() == 1.0
    assert grad_std.item() == 0.0
