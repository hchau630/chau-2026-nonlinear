import numpy as np
import pytest
import torch
from scipy import stats

from niarb import nn, random


@pytest.mark.parametrize(
    "bounds",
    [
        [1e-5, 1],
        [[1e-5, 1], [1e-5, 0.5]],
        [[[1e-5, 1], [1e-5, 0.5]], [[1e-5, 0.5], [1e-5, 1]]],
    ],
)
@pytest.mark.parametrize("mask", [None, [True, False]])
def test_W_(bounds, mask):
    shape = (10000, 2, 2)
    mask_ = torch.tensor(mask).broadcast_to(shape) if mask else None

    out = torch.empty(shape)
    with random.set_seed(0):
        nn.init.W_(out, 0.5, bounds, mask_)
    assert out[out != 0.0].min().item() >= 1e-5
    assert out.max().item() <= 1.0

    bounds_shape = torch.tensor(bounds).shape
    if bounds_shape == (2,):
        pass
    elif bounds_shape == (2, 2):
        assert out[..., 0].max().item() >= 0.5
        assert out[..., 1].max().item() <= 0.5
    else:
        assert out[:, 0, 0].max().item() >= 0.5
        assert out[:, 0, 1].max().item() <= 0.5
        assert out[:, 1, 0].max().item() <= 0.5
        if mask is None:
            assert out[:, 1, 1].max().item() >= 0.5

    if mask:
        assert (out[~mask_] == 0.0).all()

    bounds = torch.tensor(bounds).broadcast_to(shape[1:] + (2,))
    mask_ = (
        torch.ones(shape[1:], dtype=torch.bool)
        if mask is None
        else torch.tensor(mask).broadcast_to(shape[1:])
    )

    for i, j in np.ndindex(mask_.shape):
        expected = torch.zeros(shape[0])
        if mask_[i, j]:
            with random.set_seed(0):
                torch.nn.init.trunc_normal_(
                    expected, 0.0, 0.5, bounds[i, j, 0].item(), bounds[i, j, 1].item()
                )
        assert stats.kstest(out[..., i, j].numpy(), expected.numpy()).pvalue > 0.05
