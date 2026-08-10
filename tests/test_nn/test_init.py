import pytest
import torch

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
    shape = (1000, 2, 2)
    mask = torch.tensor(mask).broadcast_to(shape) if mask else None

    out = torch.empty(shape)
    with random.set_seed(0):
        nn.init.W_(out, 0.5, bounds, mask)
    assert out[out != 0.0].min().item() >= 1e-5
    assert out.max().item() <= 1.0

    bounds_shape = torch.tensor(bounds).shape
    if bounds_shape == (2,):
        expected = torch.empty(shape)
        with random.set_seed(0):
            torch.nn.init.trunc_normal_(expected, 0.0, 0.5, 1e-5, 1.0)
        if mask is not None:
            assert torch.allclose(out[mask], expected[mask])
        else:
            assert torch.allclose(out, expected)
    elif bounds_shape == (2, 2):
        assert out[..., 0].max().item() >= 0.5
        assert out[..., 1].max().item() <= 0.5
    else:
        assert out[:, 0, 0].max().item() >= 0.5
        assert out[:, 0, 1].max().item() <= 0.5
        assert out[:, 1, 0].max().item() <= 0.5
        if mask is None:
            assert out[:, 1, 1].max().item() >= 0.5

    if mask is not None:
        assert (out[~mask] == 0.0).all()
