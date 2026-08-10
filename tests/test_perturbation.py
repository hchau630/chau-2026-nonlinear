import contextlib
from math import exp

import numpy as np
import pytest
import torch

from niarb import perturbation, random, special
from niarb.nn.modules import frame
from niarb.tensors import categorical, periodic

atan21 = np.arctan2(2.0, 1.0) / np.pi * 180
atan12 = np.arctan2(1.0, 2.0) / np.pi * 180
sqrt2 = 2**0.5
sqrt5 = 5**0.5
hpi = np.pi / 2


@pytest.fixture
def x():
    return frame.ParameterFrame(
        {
            "cell_type": categorical.as_tensor(
                [[0, 0, 1, 1, 1], [1, 1, 1, 0, 1]], categories=["PYR", "PV"]
            ),
            "space": periodic.as_tensor(
                torch.tensor(
                    [
                        [[0.0, -5.0], [0.0, -3.0], [0.0, 0.0], [0.0, 2.0], [0.0, 5.0]],
                        [[1.0, -5.0], [1.0, -3.0], [1.0, 0.0], [1.0, 2.0], [1.0, 5.0]],
                    ]
                ),
                w_dims=[0, 1],
                extents=[(-5.0, 5.0)] * 2,
            ),
            "ori": periodic.as_tensor(
                torch.tensor(
                    [
                        [[0.0], [0.0], [90.0], [0.0], [90.0]],
                        [[90.0], [90.0], [0.0], [0.0], [0.0]],
                    ]
                ),
                w_dims=[0],
                extents=[(-180.0, 180.0)],
            ),
            "osi": torch.tensor([[1.0, 0.5, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 0.5, 0.0]]),
            "dh": torch.tensor(
                [
                    [0.0, 1.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0],
                ]
            ),
        },
        ndim=2,
    )


def test_min_distance():
    x = torch.tensor([[-1.0, -1.0], [0.0, 0.0]])
    y = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    scale = [1, 2]
    output = perturbation.min_distance(x, y, scale=scale)
    expected = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "dim, expected",
    [
        (-1, [[0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]]),
        (1, [[0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]]),
        ((1,), [[0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]]),
        (0, [[1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]]),
        ((0, 1), [[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]),
        (range(2), [[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]),
    ],
)
def test_min_distance_to_ensemble(x, dim, expected):
    output = x.apply(perturbation.min_distance_to_ensemble, dim=dim)
    expected = torch.tensor(expected).float()
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "dim, expected",
    [
        (-1, [[2.0], [2.0]]),
        (1, [[2.0], [2.0]]),
        ((1,), [[2.0], [2.0]]),
        (0, [[np.nan, np.nan, 1.0, np.nan, np.nan]]),
        ((0, 1), [[1.0]]),
        (range(2), [[1.0]]),
    ],
)
def test_inter_target_distance_statistics(x, dim, expected):
    output = x.apply(
        perturbation.inter_target_distance_statistics,
        dim=dim,
        args=(["min"],),
        keepdim=True,
    )

    assert isinstance(output, frame.ParameterFrame)
    assert output.ndim == 2
    assert list(output.keys()) == ["min"]

    expected = torch.tensor(expected).float()
    torch.testing.assert_close(output["min"], expected, equal_nan=True)


@pytest.mark.parametrize(
    "dim, osi, expected",
    [
        (-1, None, [[[atan21]], [[atan12]]]),
        (1, None, [[[atan21]], [[atan12]]]),
        ((1,), None, [[[atan21]], [[atan12]]]),
        (0, None, [[[90.0], [0.0], [45.0], [0.0], [90.0]]]),
        ((0, 1), None, [[[45.0]]]),
        (range(2), None, [[[45.0]]]),
        (0, [[0.0], [1.0]], [[[90.0], [np.nan], [0.0], [0.0], [np.nan]]]),
        (1, [[0.0, 1.0, 1.0, 0.0, 0.0]], [[[45.0]], [[0.0]]]),
    ],
)
def test_ensemble_ori(x, dim, osi, expected):
    if osi is not None:
        x["osi"] = torch.tensor(osi)
    else:
        del x["osi"]
    output = x.apply(perturbation.ensemble_ori, dim=dim, keepdim=True)
    expected = periodic.as_tensor(
        torch.tensor(expected).float(),
        w_dims=[0],
        extents=[(-180.0, 180.0)],
    )
    torch.testing.assert_close(output, expected, equal_nan=True)


@pytest.mark.parametrize(
    "dim, osi, expected",
    [
        (0, None, [[1.0, 1.0, sqrt2 / 2, 1.0, 1.0]]),
        (1, None, [[sqrt5 / 3], [sqrt5 / 3]]),
        (
            0,
            [[0.0], [1.0]],
            [[1.0, np.nan, 0.5, 1.0, np.nan]],
        ),
        (1, [[0.0, 1.0, 1.0, 0.0, 0.0]], [[sqrt2 / 3], [1 / 3]]),
    ],
)
def test_ensemble_osi(x, dim, osi, expected):
    if osi is not None:
        x["osi"] = torch.tensor(osi)
    else:
        del x["osi"]
    output = x.apply(perturbation.ensemble_osi, dim=dim, keepdim=True)
    expected = torch.tensor(expected).float()
    torch.testing.assert_close(output, expected, equal_nan=True)


@pytest.mark.parametrize(
    "P, error",
    [
        (5, None),
        (1000, ValueError),
    ],
)
def test_categorical_sample(P, error):
    with random.set_seed(0):
        prob = torch.rand(20, 5)
        prob[prob < 1.0e-1] = 0.0

        with pytest.raises(error) if error is not None else contextlib.nullcontext():
            h = torch.stack(
                [perturbation.categorical_sample(prob, P) for _ in range(10000)]
            )

    if error is None:
        assert (h.count_nonzero(dim=(1, 2)) == P).all()

        mean = h.sum(dim=0) / h.sum()
        sem = h.sum(dim=0) ** 0.5 / h.sum()
        expected = prob / prob.sum()

        isclose = (mean - expected).abs() <= 3.5 * sem + 1.0e-5
        assert isclose.all()


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("var", ["space", "ori"])
def test_sample(affine, var):
    x = frame.ParameterFrame(
        {
            "cell_type": categorical.as_tensor(
                [1, 1, 1, 1, 1, 1, 1, 0], categories=["PV", "PYR"]
            ),
            "space": periodic.linspace(-400.0, 400.0, 8),
            "ori": periodic.linspace(-90.0, 90.0, 8),
        }
    )
    with random.set_seed(0):
        dh = [
            perturbation.sample(
                x,
                2,
                affine and (var == "space"),
                affine and (var == "ori"),
                cell_probs={"PYR": 1.0},
                space=("uniform", 350) if var == "space" else ("uniform", torch.inf),
                ori=("uniform", 80) if var == "ori" else ("von_mises", 0.0),
            )
            for _ in range(100)
        ]
    dh = torch.stack(dh)

    assert dh.shape == (100, 8)

    for dhi in dh:
        assert dhi.count_nonzero() == 2
        indices = set(dhi.nonzero().squeeze().tolist())
        assert 7 not in indices  # no PV cells in the sample

        if not affine:
            assert indices.issubset({3, 4, 5})
        else:
            diff = max(indices) - min(indices)
            assert min(diff, 8 - diff) < 4
    if affine:
        assert len(set(dh.nonzero()[:, 1].tolist()) - {3, 4, 5}) > 0


@pytest.mark.parametrize("same_cell_type", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_convolve(x, same_cell_type, normalize):
    x["cell_type"] = categorical.as_tensor(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
        ],
        categories=["PYR", "PV"],
    )
    x["dh"] = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    if same_cell_type:
        expected0 = torch.tensor(
            [
                [0.0, exp(-5 / 2), 0.0, 0.0, 0.0],
                [1.0, 0.0, exp(-25 / 2), 0.0, 0.0],
            ]
        )
    else:
        expected0 = torch.tensor(
            [
                [exp(-1 / 2), exp(-5 / 2), exp(-26 / 2), exp(-10 / 2), exp(-1 / 2)],
                [1.0, exp(-4 / 2), exp(-25 / 2), exp(-9 / 2), 1.0],
            ]
        )
    if normalize:
        expected0 = expected0 / expected0.sum()
    expected0 = expected0 * 2
    if same_cell_type:
        expected1 = torch.tensor(
            [
                [exp(-25 / 2), 0.0, 1.0, exp(-4 / 2), exp(-25 / 2)],
                [0.0, exp(-10 / 2), 0.0, exp(-5 / 2), exp(-26 / 2)],
            ]
        )
    else:
        expected1 = torch.tensor(
            [
                [exp(-25 / 2), exp(-9 / 2), 1.0, exp(-4 / 2), exp(-25 / 2)],
                [exp(-26 / 2), exp(-10 / 2), exp(-1 / 2), exp(-5 / 2), exp(-26 / 2)],
            ]
        )
    if normalize:
        expected1 = expected1 / expected1.sum()
    expected = expected0 + expected1
    out = perturbation.convolve(x, 1.0, same_cell_type, normalize)
    torch.testing.assert_close(out, expected)


def test_perturbation_prob_eval(x):
    expr = (
        "special.uniform(extents, dspace).prod(dim=-1) * "
        "special.von_mises(torch.exp(k0 + k1 * osi + k2 * osi**2), dori)"
    )
    out = perturbation.perturbation_prob_eval(
        x,
        expr,
        cell_probs={"PV": 0.0, "PYR": 1.0},
        space_loc=torch.tensor([0.0, -5.0]),
        ori_loc=torch.tensor([-90.0]),
        extents=[3, 5],
        k0=-4.6,
        k1=11,
        k2=-6.1,
    )
    expected = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 1, 0]]).float()
    expected *= torch.tensor([[1, 1, 0, 0, 1], [1, 1, 0, 0, 1]])
    kappa = torch.exp(-4.6 + 11 * x["osi"] - 6.1 * x["osi"] ** 2)
    dori = torch.tensor(
        [[hpi, hpi, torch.pi, hpi, torch.pi], [torch.pi, torch.pi, hpi, hpi, hpi]]
    )
    expected *= special.von_mises(kappa, dori)

    torch.testing.assert_close(out, expected)


def test_perturbation_prob_eval_2(x):
    expr = "mask.float()"
    x["mask"] = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 0, 0, 1]]).bool()
    out = perturbation.perturbation_prob_eval(x, expr)
    expected = x["mask"].float()

    torch.testing.assert_close(out, expected)
