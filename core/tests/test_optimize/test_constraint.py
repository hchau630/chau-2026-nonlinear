import math

import numpy as np
import pytest
import torch

from niarb import neurons, nn, optimize, perturbation
from niarb.nn.modules.frame import ParameterFrame


@pytest.fixture
def state_dict():
    return {
        "gW": torch.tensor(
            [
                [0.1, 0.05],
                [0.2, 0.15],
            ]
        ),
        "sigma": torch.tensor(
            [
                [150.0, 220.0],
                [200.0, 175.0],
            ]
        ),
        "kappa": torch.tensor(
            [
                [0.5, 0.25],
                [0.3, 0.15],
            ]
        ),
    }


@pytest.fixture
def state_dict2():
    return {
        "gW": torch.tensor(
            [
                [1.1665e01, 1.9297e01, 2.0000e01, 9.9741e-01],
                [1.2298e01, 1.9750e01, 2.0000e01, 9.3013e-01],
                [8.5061e00, 1.4596e00, 1.0000e00, 1.0938e01],
                [1.2099e01, 1.8378e01, 5.8327e00, 2.4508e-03],
            ]
        ),
        "kappa": torch.tensor(
            [
                [0.2219, -0.0211, 0.0831, -0.4919],
                [-0.1000, -0.0729, 0.5000, -0.4660],
                [0.5000, -0.5000, -0.5000, -0.0964],
                [-0.4974, 0.0547, -0.2659, 0.5000],
            ]
        ),
    }


@pytest.mark.parametrize(
    "exclude, positive, expected",
    [
        (None, True, -3.1),
        (None, False, 2.9),
        (("PV", "SST"), True, -2.1),
        (("PV", "SST"), False, 1.9),
    ],
)
def test_determinant_con(exclude, positive, expected):
    state_dict = {
        "gW": torch.tensor(
            [
                [2.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        * 2,
    }
    con = optimize.DeterminantCon(eps=0.1, exclude=exclude, positive=positive)
    model = nn.V1(["cell_type"], cell_types=["PYR", "PV", "SST"], f="SSN")
    nn.load_state_dict(model, state_dict, strict=False)
    result = con(model)
    torch.testing.assert_close(result, torch.tensor(expected))


def test_linear_response_ori_con(state_dict):
    theta = 60.0
    cell_types = ["PV"]
    con = optimize.LinearResponseOriCon(theta, eps=0.0, cell_types=cell_types)

    variables = ["cell_type", "space", "ori"]
    cell_types = ["PYR", "PV"]
    N_space = 151
    N_ori = 18
    x = neurons.as_grid(
        n=len(cell_types),
        N_space=(N_space,),
        N_ori=N_ori,
        cell_types=cell_types,
        space_extent=[3000],
    )  # (n, N_space, N_ori)
    x = ParameterFrame(
        {k: v[..., 9:16:3, :] if k == "ori" else v for k, v in x._items()},
        ndim=x.ndim,
    )  # (n, N_space, 3)
    torch.testing.assert_close(x.data["ori"].squeeze(), torch.tensor([0.0, 30.0, 60.0]))
    x["dh"] = torch.zeros(x.shape)
    x["dh"][0, (N_space - 1) // 2, 0] = 1.0

    model = nn.V1(
        variables,
        cell_types=cell_types,
        f="SSN",
        vf_symmetry=False,
        init_vf=[0.5, 1.0],
        mode="linear_approx",
    )
    nn.load_state_dict(model, state_dict, strict=False)

    out = con(model)
    out = out * x.data["ori"].period.item() / N_ori
    expected = model(x, ndim=x.ndim, to_dataframe=False)["dr"]
    expected = expected[1, :, 1:].sum(dim=0)

    torch.testing.assert_close(out, expected.min())


def test_linear_response_ori_con_2(state_dict):
    theta = 60.0
    cell_types = ["PV"]
    con = optimize.LinearResponseOriCon(theta, eps=0.0, cell_types=cell_types)

    variables = ["cell_type", "space", "ori", "osi"]
    cell_types = ["PYR", "PV"]
    osi_prob = torch.distributions.Beta(
        torch.tensor([1.5, 2.0]), torch.tensor([2.0, 4.0])
    )
    N_space = 151
    N_ori = 18
    N_osi = 100
    x = neurons.as_grid(
        n=len(cell_types),
        N_space=(N_space,),
        N_ori=N_ori,
        N_osi=N_osi,
        cell_types=cell_types,
        space_extent=[3000],
        osi_prob=osi_prob,
    )  # (n, N_space, N_ori, N_osi)
    x = ParameterFrame(
        {k: v[..., 9:16:3, :, :] if k == "ori" else v for k, v in x._items()},
        ndim=x.ndim,
    )  # (n, N_space, 3, N_osi)
    torch.testing.assert_close(x.data["ori"].squeeze(), torch.tensor([0.0, 30.0, 60.0]))
    x = x.unsqueeze(0)  # (1, n, N_space, 3, N_osi)
    x["dh"] = torch.zeros((N_osi, *x.shape[1:]))
    for i in range(N_osi):
        x["dh"][i, 0, (N_space - 1) // 2, 0, i] = 1.0

    model = nn.V1(
        variables,
        cell_types=cell_types,
        osi_func=2.0,
        osi_prob=osi_prob,
        f="SSN",
        vf_symmetry=False,
        init_vf=[0.5, 1.0],
        mode="linear_approx",
    )
    nn.load_state_dict(model, state_dict, strict=False)

    out = con(model)
    out = out * x.data["ori"].period.item() / N_ori
    expected = model(x, ndim=(x.ndim - 1), to_dataframe=False)["dr"]
    expected = expected[:, 1, :, 1:, :].sum(dim=(1, 3)).mean(dim=0)

    torch.testing.assert_close(out, expected.min())


@pytest.mark.parametrize("ct", [("PV", "PYR")])
def test_linear_response_ori_analytic_con(state_dict2, ct):
    con = optimize.LinearResponseOriAnalyticCon(
        positive=True, post=ct[0], pre=ct[1], eps=0.0
    )

    variables = ["cell_type", "ori", "osi"]
    cell_types = ["PYR", "PV", "SST", "VIP"]
    N_ori = 2
    N_osi = 3
    x = neurons.as_grid(
        n=len(cell_types),
        N_ori=N_ori,
        N_osi=N_osi,
        cell_types=cell_types,
    )  # (n, N_ori, N_osi)
    x["dh"] = torch.zeros(x.shape)
    x["dh"][cell_types.index(ct[1]), 0, -1] = 1.0

    model = nn.V1(variables, cell_types=cell_types, osi_func=0.25)
    nn.load_state_dict(model, state_dict2, strict=False)

    out = con(model)
    out = out * 4 / (N_ori * N_osi)
    expected = model(x, ndim=x.ndim, to_dataframe=False)["dr"]
    expected = expected[cell_types.index(ct[0]), :, -1]
    expected = expected[0] - expected[1]

    torch.testing.assert_close(out, expected, rtol=5e-3, atol=0)


@pytest.mark.parametrize("ct", [("PYR", "PV"), ("PV", "PYR")])
def test_linear_response_space_2d_con(state_dict, ct):
    a, b = 400.0, 500.0
    min_r = 15.0
    con = optimize.LinearResponseSpace2dCon(
        a, b, min_r=min_r, dr=1.0, eps=0.0, cell_types=ct[1:], perturbed_cell_type=ct[0]
    )

    variables = ["cell_type", "space", "ori"]
    cell_types = ["PYR", "PV"]
    N_space = (41, 51)
    N_ori = 4
    x = neurons.as_grid(
        n=len(cell_types),
        N_space=N_space,
        N_ori=N_ori,
        cell_types=cell_types,
        space_extent=[a, b],
    )  # (n, *N_space, N_ori)
    x["space"] = x.data["space"].tensor
    x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, n, *N_space, N_ori)
    x["dh"] = torch.zeros((*N_space, *x.shape[2:]))
    for i, j in np.ndindex(N_space):
        x["dh"][i, j, cell_types.index(ct[0]), i, j, 0] = 1.0
    x["distance"] = x.apply(perturbation.min_distance_to_ensemble, dim=range(2, x.ndim))
    mask = (x["distance"] >= min_r).any(dim=(2, -1))  # (*N_space, *N_space)

    model = nn.V1(
        variables,
        cell_types=cell_types,
        f="SSN",
        vf_symmetry=False,
        init_vf=[0.5, 0.5],
        mode="linear_approx",
    )
    nn.load_state_dict(model, state_dict, strict=False)

    out = con(model)
    out = out * x.data["dV"].item() / 180 * math.prod(N_space) ** 2
    expected = model(x, ndim=(x.ndim - 2), to_dataframe=False)["dr"]
    expected = expected[:, :, cell_types.index(ct[1]), ...][mask].mean(dim=-1).sum()

    torch.testing.assert_close(out, expected, rtol=5e-3, atol=0)


def test_linear_response_space_2d_ori_con(state_dict):
    a, b = 400.0, 500.0
    min_r = 15.0
    theta = 60.0
    cell_types = ["PV"]
    con = optimize.LinearResponseSpace2dOriCon(
        a, b, theta=theta, min_r=min_r, dr=1.0, eps=0.0, cell_types=cell_types
    )

    variables = ["cell_type", "space", "ori"]
    cell_types = ["PYR", "PV"]
    N_space = (41, 51)
    N_ori = 18
    x = neurons.as_grid(
        n=len(cell_types),
        N_space=N_space,
        N_ori=N_ori,
        cell_types=cell_types,
        space_extent=[a, b],
    )  # (n, *N_space, N_ori)
    x = ParameterFrame(
        {k: v[..., 9:16:3, :] if k == "ori" else v for k, v in x._items()},
        ndim=x.ndim,
    )  # (n, *N_space, 3)
    x["space"] = x.data["space"].tensor
    torch.testing.assert_close(x.data["ori"].squeeze(), torch.tensor([0.0, 30.0, 60.0]))
    x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, n, *N_space, 3)
    x["dh"] = torch.zeros((*N_space, *x.shape[2:]))
    for i, j in np.ndindex(N_space):
        x["dh"][i, j, 0, i, j, 0] = 1.0
    x["distance"] = x.apply(perturbation.min_distance_to_ensemble, dim=range(2, x.ndim))
    mask = (x["distance"] >= min_r).any(dim=(2, -1))  # (*N_space, *N_space)

    model = nn.V1(
        variables,
        cell_types=cell_types,
        f="SSN",
        vf_symmetry=False,
        init_vf=[0.5, 1.0],
        mode="linear_approx",
    )
    nn.load_state_dict(model, state_dict, strict=False)

    out = con(model)
    out = out / N_ori * math.prod(N_space)
    expected = model(x, ndim=(x.ndim - 2), to_dataframe=False)["dr"]
    expected = expected[:, :, 1, :, :, 1:][mask].sum(dim=0)

    torch.testing.assert_close(out, expected.min(), rtol=5e-3, atol=0)
