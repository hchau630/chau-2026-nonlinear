import copy
import math
from collections.abc import Sequence
from itertools import product
from math import exp

import hyclib as lib
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from niarb import linalg, neurons, nn, numerics, perturbation, random, special, utils
from niarb.cell_type import CellType
from niarb.dataset import Dataset, collate_fn
from niarb.nn import functional
from niarb.nn.modules import frame
from niarb.nn.modules.v1 import _cdim, compute_osi_scale
from niarb.tensors import circulant, periodic
from niarb.tensors.circulant import CirculantTensor


def spatial_component(d, s, r):
    if d == 1:
        return torch.exp(-r / s) / (2 * s)
    elif d == 2:
        return special.k0(r / s) / (2 * torch.pi * s**2)
    elif d == 3:
        return torch.exp(-r / s) / r / (4 * torch.pi * s**2)
    else:
        raise NotImplementedError()


def get_parameters(**kwargs):
    parameters = []
    for p in product(*kwargs.values()):
        p = dict(zip(kwargs.keys(), p))

        if "cell_type" not in p.get("variables", []):
            if p.get("f") == "Match":
                continue

            if p.get("subcircuit_cell_types"):
                continue

            if "osi_prob" in p and isinstance(p["osi_prob"][1], list):
                continue

        if (
            "ori" not in p.get("variables", [])
            and "ori_func" in p
            and p["ori_func"] == "von_mises"
        ):
            continue

        if "osi" not in p.get("variables", []):
            if "osi_func" in p and p["osi_func"] != ("Identity",):
                continue

            if "osi_prob" in p and p["osi_prob"] != ("Uniform", 0.0, 1.0):
                continue

        if (
            "d" in p
            and "space" in p.get("variables", [])
            and ("ori" in p["variables"] or "osi" in p["variables"])
            and p["d"] > 1
        ):
            continue

        if "d" in p and p["d"] > 1 and "space" not in p.get("variables", []):
            continue

        if (
            not {"cell_type", "space"}.issubset(p.get("variables", []))
            and p.get("sigma_symmetry") != "pre"
        ):
            continue

        parameters.append(list(p.values()))
    return parameters


def get_state_dict(n, sigma_symmetry="pre", vf_symmetry=True, von_mises=False):
    sigma = torch.tensor([[150.0, 220.0], [200.0, 175.0]])

    if n == 1:
        state_dict = {
            "gW": torch.tensor([[0.1]]),
            "sigma": sigma[:1, :1],
            "kappa": torch.tensor([[3.0 if von_mises else 0.3]]),
        }
    elif n == 2:
        if sigma_symmetry == "pre":
            sigma = sigma[:1, :]
        elif sigma_symmetry == "post":
            sigma = sigma[:, :1]
        elif sigma_symmetry == "full":
            sigma = sigma[:1, :1]
        elif isinstance(sigma_symmetry, Sequence) and not isinstance(
            sigma_symmetry, str
        ):
            sigma = sigma[:, 0]
        if von_mises:
            kappa = torch.tensor([[3.0, -2.5], [-2.0, 1.5]])
        else:
            kappa = torch.tensor([[0.5, 0.25], [0.3, 0.15]])
        state_dict = {
            "gW": torch.tensor(
                [
                    [0.1, 0.05],
                    [0.2, 0.15],
                ]
            ),
            "sigma": sigma,
            "kappa": kappa,
        }
    else:
        raise NotImplementedError()

    if not vf_symmetry:
        if n == 2:
            state_dict["vf"] = torch.tensor([0.8, 1.2])
        elif n != 1:
            raise NotImplementedError()
    return state_dict


class TestV1:
    @pytest.mark.parametrize(
        "sigma_symmetry", [[[0, 1], [1, 0]], "pre", "post", "full", None]
    )
    def test_sigma(self, sigma_symmetry):
        model = nn.V1(
            ["cell_type", "space"],
            cell_types=["PYR", "PV"],
            sigma_symmetry=sigma_symmetry,
        )
        sigma, S = model.sigma, model.S
        assert S.shape == (2, 2)
        if sigma_symmetry == "pre":
            assert sigma.shape == (1, 2)
        elif sigma_symmetry == "post":
            assert sigma.shape == (2, 1)
        elif sigma_symmetry == "full":
            assert sigma.shape == (1, 1)
        elif sigma_symmetry is None:
            assert sigma.shape == (2, 2)
        else:
            assert sigma.shape == (2,)
            assert S[0, 0] == S[1, 1]
            assert S[1, 0] == S[0, 1]
            assert S[0, 0] != S[0, 1]
        assert (S == model.sigma_**2).all()

    @pytest.mark.parametrize(
        "cell_types, null_connections, null_indices",
        [
            (["PYR", "PV"], [], []),
            (["PYR", "PV", "SST"], [("SST", "SST")], [(2, 2)]),
            (
                ["PYR", "PV", "SST", "VIP"],
                [("SST", "SST"), ("VIP", "VIP")],
                [(2, 2), (3, 3)],
            ),
        ],
    )
    def test_null_connections(self, cell_types, null_connections, null_indices):
        cell_types = [getattr(CellType, ct) for ct in cell_types]
        if null_connections is not None:
            null_connections = [
                (getattr(CellType, cti), getattr(CellType, ctj))
                for cti, ctj in null_connections
            ]

        model = nn.V1(
            ["cell_type", "space"],
            cell_types=cell_types,
            sigma_symmetry="pre",
            null_connections=null_connections,
        )

        if len(null_indices) > 0:
            null_indices = tuple(zip(*null_indices))
            assert (model.gW[null_indices] == 0.0).all()
            assert (model.gW.requires_optim[null_indices] == False).all()
        else:
            assert (model.gW > 0.0).all()

    @pytest.mark.parametrize("d", [1, 2, 3])
    @pytest.mark.parametrize(
        "variables, osi_func, osi_prob, sigma_symmetry",
        get_parameters(
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
                ["cell_type", "space", "ori"],
                ["cell_type", "ori", "osi"],
                ["cell_type", "space", "ori", "osi"],
            ],
            osi_func=[("Identity",), ("Pow", (0.5,))],
            osi_prob=[
                ("Uniform", 0.0, 1.0),
                ("Beta", [2.0, 1.5], [3.0, 2.5]),
            ],
            sigma_symmetry=[[[0, 1], [1, 0]], "pre", "post", "full", None],
        ),
    )
    def test_weights(self, variables, d, osi_func, osi_prob, sigma_symmetry):
        model = nn.V1(
            variables,
            cell_types=(CellType.PYR, CellType.PV),
            osi_func=osi_func,
            osi_prob=osi_prob,
            f=(
                nn.Match({"PV": nn.SSN(3)}, nn.SSN(2))
                if "cell_type" in variables
                else nn.SSN(2)
            ),
            sigma_symmetry=sigma_symmetry,
        )
        state_dict = get_state_dict(model.n, sigma_symmetry)
        model.load_state_dict(state_dict, strict=False)

        if osi_func is None:
            osi_func = lambda x: x

        N_space, N_ori, N_osi = [5] * d, 4, 4
        x = neurons.as_grid(
            n=(model.n if "cell_type" in variables else 0),
            N_space=(N_space if "space" in variables else ()),
            N_ori=(N_ori if "ori" in variables else 0),
            N_osi=(N_osi if "osi" in variables else 0),
            space_extent=[1000.0] * d,
            ori_extent=(-90.0, 90.0),
            osi_prob=osi_prob,
        )

        with torch.no_grad():
            out = model(x, output="weight", ndim=x.ndim, to_dataframe=False)

        out = out.dense(keep_shape=False) if isinstance(out, CirculantTensor) else out
        x = x.reshape(-1)

        G = torch.atleast_1d(model.gain()).diag()  # (n, n)
        W = torch.linalg.inv(G) @ model.gW  # (n, n)
        sigma = model.S.clone() ** 0.5  # (n, n)
        kappa = model.kappa.clone()  # (n, n)

        if "cell_type" in variables:
            W[:, 1] = -W[:, 1]
            W = W[x["cell_type"], :][:, x["cell_type"]]  # (M, M)
            sigma = sigma[x["cell_type"], :][:, x["cell_type"]]  # (M, M)
            kappa = kappa[x["cell_type"], :][:, x["cell_type"]]  # (M, M)

        if "osi" in variables:
            osi_func = utils.call(nn, osi_func)
            kappa = (
                kappa * osi_func(x["osi"][:, None]) * osi_func(x["osi"][None, :])
            )  # (M, M)

        expected = W

        if "space" in variables:
            r = functional.diff(x["space"][:, None], x["space"][None, :])
            r = r.norm(dim=-1)  # (M, M)
            expected = (
                expected * spatial_component(d, sigma, r) * 1000**d / math.prod(N_space)
            )
            if d > 1:
                expected[r == 0] = 0.0

        if "ori" in variables:
            theta = functional.diff(x["ori"][:, None], x["ori"][None, :])
            theta = theta.tensor.squeeze(-1) / 90.0 * torch.pi  # (M, M)
            expected = expected * (1 + 2 * kappa * torch.cos(theta)) / N_ori

        if "osi" in variables:
            expected = expected / N_osi

        torch.testing.assert_close(out, expected)

    def test_weights_sparse(self):
        prob_kernel = nn.Gaussian(
            nn.Matrix([[100.0, 200.0], [200.0, 100.0]], "cell_type"), "space"
        )
        with random.set_seed(0):
            model = nn.V1(
                ["cell_type", "space"],
                cell_types=(CellType.PYR, CellType.PV),
                prob_kernel={"space": prob_kernel},
            )
        x = neurons.as_grid(
            2, (4,), cell_types=(CellType.PYR, CellType.PV), space_extent=(400,)
        )
        M = 50000
        x_ = frame.stack([x] * M)
        with random.set_seed(0):
            out = model(x_, output="weight", ndim=2, to_dataframe=False)

        prob1 = torch.tensor(
            [
                [1.0, exp(-0.5), exp(-2.0), exp(-0.5)],
                [exp(-0.5), 1.0, exp(-0.5), exp(-2.0)],
                [exp(-2.0), exp(-0.5), 1.0, exp(-0.5)],
                [exp(-0.5), exp(-2.0), exp(-0.5), 1.0],
            ]
        )
        prob2 = torch.tensor(
            [
                [1.0, exp(-0.125), exp(-0.5), exp(-0.125)],
                [exp(-0.125), 1.0, exp(-0.125), exp(-0.5)],
                [exp(-0.5), exp(-0.125), 1.0, exp(-0.125)],
                [exp(-0.125), exp(-0.5), exp(-0.125), 1.0],
            ]
        )
        prob = torch.stack([torch.stack([prob1, prob2]), torch.stack([prob2, prob1])])
        prob = prob.movedim(1, 2)
        sem = (prob * (1 - prob) / M).sqrt()
        out_prob = out.count_nonzero(dim=0) / M
        assert (out_prob >= prob - 2.5 * sem).all()
        assert (out_prob <= prob + 2.5 * sem).all()

        model.prob_kernel = nn.Prod([])
        expected = model(x, output="weight", ndim=2, to_dataframe=False).dense()
        torch.testing.assert_close(out.mean(dim=0), expected, atol=1e-5, rtol=5e-2)

    def test_weights_von_mises(self):
        model = nn.V1(
            ["cell_type", "ori"],
            cell_types=(CellType.PYR, CellType.PV),
            ori_func="von_mises",
        )
        state_dict = get_state_dict(model.n, sigma_symmetry=None, von_mises=True)
        model.load_state_dict(state_dict, strict=False)

        expected_model = nn.V1(
            ["cell_type", "ori"],
            cell_types=(CellType.PYR, CellType.PV),
            ori_func="von_mises",
            ori_order=16,
        )
        expected_model.load_state_dict(state_dict, strict=False)

        x = neurons.as_grid(n=model.n, N_ori=50, ori_extent=(-90.0, 90.0))

        with torch.no_grad():
            out = model(x, output="weight", ndim=x.ndim, to_dataframe=False)
            expected = expected_model(
                x, output="weight", ndim=x.ndim, to_dataframe=False
            )

        out = out.dense(keep_shape=False)
        expected = expected.dense(keep_shape=False)

        assert (out != expected).any()  # check that the two methods are different
        torch.testing.assert_close(out, expected)

    @pytest.mark.parametrize(
        "f, variables, d, ori_func, osi_func, osi_prob, sigma_symmetry, mode",
        get_parameters(
            f=[("Identity",)],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
                ["cell_type", "space", "ori"],
                ["cell_type", "ori", "osi"],
                ["cell_type", "space", "ori", "osi"],
            ],
            d=[1, 3],
            ori_func=["cosine", "von_mises"],
            osi_func=[("Identity",), 0.5],
            osi_prob=[
                ("Uniform", 0.0, 1.0),
                ("Beta", [2.0, 1.5], [1.0, 1.0]),
            ],
            sigma_symmetry=[[[0, 1], [1, 0]], "pre", "post", "full", None],
            mode=["analytical"],
        )
        + get_parameters(
            f=[("SSN", (2,)), "Match"],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["ori", "osi"],
            ],
            d=[1],
            ori_func=["cosine", "von_mises"],
            osi_func=[("Identity",)],
            osi_prob=[("Uniform", 0.0, 1.0)],
            sigma_symmetry=["pre"],
            mode=["analytical", "broyden1"],
        )
        + get_parameters(
            f=[("SSN", (2,)), "Match"],
            variables=[["cell_type"]],
            d=[1],
            ori_func=["cosine"],
            osi_func=[("Identity",)],
            osi_prob=[("Uniform", 0.0, 1.0)],
            sigma_symmetry=["pre"],
            mode=["matrix", "newton"],
        ),
    )
    @pytest.mark.parametrize("vf_symmetry", [True, False])
    def test_forward(
        self,
        f,
        variables,
        d,
        ori_func,
        osi_func,
        osi_prob,
        sigma_symmetry,
        vf_symmetry,
        mode,
    ):
        if f == "Match":
            f = nn.Match({"PV": nn.SSN()}, nn.Identity())

        model = nn.V1(
            variables,
            cell_types=(CellType.PYR, CellType.PV),
            f=f,
            ori_func=ori_func,
            ori_order=(None if ori_func == "cosine" else 4),
            osi_func=osi_func,
            osi_prob=osi_prob,
            sigma_symmetry=sigma_symmetry,
            vf_symmetry=vf_symmetry,
            autapse=True,
            mode=mode,
        )
        von_mises = ori_func == "von_mises"
        state_dict = get_state_dict(model.n, sigma_symmetry, vf_symmetry, von_mises)
        model.load_state_dict(state_dict, strict=False)
        expected_model = copy.deepcopy(model)
        expected_model.mode = "matrix" if f == ("Identity",) else "numerical"

        N_ori, N_osi = 10, 30
        N_space = [350] if d == 1 else [125] * d
        space_extent = [4000.0] if d == 1 else [2000.0] * d
        x = neurons.as_grid(
            n=(model.n if "cell_type" in variables else 0),
            N_space=(N_space if "space" in variables else ()),
            N_ori=(N_ori if "ori" in variables else 0),
            N_osi=(N_osi if "osi" in variables else 0),
            space_extent=space_extent,
            ori_extent=(-90.0, 90.0),
            osi_prob=osi_prob,
        )

        prob = torch.ones(x.shape)

        if "cell_type" in variables:
            prob = prob * (x["cell_type"] == "PYR").float()

        if "osi" in variables:
            prob = prob * (x["osi"] > 0.3).float()

        x["dh"] = torch.zeros(x.shape)
        x["dh"][tuple(prob.nonzero()[0])] = 1.0

        # check unperturbed cells
        x["mask"] = torch.ones(x.shape, dtype=bool)
        x["mask"][x["dh"] != 0] = False

        if d == 1 or variables == ["space"]:
            model.to(torch.double)
            expected_model.to(torch.double)
            x.to(torch.double)

        with torch.no_grad():
            out = model(x, ndim=x.ndim)["dr"]
            expected = expected_model(x, ndim=x.ndim)["dr"]

        if variables != ["cell_type"]:
            assert (out != expected).any()  # check that the two methods are different

        torch.testing.assert_close(
            out, expected, rtol=1.3e-6, atol=expected.abs().max().item() * 2e-4
        )

        # check perturbed cells
        x["mask"] = torch.ones(x.shape, dtype=bool)
        x["mask"][x["dh"] == 0] = False

        with torch.no_grad():
            out = model(x, ndim=x.ndim)["dr"].float()
            expected = expected_model(x, ndim=x.ndim)["dr"].float()

        torch.testing.assert_close(out, expected)

    def test_forward_von_mises(self):
        model = nn.V1(
            ["cell_type", "space", "ori", "osi"],
            cell_types=(CellType.PYR, CellType.PV),
            ori_func="von_mises",
            ori_order=8,
            sigma_symmetry=None,
            autapse=True,
        )
        state_dict = get_state_dict(model.n, sigma_symmetry=None, von_mises=True)
        model.load_state_dict(state_dict, strict=False)
        expected_model = nn.V1(
            ["cell_type", "space", "ori", "osi"],
            cell_types=(CellType.PYR, CellType.PV),
            ori_func="von_mises",
            sigma_symmetry=None,
            autapse=True,
            mode="matrix",
        )
        expected_model.load_state_dict(state_dict, strict=False)

        x = neurons.as_grid(
            n=model.n,
            N_space=[350],
            N_ori=10,
            N_osi=30,
            space_extent=[4000.0],
            ori_extent=(-90.0, 90.0),
        )

        prob = torch.ones(x.shape)
        prob = prob * (x["cell_type"] == "PYR").float()
        prob = prob * (x["osi"] > 0.3).float()

        x["dh"] = torch.zeros(x.shape)
        x["dh"][tuple(prob.nonzero()[0])] = 1.0

        # check unperturbed cells
        x["mask"] = torch.ones(x.shape, dtype=bool)
        x["mask"][x["dh"] != 0] = False

        model.to(torch.double)
        expected_model.to(torch.double)
        x.to(torch.double)

        with torch.no_grad():
            out = model(x, ndim=x.ndim)["dr"]
            expected = expected_model(x, ndim=x.ndim)["dr"]

        assert (out != expected).any()  # check that the two methods are different

        torch.testing.assert_close(
            out, expected, rtol=1.3e-6, atol=expected.abs().max().item() * 2e-4
        )

        # check perturbed cells
        x["mask"] = torch.ones(x.shape, dtype=bool)
        x["mask"][x["dh"] == 0] = False

        with torch.no_grad():
            out = model(x, ndim=x.ndim)["dr"].float()
            expected = expected_model(x, ndim=x.ndim)["dr"].float()

        torch.testing.assert_close(out, expected)

    def test_forward_sample(self):
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)
        model = nn.V1(variables, cell_types=cell_types)
        state_dict = get_state_dict(model.n, sigma_symmetry=None)
        model.load_state_dict(state_dict, strict=False)
        expected_model = copy.deepcopy(model)
        expected_model.mode = "matrix"

        data = pd.DataFrame(
            {
                "distance": pd.Categorical.from_codes(
                    list(range(10)) + list(range(0, 10, 2)),
                    categories=pd.interval_range(
                        start=10, end=300, periods=10, closed="left"
                    ),
                ),
                "cell_type": ["PYR"] * 10 + ["PV"] * 5,
            }
        )

        model = nn.Pipeline(model=model, data=[data])
        expected_model = nn.Pipeline(model=expected_model, data=[data])

        N_stim, N, space_extent = 15, 3000, 2000.0
        dataset = Dataset(
            neurons={
                "N": N,
                "variables": variables,
                "cell_types": cell_types,
                "space_extent": [space_extent],
            },
            perturbations={"configs": [{"N": 2, "cell_probs": {"PYR": 1.0}}]},
            metrics={"distance": "min_distance_to_ensemble"},
            N_instantiations=N_stim,
            seed=0,
        )

        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
        x, kwargs = next(iter(dataloader))
        assert x.shape == (N_stim, 1, N)

        model.to(torch.double)
        expected_model.to(torch.double)
        x.to(torch.double)

        with torch.no_grad():
            out = model(x, **kwargs)
            expected = expected_model(x, **kwargs)

        assert (out != expected).any()  # check that the two methods are different

        # print(out)
        # print(expected)
        # print((out - expected) / expected)
        torch.testing.assert_close(out, expected, rtol=5e-3, atol=0)

    @pytest.mark.parametrize(
        "mode",
        [
            "linear_approx",
            "quasi_linear_approx",
            "second_order_approx",
            "second_order_approx_naive",
        ],
    )
    @pytest.mark.parametrize("matrix_mode", [True, False])
    @pytest.mark.parametrize("f", ["SSN", "Ricciardi", "Match"])
    def test_forward_approx(self, f, mode, matrix_mode):
        # note that when mode == "linear_approx" or "quasi_linear_approx",
        # or f == "Match", the is_circulant == False branch is tested, so
        # we don't need to write another test with neurons not on a grid
        if f == "Match":
            f = nn.Match({"PYR": nn.SSN(2)}, nn.Rectified())
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)
        _mode = f"matrix_{mode}" if matrix_mode else mode
        model = nn.V1(variables, cell_types=cell_types, f=f, init_vf=1.2, mode=_mode)
        state_dict = get_state_dict(model.n, sigma_symmetry=None)
        model.load_state_dict(state_dict, strict=False)
        expected_model = copy.deepcopy(model)
        expected_model.mode = "numerical"

        x = neurons.as_grid(n=model.n, N_space=[1000], space_extent=[4000.0])

        x["dh"] = torch.zeros(x.shape)
        indices = (
            torch.arange(0, 1000, 10) if mode.startswith("second_order_approx") else 0
        )
        value_dict = {"linear_approx": 0.1, "second_order_approx_naive": 0.5}
        x["dh"][0, indices] = value_dict.get(mode, 5.0)
        if f == "Ricciardi" and mode in {"quasi_linear_approx", "second_order_approx"}:
            # since Ricciardi is less 'nonlinear' than SSN, we can test with a larger perturbation
            x["dh"] = x["dh"] * 10

        # check unperturbed cells
        x["mask"] = torch.ones(x.shape, dtype=bool)
        x["mask"][x["dh"] != 0] = False

        model.to(torch.double)
        expected_model.to(torch.double)
        x.to(torch.double)

        # matrix approx modes does not support circulant weights
        if matrix_mode:
            x = x.reshape(-1)
        with torch.no_grad():
            out = model(x, ndim=x.ndim)["dr"]
            expected = expected_model(x, ndim=x.ndim)["dr"]

        assert (out != expected).any()  # check that the two methods are different

        torch.testing.assert_close(
            out, expected, rtol=0.05, atol=expected.abs().max().item() * 1e-4
        )

    @pytest.mark.parametrize("f", ["Identity", "Ricciardi"])
    @pytest.mark.parametrize("mode", ["analytical", "numerical"])
    @pytest.mark.parametrize("masked", [True, False])
    @pytest.mark.parametrize("tau", [1.0, [1.0, 0.5]])
    def test_forward_double(self, f, mode, masked, tau):
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)

        model = nn.V1(
            variables,
            cell_types=cell_types,
            tau=tau,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
        )
        state_dict = get_state_dict(model.n)
        model.load_state_dict(state_dict, strict=False)

        with random.set_seed(0):
            x = neurons.sample(
                200, variables, cell_types=cell_types, space_extent=[1000.0]
            )

            x["dh"] = perturbation.sample(
                x, 10, cell_probs={"PYR": 1.0}, space=("uniform", 500.0)
            )

        if masked:
            x["mask"] = torch.ones(x.shape, dtype=bool)
            x["mask"][x["dh"] != 0] = False

        model = model.to(torch.double)
        x = x.to(torch.double)
        out = model(x)
        assert out["dr"].dtype == torch.double

    def test_forward_invalid_f(self):
        variables = ["space"]
        cell_types = (CellType.PYR, CellType.PV)
        with pytest.raises(ValueError):
            nn.V1(
                variables,
                cell_types=cell_types,
                f=nn.Match({"PV": nn.SSN(3)}, nn.SSN(2)),
            )

    @pytest.mark.parametrize("f", ["Identity", "Ricciardi", "Match"])
    @pytest.mark.parametrize(
        "mode",
        [
            "analytical",
            "linear_approx",
            "quasi_linear_approx",
            "second_order_approx",
            "numerical",
            "newton",
            "broyden1",
            "matrix",
        ],
    )
    @pytest.mark.parametrize("tau", [1.0, [1.0, 0.5]])
    @pytest.mark.parametrize(
        "variables", [["cell_type", "space"], ["cell_type", "space", "ori", "osi"]]
    )
    def test_forward_batched(self, f, mode, tau, variables):
        cell_types = (CellType.PYR, CellType.PV)

        if f == "Match":
            f = nn.Match({"PV": nn.Ricciardi(tau=0.01)}, nn.Identity())

        if f == "Identity" and mode.endswith("approx"):
            with pytest.raises(ValueError):
                model = nn.V1(variables, cell_types=cell_types, f=f, mode=mode)
            return

        model = nn.V1(
            variables,
            cell_types=cell_types,
            tau=tau,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
        )
        state_dict = get_state_dict(model.n)
        model.load_state_dict(state_dict, strict=False)

        dataset = Dataset(
            neurons={
                "N": 200,
                "variables": variables,
                "cell_types": cell_types,
                "space_extent": [1000.0],
            },
            perturbations={
                "configs": [
                    {"N": 1, "cell_probs": {"PYR": 1.0}, "space": ["uniform", 500.0]},
                    {"N": 10, "cell_probs": {"PYR": 1.0}, "space": ["uniform", 500.0]},
                    {"N": 10, "cell_probs": {"PV": 1.0}, "space": ["uniform", 500.0]},
                ]
            },
            N_instantiations=5,
            seed=0,
        )

        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
        x, kwargs = next(iter(dataloader))
        assert x.shape == (5, 3, 200)

        model = model.to(torch.double)
        x = x.to(torch.double)

        with torch.inference_mode():
            out = model(x, **kwargs["model_kwargs"]).to_pandas()

        expected = []
        for i, j in product(range(5), range(3)):
            xij = x.iloc[i, j]
            with torch.inference_mode():
                expected.append(model(xij, **kwargs["model_kwargs"]).to_pandas())
        expected = pd.concat(expected).reset_index(drop=True)

        torch.testing.assert_close(
            torch.from_numpy(out["dr"].to_numpy()),
            torch.from_numpy(expected["dr"].to_numpy()),
            rtol=1.3e-6,
            atol=1.0e-5,
        )
        out, expected = out.drop(columns="dr"), expected.drop(columns="dr")
        pd.testing.assert_frame_equal(out, expected)

    @pytest.mark.parametrize("f", ["Identity", "Ricciardi", "Match"])
    @pytest.mark.parametrize(
        "mode",
        [
            "analytical",
            "linear_approx",
            "quasi_linear_approx",
            "second_order_approx",
            "numerical",
            "newton",
            "broyden1",
            "matrix",
        ],
    )
    @pytest.mark.parametrize("masked", [True, False])
    @pytest.mark.parametrize("tau", [1.0, [1.0, 0.5]])
    @pytest.mark.parametrize(
        "variables", [["cell_type", "space"], ["cell_type", "space", "ori", "osi"]]
    )
    def test_batched_forward(self, f, mode, masked, tau, variables):
        cell_types = (CellType.PYR, CellType.PV)
        batch_shape = (4, 3)
        n = len(cell_types)

        if f == "Match":
            f = nn.Match({"PV": nn.Ricciardi(tau=0.01)}, nn.Identity())

        if f == "Identity" and mode.endswith("approx"):
            with pytest.raises(ValueError):
                model = nn.V1(variables, cell_types=cell_types, f=f, mode=mode)
            return

        model = nn.V1(
            variables,
            cell_types=cell_types,
            tau=tau,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
            init_gW_std=0.25,
        )
        batched_model = nn.V1(
            variables,
            cell_types=cell_types,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
            batch_shape=batch_shape,
        )

        with random.set_seed(0):
            x = neurons.sample(
                200, variables, cell_types=cell_types, space_extent=[1000.0]
            )

            x["dh"] = perturbation.sample(
                x, 10, cell_probs={"PYR": 1.0}, space=("uniform", 500.0)
            )

        if masked:
            x["mask"] = torch.ones(x.shape, dtype=bool)
            x["mask"][x["dh"] != 0] = False

        model.to(torch.double)
        batched_model.to(torch.double)
        x = x.to(torch.double)
        state_dict = {
            "gW": torch.empty(*batch_shape, n, n, dtype=torch.double),
            "sigma": torch.empty(*batch_shape, 1, n, dtype=torch.double),
            "kappa": torch.empty(*batch_shape, n, n, dtype=torch.double),
        }
        expected = []
        for seed, (i, j) in enumerate(product(*(range(s) for s in batch_shape))):
            with random.set_seed(seed):
                model.reset_parameters()

            for k, v in state_dict.items():
                v[i, j] = model.state_dict()[k]

            with torch.inference_mode():
                expected.append(model(x).to_pandas())
        expected = pd.concat(expected).reset_index(drop=True)

        batched_model.load_state_dict(state_dict, strict=False)
        with torch.inference_mode():
            out = batched_model(x).to_pandas()

        torch.testing.assert_close(
            torch.from_numpy(out["dr"].to_numpy()),
            torch.from_numpy(expected["dr"].to_numpy()),
            rtol=1.3e-6,
            atol=1.0e-5,
        )
        out, expected = out.drop(columns="dr"), expected.drop(columns="dr")
        pd.testing.assert_frame_equal(out, expected)

    @pytest.mark.parametrize("f", ["Identity", "Ricciardi", "Match"])
    @pytest.mark.parametrize(
        "mode", ["analytical", "numerical", "newton", "broyden1", "matrix"]
    )
    @pytest.mark.parametrize(
        "variables", [["cell_type", "space"], ["cell_type", "space", "ori", "osi"]]
    )
    def test_batched_forward_batched(self, f, mode, variables):
        if f == "Match":
            f = nn.Match({"PV": nn.Ricciardi(tau=0.01)}, nn.Identity())

        cell_types = (CellType.PYR, CellType.PV)
        batch_shape = (4,)
        n = len(cell_types)

        model = nn.V1(
            variables,
            cell_types=cell_types,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
            init_gW_std=0.25,
        )
        batched_model = nn.V1(
            variables,
            cell_types=cell_types,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
            batch_shape=batch_shape,
        )

        dataset = Dataset(
            neurons={
                "N": 200,
                "variables": variables,
                "cell_types": cell_types,
                "space_extent": [1000.0],
            },
            perturbations={
                "configs": [
                    {"N": 1, "cell_probs": {"PYR": 1.0}, "space": ["uniform", 500.0]},
                    {"N": 10, "cell_probs": {"PYR": 1.0}, "space": ["uniform", 500.0]},
                    {"N": 10, "cell_probs": {"PV": 1.0}, "space": ["uniform", 500.0]},
                    {"N": 10, "space": ["uniform", 500.0]},
                ]
            },
            seed=0,
            N_instantiations=2,
        )

        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
        x, kwargs = next(iter(dataloader))
        assert x.shape == (2, 4, 200)
        x_batch_shape = (2, 4)

        model.to(torch.double)
        batched_model.to(torch.double)
        x = x.to(torch.double)

        state_dict = {
            "gW": torch.empty(*batch_shape, n, n, dtype=torch.double),
            "sigma": torch.empty(*batch_shape, 1, n, dtype=torch.double),
            "kappa": torch.empty(*batch_shape, n, n, dtype=torch.double),
        }
        expected = []
        for seed, idx in enumerate(product(*(range(s) for s in batch_shape))):
            with random.set_seed(seed):
                model.reset_parameters()

            for k, v in state_dict.items():
                v[idx] = model.state_dict()[k]

            for i, j in product(*(range(s) for s in x_batch_shape)):
                xij = x.iloc[i, j]
                with torch.inference_mode():
                    expected.append(model(xij, **kwargs["model_kwargs"]).to_pandas())
        expected = pd.concat(expected).reset_index(drop=True)

        batched_model.load_state_dict(state_dict, strict=False)
        with torch.inference_mode():
            out = batched_model(x, **kwargs["model_kwargs"]).to_pandas()

        torch.testing.assert_close(
            torch.from_numpy(out["dr"].to_numpy()),
            torch.from_numpy(expected["dr"].to_numpy()),
            rtol=1.3e-6,
            atol=1.0e-5,
        )
        out, expected = out.drop(columns="dr"), expected.drop(columns="dr")
        pd.testing.assert_frame_equal(out, expected)

    @pytest.mark.parametrize(
        "f, variables, subcircuit_cell_types, ori_func, osi_func, osi_prob,"
        " sigma_symmetry",
        get_parameters(
            f=[("Identity",)],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
                ["cell_type", "space", "ori"],
                ["cell_type", "ori", "osi"],
            ],
            subcircuit_cell_types=[
                None,
                ["PYR"],
                ["PV"],
                ["SST"],
                ["PYR", "PV"],
                ["PYR", "SST"],
                ["PV", "SST"],
            ],
            ori_func=["cosine", "von_mises"],
            osi_func=[("Identity",), ("Pow", (0.5,))],
            osi_prob=[
                ("Uniform", 0.0, 1.0),
                ("Beta", [2.0, 1.5, 2.0], [3.0, 2.5, 1.5]),
            ],
            sigma_symmetry=[
                [[0, 1, 2], [1, 0, 2], [2, 2, 0]],
                "pre",
                "post",
                "full",
                None,
            ],
        )
        + get_parameters(
            f=[("SSN", (2,))],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
            ],
            subcircuit_cell_types=[
                None,
                ["PYR"],
                ["PV"],
                ["SST"],
                ["PYR", "PV"],
                ["PYR", "SST"],
                ["PV", "SST"],
            ],
            ori_func=["cosine", "von_mises"],
            osi_func=[("Identity",), ("Pow", (0.5,))],
            osi_prob=[("Uniform", 0.0, 1.0)],
            sigma_symmetry=["pre"],
        ),
    )
    def test_spectral_summary(
        self,
        f,
        variables,
        subcircuit_cell_types,
        ori_func,
        osi_func,
        osi_prob,
        sigma_symmetry,
    ):
        cell_types = (CellType.PYR, CellType.PV, CellType.SST)
        if subcircuit_cell_types:
            subcircuit_cell_types = tuple(CellType[ct] for ct in subcircuit_cell_types)
        n = len(cell_types)

        model_kwargs = {
            "cell_types": cell_types,
            "f": f,
            "init_vf": 0.75,
            "init_sigma_bounds": (50.0, 150.0),
            "ori_func": ori_func,
            "ori_order": (None if ori_func == "cosine" else 4),
            "osi_func": osi_func,
            "osi_prob": osi_prob,
            "sigma_symmetry": sigma_symmetry,
            "autapse": True,
        }

        model = nn.V1(variables, **model_kwargs)

        for i in range(10):
            print(i)
            with random.set_seed(i):
                model.reset_parameters()

            if "ori" in variables:
                with torch.no_grad():
                    model.kappa[:, 0] = 0.5 if ori_func == "cosine" else 3.0
                    model.kappa[:, 1:] = -0.5 if ori_func == "cosine" else -3.0

            output = model.spectral_summary(cell_types=subcircuit_cell_types)._asdict()

            N_space = (20,)
            N_ori = 10
            N_osi = 20
            for _ in range(6):
                print(N_space, N_ori, N_osi)
                neuron_kwargs = {
                    "n": (n if "cell_type" in variables else 0),
                    "N_space": (N_space if "space" in variables else ()),
                    "N_ori": (N_ori if "ori" in variables else 0),
                    "N_osi": (N_osi if "osi" in variables else 0),
                    "space_extent": [2000.0] * len(N_space),
                    "ori_extent": (-90.0, 90.0),
                    "osi_prob": osi_prob,
                }
                x = neurons.as_grid(**neuron_kwargs)

                if subcircuit_cell_types:
                    indices = [cell_types.index(ct) for ct in subcircuit_cell_types]
                    x = x.iloc[indices]

                with torch.no_grad():
                    W = model(x, output="weight", ndim=x.ndim, to_dataframe=False)

                if f[0] == "SSN":
                    (p,) = f[1]
                    W = W * p * model_kwargs["init_vf"] ** (p - 1)
                spectrum = torch.linalg.eigvals(W)

                expected = {
                    "abscissa": spectrum.real.max(),
                    "radius": spectrum.abs().max(),
                }

                for k, v1, v2 in lib.itertools.dict_zip(output, expected):
                    print(k, v1.item(), v2.item())
                    passed = ((v1 == 0.0) & (v2.abs() < 5.0e-4)).item()
                    passed |= torch.allclose(
                        v1, v2, rtol=1.0e-2 * len(variables), atol=1.0e-5
                    )
                    if not passed:
                        N_space = tuple(Ni * 2 for Ni in N_space)
                        # N_ori = N_ori * 2
                        N_osi = N_osi * 2
                        break
                else:
                    break

            else:
                assert False

    @pytest.mark.parametrize(
        "subcircuit_cell_types",
        [None, ["PYR"], ["PV"], ["PYR", "PV"]],
    )
    def test_spectral_summary_2d(self, subcircuit_cell_types):
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)
        if subcircuit_cell_types:
            subcircuit_cell_types = tuple(CellType[ct] for ct in subcircuit_cell_types)
        n = len(cell_types)

        model_kwargs = {
            "cell_types": cell_types,
            "sigma_symmetry": "pre",
            "init_sigma_bounds": (50.0, 150.0),
            "autapse": True,
        }

        model = nn.V1(variables, **model_kwargs)

        for i in range(1, 10):
            print(i)
            with random.set_seed(i):
                model.reset_parameters()

            output = model.spectral_summary(cell_types=subcircuit_cell_types)._asdict()

            N_space = (60, 60)
            for _ in range(4):
                print(N_space)
                neuron_kwargs = {
                    "n": n,
                    "N_space": N_space,
                    "space_extent": [2000.0] * len(N_space),
                }
                x = neurons.as_grid(**neuron_kwargs)

                if subcircuit_cell_types:
                    indices = [cell_types.index(ct) for ct in subcircuit_cell_types]
                    x = x.iloc[indices]

                with torch.no_grad():
                    W = model(x, output="weight", ndim=x.ndim, to_dataframe=False)

                spectrum = torch.linalg.eigvals(W)

                expected = {
                    "abscissa": spectrum.real.max(),
                    "radius": spectrum.abs().max(),
                }

                for k, v1, v2 in lib.itertools.dict_zip(output, expected):
                    print(k, v1.item(), v2.item())
                    passed = ((v1 == 0.0) & (v2.abs() < 5.0e-4)).item()
                    passed |= torch.allclose(
                        v1, v2, rtol=1.0e-2 * len(variables), atol=1.0e-5
                    )
                    if not passed:
                        N_space = tuple(Ni * 2 for Ni in N_space)
                        break
                else:
                    break

            else:
                assert False

    def test_spectral_summary_jacobian(self):
        cell_types = (CellType.PYR, CellType.PV, CellType.SST)
        tau = [1.0, 0.5, 0.75]

        model = nn.V1(["cell_type"], cell_types=cell_types, tau=tau)

        for i in range(10):
            print(i)
            with random.set_seed(i):
                model.reset_parameters()

            output = model.spectral_summary(kind="J")._asdict()

            W = model.gW * model.sign[..., None, :]
            eye = torch.eye(model.n)
            spectrum = torch.linalg.eigvals((W - eye) / torch.tensor(tau)[:, None])

            expected = {
                "abscissa": spectrum.real.max(),
                "radius": spectrum.abs().max(),
            }

            for k, v1, v2 in lib.itertools.dict_zip(output, expected):
                print(k, v1.item(), v2.item())
                torch.testing.assert_close(v1, v2)

    def test_spectral_summary_jacobian_space(self):
        cell_types = (CellType.PYR, CellType.PV)
        tau = [1.0, 0.5]

        model = nn.V1(
            ["cell_type", "space"],
            cell_types=cell_types,
            tau=tau,
            sigma_symmetry="pre",
            init_sigma_bounds=(50.0, 150.0),
            autapse=True,
        )

        tau = torch.tensor(tau)[:, None].broadcast_to(2, 480).reshape(-1)

        for i in range(10):
            print(i)
            with random.set_seed(i):
                model.reset_parameters()

            output = model.spectral_summary(kind="J")._asdict()

            x = neurons.as_grid(n=2, N_space=[480], space_extent=[2000.0])

            with torch.no_grad():
                W = model(x, output="weight", ndim=x.ndim, to_dataframe=False).dense(
                    keep_shape=False
                )

            eye = torch.eye(W.shape[-1])
            spectrum = torch.linalg.eigvals((W - eye) / tau[:, None])

            expected = {
                "abscissa": spectrum.real.max(),
                "radius": spectrum.abs().max(),
            }

            for k, v1, v2 in lib.itertools.dict_zip(output, expected):
                print(k, v1.item(), v2.item())
                torch.testing.assert_close(v1, v2, rtol=2e-3, atol=1e-5)

    @pytest.mark.parametrize("dh", [1.0, [1.0, 2.0]])
    @pytest.mark.parametrize("vf_symmetry", [True, False])
    def test_spectral_summary_dh(self, dh, vf_symmetry):
        cell_types = (CellType.PYR, CellType.PV)
        tau = [1.0, 0.5]
        dh = torch.tensor(dh) if not isinstance(dh, float) else dh

        model = nn.V1(
            ["cell_type"],
            sigma_symmetry="pre",
            cell_types=cell_types,
            tau=tau,
            f=nn.SSN(),
            vf_symmetry=vf_symmetry,
        )
        state_dict = get_state_dict(2, vf_symmetry=vf_symmetry)
        nn.load_state_dict(model, state_dict, strict=False)

        output = model.spectral_summary(kind="J", dh=dh)._asdict()

        new_vf = model.vf + dh
        gain = 2 * torch.clip(new_vf, min=0.0)
        assert gain.shape == () or gain.shape == (2,)
        W = model.W(with_gain=False)
        W = gain * W if gain.ndim == 0 else gain.diag() @ W
        eye = torch.eye(model.n)
        spectrum = torch.linalg.eigvals((W - eye) / torch.tensor(tau)[:, None])

        expected = {
            "abscissa": spectrum.real.max(),
            "radius": spectrum.abs().max(),
        }

        for k, v1, v2 in lib.itertools.dict_zip(output, expected):
            print(k, v1.item(), v2.item())
            torch.testing.assert_close(v1, v2)

    @pytest.mark.parametrize(
        "f, variables, osi_func, osi_prob, sigma_symmetry",
        get_parameters(
            f=[("Identity",)],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
                ["cell_type", "space", "ori"],
                ["cell_type", "ori", "osi"],
            ],
            osi_func=[("Identity",), ("Pow", (0.5,))],
            osi_prob=[
                ("Uniform", 0.0, 1.0),
                ("Beta", [2.0, 1.5, 2.0], [3.0, 2.5, 1.5]),
            ],
            sigma_symmetry=[
                [[0, 1, 2], [1, 0, 2], [2, 2, 0]],
                "pre",
                "post",
                "full",
                None,
            ],
        )
        + get_parameters(
            f=[("SSN", (2,)), "Match"],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
            ],
            osi_func=[("Identity",), ("Pow", (0.5,))],
            osi_prob=[("Uniform", 0.0, 1.0)],
            sigma_symmetry=["pre"],
        ),
    )
    @pytest.mark.parametrize("Ginv", [True, False])
    @pytest.mark.parametrize("H", [True, False])
    def test_spectral_norm(
        self, f, variables, osi_func, osi_prob, sigma_symmetry, Ginv, H
    ):
        if f == "Match":
            f = nn.Match({"SST": nn.SSN(2)}, nn.Identity())

        cell_types = (CellType.PYR, CellType.PV, CellType.SST)
        n = len(cell_types)

        model_kwargs = {
            "cell_types": cell_types,
            "f": f,
            "init_vf": 0.75,
            "init_sigma_bounds": (50.0, 150.0),
            "osi_func": osi_func,
            "osi_prob": osi_prob,
            "sigma_symmetry": sigma_symmetry,
            "vf_symmetry": False,
            "autapse": True,
        }

        model = nn.V1(variables, **model_kwargs)

        for i in list(range(6)) + list(range(7, 11)):
            print(i)
            with random.set_seed(i):
                model.reset_parameters()

            if "ori" in variables:
                with torch.no_grad():
                    model.kappa[:, 0] = 0.5
                    model.kappa[:, 1:] = -0.5

            output = model.spectral_norm(Ginv=Ginv, H=H)

            N_space = (20,)
            N_ori = 10
            N_osi = 20
            for _ in range(6):
                print(N_space, N_ori, N_osi)
                neuron_kwargs = {
                    "n": (n if "cell_type" in variables else 0),
                    "N_space": (N_space if "space" in variables else ()),
                    "N_ori": (N_ori if "ori" in variables else 0),
                    "N_osi": (N_osi if "osi" in variables else 0),
                    "space_extent": [2000.0] * len(N_space),
                    "ori_extent": (-90.0, 90.0),
                    "osi_prob": osi_prob,
                }
                x = neurons.as_grid(**neuron_kwargs)

                with torch.no_grad():
                    W = model(x, output="weight", ndim=x.ndim, to_dataframe=False)

                G = model.gain()
                H_ = model.kth_deriv(2) / 2
                if "osi" in variables:
                    G = G.unsqueeze(-1)
                    H_ = H_.unsqueeze(-1)
                if isinstance(W, CirculantTensor):
                    G = circulant.diag_like(W, G)
                    H_ = circulant.diag_like(W, H_)
                else:
                    G = G.diag()
                    H_ = H_.diag()

                eye = linalg.eye_like(W)
                tL = torch.linalg.inv(eye - G @ W) - eye
                if Ginv:
                    tL = torch.linalg.inv(G) @ tL
                if H:
                    tL = tL @ H_
                expected = torch.linalg.eigvalsh(tL.adjoint() @ tL)
                # theoretical the eigenvalues should be non-negative,
                # but numerical errors can lead to small negative values
                expected = expected.clip(min=0).sqrt().max()

                if variables == ["cell_type"]:
                    torch.testing.assert_close(expected, torch.linalg.svdvals(tL).max())

                assert output.shape == expected.shape
                passed = torch.allclose(
                    output, expected, rtol=1.0e-2 * len(variables), atol=1.0e-5
                )
                if not passed:
                    N_space = tuple(Ni * 2 for Ni in N_space)
                    # N_ori = N_ori * 2
                    N_osi = N_osi * 2
                else:
                    break

            else:
                assert False

    @pytest.mark.parametrize("n", [2, 2.5, 3])
    def test_gain(self, n):
        model = nn.V1(
            variables=["cell_type", "space", "ori", "ori"],
            cell_types=(CellType.PYR, CellType.PV),
            f=nn.Rectified() ** n,
            init_vf=3.0,
        )
        torch.testing.assert_close(model.gain(), torch.tensor(n * 3.0 ** (n - 1)))

    def test_vector_gain(self):
        model = nn.V1(
            variables=["cell_type", "space"],
            cell_types=(CellType.PYR, CellType.PV, CellType.SST),
            f=nn.Match({"PV": nn.SSN(3)}, nn.SSN(2)),
            init_vf=3.0,
        )
        torch.testing.assert_close(model.gain(), torch.tensor([6.0, 27.0, 6.0]))

    @pytest.mark.parametrize("n", [2, 2.5, 3])
    @pytest.mark.parametrize("k", [1, 2])
    def test_kth_deriv(self, n, k):
        model = nn.V1(
            variables=["cell_type", "space", "ori", "ori"],
            cell_types=(CellType.PYR, CellType.PV),
            f=nn.Rectified() ** n,
            init_vf=3.0,
        )
        if k == 1:
            expected = n * 3.0 ** (n - 1)
        else:
            expected = n * (n - 1) * 3.0 ** (n - 2)
        torch.testing.assert_close(model.kth_deriv(k), torch.tensor(expected))

    @pytest.mark.parametrize("k", [1, 2])
    def test_vector_kth_deriv(self, k):
        model = nn.V1(
            variables=["cell_type", "space"],
            cell_types=(CellType.PYR, CellType.PV, CellType.SST),
            f=nn.Match({"PV": nn.SSN(3)}, nn.SSN(2)),
            init_vf=3.0,
        )
        if k == 1:
            expected = [6.0, 27.0, 6.0]
        else:
            expected = [2.0, 18.0, 2.0]
        torch.testing.assert_close(model.kth_deriv(k), torch.tensor(expected))

    @pytest.mark.parametrize(
        "osi_prob", [("Beta", 2.0, 3.0), ("Beta", [2.0, 1.0], [3.0, 2.0])]
    )
    def test_osi_func(self, osi_prob):
        model = nn.V1(
            variables=["cell_type", "space", "ori", "osi"],
            cell_types=(CellType.PYR, CellType.PV),
            osi_func=0.5,
            osi_prob=osi_prob,
        )
        x = torch.linspace(0, 1, 50)
        out = model.osi_func(x)
        expected = torch.distributions.Beta(2.0, 3.0).cdf(x) ** 0.5
        torch.testing.assert_close(out, expected)

    def test_rf_cv_default_is_zero(self):
        model = nn.V1(["cell_type"], cell_types=(CellType.PYR, CellType.PV))
        torch.testing.assert_close(model.rf_cv, torch.tensor(0.0))

    @pytest.mark.parametrize("rf_cv", [-0.1, [0.1, -0.2]])
    def test_rf_cv_negative_raises(self, rf_cv):
        with pytest.raises(ValueError, match="rf_cv must be non-negative"):
            nn.V1(["cell_type"], cell_types=(CellType.PYR, CellType.PV), rf_cv=rf_cv)

    @pytest.mark.parametrize("mode", ["analytical", "newton"])
    def test_rf_cv_response_requires_numerical_or_matrix(self, mode):
        # rf_cv > 0 with output == "response" is only supported when mode is
        # "numerical" or one of the matrix modes; other modes must raise
        # NotImplementedError, while rf_cv == 0 leaves the guard inactive and runs
        # normally.
        cell_types = (CellType.PYR, CellType.PV)
        # Flatten to a non-grid population so that "matrix" and "newton", which
        # reject circulant weights, still run when the guard is inactive.
        x = neurons.as_grid(n=2, N_space=[50], space_extent=[1000.0]).reshape(-1)
        x["dh"] = torch.zeros(x.shape)
        x["dh"][0] = 1.0

        def build(rf_cv):
            model = nn.V1(
                ["cell_type", "space"],
                cell_types=cell_types,
                f=nn.Ricciardi(scale=1.0),
                init_vf=-0.1,
                rf_cv=rf_cv,
                mode=mode,
                sigma_symmetry="pre",
            )
            return model

        with pytest.raises(NotImplementedError, match="rf_cv"):
            build(0.2)(x, ndim=x.ndim)

        # rf_cv == 0 should not trigger the guard.
        build(0.0)(x, ndim=x.ndim)

    def test_rf_cv_allowed_for_weight_output(self):
        # The rf_cv restriction only applies to output == "response"; computing
        # weights with a non-numerical mode and rf_cv > 0 must not raise.
        model = nn.V1(
            ["cell_type", "space"],
            cell_types=(CellType.PYR, CellType.PV),
            f=nn.Ricciardi(scale=1.0),
            init_vf=-0.1,
            rf_cv=0.2,
            mode="analytical",
            sigma_symmetry="pre",
        )
        x = neurons.as_grid(n=2, N_space=[50], space_extent=[1000.0])
        model(x, output="weight", ndim=x.ndim, to_dataframe=False)

    @pytest.mark.parametrize("mode", ["numerical", "matrix"])
    def test_rf_cv_zero_is_noop(self, mode):
        # With rf_cv == 0 the per-neuron resampling is skipped, so the response
        # is deterministic and identical to a model constructed without rf_cv.
        cell_types = (CellType.PYR, CellType.PV)
        common = {
            "cell_types": cell_types,
            "f": nn.Ricciardi(scale=1.0),
            "init_vf": -0.1,
            "mode": mode,
            "sigma_symmetry": "pre",
        }
        with random.set_seed(0):
            model = nn.V1(["cell_type", "space"], rf_cv=0.0, **common)
            model.reset_parameters()
        expected_model = nn.V1(["cell_type", "space"], **common)
        expected_model.load_state_dict(model.state_dict())

        x = neurons.as_grid(n=2, N_space=[50], space_extent=[1000.0])
        x["dh"] = torch.zeros(x.shape)
        x["dh"][0, 0] = 1.0
        x = x.reshape(-1)

        with random.set_seed(1):
            out = model(x, ndim=x.ndim, to_dataframe=False)["dr"]
        with random.set_seed(2):
            expected = expected_model(x, ndim=x.ndim, to_dataframe=False)["dr"]
        # Different seeds, yet identical output, since rf_cv == 0 draws no noise.
        torch.testing.assert_close(out, expected)

    @pytest.mark.parametrize("mode", ["numerical", "matrix"])
    def test_rf_cv_response_is_stochastic(self, mode):
        # With rf_cv > 0 the response depends on per-neuron sampled vf:
        # reproducible under a fixed seed, different across seeds, and different
        # from the rf_cv == 0 (homogeneous) response.
        cell_types = (CellType.PYR, CellType.PV)
        common = {
            "cell_types": cell_types,
            "f": nn.Ricciardi(scale=1.0),
            "init_vf": -0.1,
            "mode": mode,
            "sigma_symmetry": "pre",
        }
        with random.set_seed(0):
            model = nn.V1(["cell_type", "space"], rf_cv=0.3, **common)
            model.reset_parameters()
        homogeneous = nn.V1(["cell_type", "space"], rf_cv=0.0, **common)
        homogeneous.load_state_dict(model.state_dict())

        x = neurons.as_grid(n=2, N_space=[50], space_extent=[1000.0])
        x["dh"] = torch.zeros(x.shape)
        x["dh"][0, 0] = 1.0
        x = x.reshape(-1)

        with random.set_seed(1):
            out1 = model(x, ndim=x.ndim, to_dataframe=False)["dr"]
        with random.set_seed(1):
            out2 = model(x, ndim=x.ndim, to_dataframe=False)["dr"]
        with random.set_seed(2):
            out3 = model(x, ndim=x.ndim, to_dataframe=False)["dr"]
        base = homogeneous(x, ndim=x.ndim, to_dataframe=False)["dr"]

        torch.testing.assert_close(out1, out2)  # reproducible under the same seed
        assert not torch.allclose(out1, out3)  # different seeds -> different draws
        assert not torch.allclose(out1, base)  # heterogeneity changes the response

    @pytest.mark.parametrize("rf_cv", [0.2, [0.2, 0.35]])
    def test_rf_cv_matches_target_moments(self, rf_cv, monkeypatch):
        # The per-neuron vf is drawn so that the baseline rate f(vf) has mean
        # f(init_vf) and coefficient of variation rf_cv (per cell type). Intercept
        # the vf handed to the numerical solver and check its empirical moments.
        cell_types = (CellType.PYR, CellType.PV)
        model = nn.V1(
            ["cell_type"],
            cell_types=cell_types,
            f=nn.Ricciardi(scale=1.0),
            init_vf=-0.1,
            rf_cv=rf_cv,
            mode="numerical",
        )

        N = 100_000
        x = neurons.as_grid(n=2, N_space=[N], cell_types=cell_types)  # (2, N)
        x["dh"] = torch.zeros(x.shape)

        class _Stop(Exception):
            pass

        captured = {}

        def spy(vf, *args, **kwargs):
            captured["vf"] = vf.detach().clone()
            raise _Stop()

        monkeypatch.setattr(numerics, "perturbed_response", spy)
        with pytest.raises(_Stop), random.set_seed(0):
            model(x, ndim=2)

        vf = captured["vf"]
        assert vf.shape == (2, N)
        rf = model.f(vf)  # (2, N) baseline rates
        target_mean = model.f(model.vf)  # f(init_vf), shared across cell types
        emp_mean = rf.mean(dim=-1)
        emp_cv = rf.std(dim=-1) / emp_mean

        torch.testing.assert_close(
            emp_mean, target_mean.broadcast_to(2), rtol=2e-2, atol=1e-5
        )
        torch.testing.assert_close(
            emp_cv, model.rf_cv.broadcast_to(2), rtol=2e-2, atol=1e-5
        )


def test_cdim():
    x = neurons.as_grid(
        n=2,
        N_space=[100] * 3,
        N_ori=10,
        N_osi=10,
        space_extent=[2000.0] * 3,
        ori_extent=(-90.0, 90.0),
    )
    out = _cdim(x, x.ndim)
    expected = (-5, -4, -3, -2)

    assert out == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([[0.0], [1.0], [2.0 + 1.0e-6]], (-1,)),
        ([[0.0], [1.5], [2.0 + 1.0e-6]], ()),
    ],
)
def test_cdim2(x, expected):
    x = frame.ParameterFrame({"space": periodic.tensor(x, extents=[(0.0, 3.0)])})
    out = _cdim(x, x.ndim)
    assert out == expected


@pytest.mark.parametrize(
    "osi_func, osi_prob, expected",
    [
        ("Identity", ("Beta", (2.0, 3.0)), 1 / 5),
        ("Identity", ("Beta", ([2.0, 1.0], [3.0, 2.0])), [1 / 5, 1 / 6]),
        (("Pow", (0.5,)), ("Beta", (2.0, 3.0)), 2 / 5),
        (("Pow", (0.5,)), ("Beta", ([2.0, 1.0], [3.0, 2.0])), [2 / 5, 1 / 3]),
        (1.0, ("Beta", (2.0, 3.0)), 1 / 3),
        (0.5, ("Beta", (2.0, 3.0)), 1 / 2),
        (1.0, ("Beta", ([2.0, 1.0], [3.0, 2.0])), [1 / 3, 4 / 15]),
        (0.5, ("Beta", ([2.0, 1.0], [3.0, 2.0])), [1 / 2, 2 / 5]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_compute_osi_scale(osi_func, osi_prob, expected, dtype):
    if not isinstance(osi_func, float):
        osi_func = utils.call(nn, osi_func)
    osi_prob = utils.call(
        torch.distributions, (osi_prob[0], [torch.tensor(x) for x in osi_prob[1]])
    )
    out = compute_osi_scale(osi_prob, osi_func=osi_func, dtype=dtype)
    expected = torch.tensor(expected, dtype=dtype)
    torch.testing.assert_close(out, expected)
