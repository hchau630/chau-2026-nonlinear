import pytest
import torch
from scipy import spatial, stats

from niarb import neurons, random
from niarb.cell_type import CellType


@pytest.mark.parametrize(
    "osi_prob",
    [(2.5, 6.0), ([2.5, 1.0, 2.0], [6.0, 3.0, 5.0])],
)
def test_as_grid(osi_prob):
    osi_prob = torch.distributions.Beta(*torch.tensor(osi_prob))

    n = 3
    x = neurons.as_grid(
        n=n,
        N_ori=10,
        N_osi=10,
        osi_prob=osi_prob,
    )

    if osi_prob is None:
        alpha, beta = [1.0] * n, [1.0] * n
    else:
        osi_prob = osi_prob.expand((n,))
        alpha, beta = osi_prob.concentration1.tolist(), osi_prob.concentration0.tolist()

    for i in range(n):
        res = stats.kstest(x["osi"][i].reshape(-1), "beta", args=(alpha[i], beta[i]))
        assert res.pvalue > 0.05


@pytest.mark.parametrize("w_dims", [None, ()])
@pytest.mark.parametrize("min_dist_cell_types", [False, True])
def test_sample(w_dims, min_dist_cell_types):
    d, N, min_dist = 3, 12500, 15.0
    osi = [(2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]
    min_dist_cell_types = {"PV": 50.0, "SST": 60.0} if min_dist_cell_types else {}
    with random.set_seed(1):
        x = neurons.sample(
            N,
            ["space", "ori", "osi", "cell_type"],
            cell_probs=[0.9, 0.05, 0.05],
            space_extent=[1000.0] * d,
            ori_extent=(-90.0, 90.0),
            osi_prob=torch.distributions.Beta(*torch.tensor(list(zip(*osi)))),
            min_dist=min_dist,
            min_dist_cell_types=min_dist_cell_types,
            w_dims=w_dims,
        )

    assert set(x.keys()) == {"space", "ori", "osi", "cell_type", "space_dV", "dV"}
    assert x.shape == (N,)
    assert x["space"].shape == (N, d)
    if w_dims is None:
        assert x["space"].extents == tuple([(-500.0, 500.0)] * d)
        assert x["space"].w_dims == tuple(range(d))
    else:
        assert x["space"].extents == ()
        assert x["space"].w_dims == ()
    assert (
        x["ori"].shape == (N, 1)
        and x["ori"].w_dims == (0,)
        and x["ori"].extents == ((-90.0, 90.0),)
    )
    assert x["osi"].shape == (N,) and x["osi"].dtype == torch.float
    assert x["cell_type"].shape == (N,) and x["cell_type"].dtype == torch.long
    assert x["dV"].shape == (N,)
    assert x["space"].max() < 500.0 and x["space"].min() > -500.0
    assert x["ori"].max() < 90.0 and x["ori"].min() > -90.0
    assert x["osi"].max() < 1.0 and x["osi"].min() > 0.0
    for i, o in enumerate(osi):
        assert stats.kstest(x["osi"][x["cell_type"] == i], "beta", args=o).pvalue > 0.04
    assert x["cell_type"].max() == 2 and x["cell_type"].min() == 0
    distances = spatial.distance.pdist(x["space"])
    assert (distances > min_dist).all()
    for ct, min_dist_ct in min_dist_cell_types.items():
        distances = spatial.distance.pdist(x["space"][x["cell_type"] == ct])
        assert (distances > min_dist_ct).all()
    assert (x["dV"][x["cell_type"] == "PYR"] == (1.0e3**d) * 180 / (0.9 * N)).all()
    assert (x["dV"][x["cell_type"] != "PYR"] == (1.0e3**d) * 180 / (0.05 * N)).all()
    assert (x["space_dV"][x["cell_type"] == "PYR"] == 1.0e3**d / (0.9 * N)).all()
    assert (x["space_dV"][x["cell_type"] != "PYR"] == 1.0e3**d / (0.05 * N)).all()


@pytest.mark.parametrize(
    "cell_types, cell_probs",
    [
        (
            [CellType.PYR, CellType.PV, CellType.SST, CellType.VIP],
            [0.85, 0.043, 0.032, 0.075],
        ),
        ([CellType.PYR, CellType.PV], [0.85, 0.15]),
        ([CellType.PYR, CellType.PV, CellType.SST], [0.85, 0.086, 0.064]),
        ([CellType.PYR, CellType.PV, CellType.VIP], [0.85, 0.05466, 0.09534]),
        ([CellType.PYR, CellType.VIP, CellType.SST], [0.85, 0.10514, 0.04486]),
        ([CellType.PV, CellType.SST, CellType.VIP], [0.043, 0.032, 0.075]),
        ([CellType.PV, CellType.SST], [0.043, 0.032]),
    ],
)
def test_sample_cell_probs(cell_types, cell_probs):
    N = 3000
    with random.set_seed(0):
        x = neurons.sample(N, ["cell_type"], cell_types=cell_types)

    f_obs = x["cell_type"].bincount().tolist()
    f_exp = [prob / sum(cell_probs) * N for prob in cell_probs]

    assert stats.chisquare(f_obs, f_exp).pvalue > 0.05


@pytest.mark.parametrize(
    "cell_probs, expected",
    [
        ([0.73, 0.145, 0.135], [7, 2, 1]),
        ([0.74, 0.13, 0.13], [8, 1, 1]),
    ],
)
def test_sample_cell_probs_strict(cell_probs, expected):
    N = 10
    cell_types = [CellType.PYR, CellType.PV, CellType.SST]

    x = neurons.sample(
        N,
        ["cell_type"],
        cell_types=cell_types,
        cell_probs=cell_probs,
        cell_probs_strict=True,
    )
    out = x["cell_type"].bincount().tolist()

    assert (x["dV"] == 1 / torch.tensor(out)[x["cell_type"]]).all()
    assert out == expected
