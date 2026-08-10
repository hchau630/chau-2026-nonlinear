import copy
import logging
import argparse

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from niarb.cli import fit
from niarb.dataset import Dataset
from niarb.tensors import categorical
from niarb import neurons, nn, perturbation, random


def setup():
    # setup E-I ring neurons
    x = neurons.as_grid(
        n=2,
        N_space=(100,),
        cell_types=["PYR", "PV"],
        space_extent=[1000.0],
    )

    # setup single PYR and PV cell perturbations
    dh = torch.zeros(2, *x.shape)
    dh[0, 0, 0] = 1.0
    dh[1, 1, 0] = 1.0

    # setup E-I ring model with Ricciardi nonlinearity
    model = nn.V1(
        ["cell_type", "space"],
        cell_types=x["cell_type"].categories,
        f="Ricciardi",
        sigma_symmetry="pre",
        init_stable=True,
        mode="numerical",  # `numerical` mode means we perform numerical simulations
        simulation_kwargs=dict(options=dict(max_num_steps=1000), max_t=1000.0),
    )

    return x, dh, model


def ground_truth(x, dh, model, dtype=None):
    x = x.copy()

    # a random perturbation strength
    x = x.unsqueeze(0)
    x["dh"] = dh * 10

    # resample model parameters. A random large number is chosen as seed
    # so that we don't accidentally use the same seed during the fitting procedure.
    model = copy.deepcopy(model)
    with random.set_seed(2**16 + 4):
        model.reset_parameters()

    # also set a different baseline voltage of model
    torch.nn.init.constant_(model.vf, 0.5)

    # also set a different alpha value of ricciardi nonlinearity
    # alpha is a multiplicative scaling factor with units which
    # modifies the threshold at which the firing rate saturates
    torch.nn.init.constant_(model.f.scale, 0.015)
    print(model.f(model.vf), model.gain())

    # compute distance of each neuron to perturbed cell
    x["distance"] = perturbation.min_distance_to_ensemble(x)

    # provide information about perturbed cell type
    x["perturbed_cell_type"] = categorical.tensor([0, 1], categories=["PYR", "PV"])[
        (slice(None),) + (None,) * (x.ndim - 1)
    ]

    # run model
    x = x.to(dtype=dtype)
    model.to(dtype=dtype)
    with torch.no_grad():
        data = model(x, ndim=x.ndim - 1).to_pandas()

    # generate pseudo experimental data
    # IMPORTANT: due to limitation of the current implementation,
    # all intervals in fitted data must be closed on the left and open on the right,
    # thus we use right=False in pd.cut.
    data["distance"] = pd.cut(
        data["distance"], bins=np.arange(10, 451, 20), right=False
    )
    data = data.groupby(
        ["perturbed_cell_type", "cell_type", "distance"], as_index=False
    )["dr"].mean()

    return copy.deepcopy(nn.state_dict(model)), x["dh"].max().item(), [data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--log-level", "--ll", dest="log_level", type=str, default="INFO"
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    if args.log_level == "DEBUG":
        logging.getLogger("matplotlib").setLevel("INFO")
    logger = logging.getLogger()

    x, dh, model = setup()

    ground_truth_state_dict, ground_truth_dh, data = ground_truth(
        x, dh, model, dtype=torch.float64
    )
    if args.save:
        torch.save(ground_truth_state_dict, "data/state_dict.pt")
        data[0].to_pickle("data/data.pkl")

    # change model to `analytical` mode for better model fitting performance
    # `analytical` mode means rather than simulating the dynamics, we start
    # from the exact solution of a linearized model, then use an iterative
    # procedure to find the exact solution of the nonlinear model.
    model.mode = "analytical"
    model.nonlinear_kwargs = {"assert_convergence": False, "max_num_steps": 10}

    dataset = Dataset(
        neurons=x,
        perturbations={
            "configs": [
                {"N": 1, "perturbed_cell_type": "PYR"},
                {"N": 1, "perturbed_cell_type": "PV"},
            ],
            "mapping": {
                "PYR": {"cell_probs": {"PYR": 1.0}},
                "PV": {"cell_probs": {"PV": 1.0}},
            },
        },
        data=data,
        metrics={"distance": "min_distance_to_ensemble"},
        seed=0,
    )
    pipeline = nn.Pipeline(model=model, data=data, scaler={"requires_optim": True})

    losses, state_dicts = fit.run(
        data,
        dataset,
        pipeline,
        N=10,
        progress=True,
        dtype="double",
        optimizer={
            "options": {"maxiter": 200}
        },  # sometimes optimizer gets stuck for a long time
        seed=0,
    )

    # now that we have fitted the model, switch model mode back to `numerical`
    # to check that the simulation output of the fitted model actually matches
    # the data.
    pipeline.model.mode = "numerical"
    # pipeline.model.simulation_kwargs = dict(dx_rtol=1.0e-7)

    min_loss = min(losses)
    best_state_dict = state_dicts[losses.index(min_loss)]

    logger.info(f"{min_loss=}")
    logger.info(f"Best state dict:\n{best_state_dict}")
    logger.info(f"Ground truth state dict:\n{ground_truth_state_dict}")
    logger.info(f"Ground truth dh: {ground_truth_dh}")

    nn.load_state_dict(pipeline, best_state_dict)
    x, _, kwargs = dataset[0]
    pred = pipeline[:"analysis"](x, **kwargs)

    df = pd.concat({"data": data[0], "model": pred[0].to_pandas()})
    df = df.reset_index(0, names="source")
    df["distance"] = pd.IntervalIndex(df["distance"]).mid

    sns.relplot(
        df,
        x="distance",
        y="dr",
        hue="perturbed_cell_type",
        col="cell_type",
        style="source",
        kind="line",
        facet_kws={"sharey": False},
    )
    if args.save:
        plt.savefig("figures/fit.pdf")
    plt.show()


if __name__ == "__main__":
    main()
