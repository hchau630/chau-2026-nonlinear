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

    # setup SSN E-I ring model
    model = nn.V1(
        ["cell_type", "space"],
        cell_types=x["cell_type"].categories,
        f="SSN",
        sigma_symmetry="pre",
        init_stable=True,
        mode="numerical",  # `numerical` mode means we perform numerical simulations
    )

    return x, dh, model


def ground_truth(x, dh, model):
    x = x.copy()

    # a random perturbation strength
    x = x.unsqueeze(0)
    x["dh"] = dh * 10.0

    # resample model parameters. A random large number is chosen as seed
    # so that we don't accidentally use the same seed during the fitting procedure.
    model = copy.deepcopy(model)
    with random.set_seed(2**16 + 4):
        model.reset_parameters()

    # also set baseline voltage of model to 0.5
    torch.nn.init.constant_(model.vf, 0.5)

    # compute distance of each neuron to perturbed cell
    x["distance"] = perturbation.min_distance_to_ensemble(x)

    # provide information about perturbed cell type
    x["perturbed_cell_type"] = categorical.tensor([0, 1], categories=["PYR", "PV"])[
        (slice(None),) + (None,) * (x.ndim - 1)
    ]

    # run model
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

    return copy.deepcopy(nn.state_dict(model)), [data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--log-level", "--ll", dest="log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    if args.log_level == "DEBUG":
        logging.getLogger("matplotlib").setLevel("INFO")
    logger = logging.getLogger()

    x, dh, model = setup()

    ground_truth_state_dict, data = ground_truth(x, dh, model)
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
    pipeline = nn.Pipeline(model=model, data=data)

    losses, state_dicts = fit.run(
        data, dataset, pipeline, N=10, progress=True, seed=0, dtype="double"
    )

    # now that we have fitted the model, switch model mode back to `numerical`
    # to check that the simulation output of the fitted model actually matches
    # the data.
    pipeline.model.mode = "numerical"

    min_loss = min(losses)
    best_state_dict = state_dicts[losses.index(min_loss)]

    logger.info(f"{min_loss=}")
    logger.info(f"Best state dict:\n{best_state_dict}")
    logger.info(f"Ground truth state dict:\n{ground_truth_state_dict}")

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
