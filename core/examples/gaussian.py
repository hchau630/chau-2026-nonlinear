import logging

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from niarb import neurons, nn, viz
from niarb.cell_type import CellType

# Set up logging
logging.basicConfig()
# Example of setting the logging level of a particular file to DEBUG
logging.getLogger("niarb.viz").setLevel(logging.DEBUG)


def main():
    cell_types = [CellType.PYR, CellType.PV]
    N_space = (50, 50)  # number of neurons along each spatial dimension
    space_extent = (150, 150)  # in degrees
    sigma = 20.0  # sigma for gaussian input

    # create a grid of neurons
    x = neurons.as_grid(
        n=len(cell_types),
        N_space=N_space,
        cell_types=cell_types,
        space_extent=space_extent,
    )  # (2, *N_space)

    # calculate distance of each neuron to origin
    x["distance"] = x["space"].norm(dim=-1)  # (2, *N_space)

    # define gaussian input
    x["dh"] = torch.exp(-x["distance"] ** 2 / (2 * sigma**2))

    # define V1 model
    model = nn.V1(
        ["cell_type", "space"],  # connectivity depends on both cell type and space
        cell_types=cell_types,
        f=nn.SSN(),  # SSN nonlinearity
        tau=[1.0, 0.5],  # relative time constants of E and I
        mode="numerical",  # run numerical simulation
        space_strength_kernel=nn.Gaussian,  # Gaussian spatial connectivity kernel
        init_vf=0.5,  # baseline voltage of neurons (a value of 0.5 is chosen so that the gain is 1)
    )

    # set model parameters
    state_dict = {
        "gW": torch.tensor(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        ),
        "sigma": torch.tensor(
            [
                [10.0, 15.0],
                [15.0, 10.0],
            ]
        ),
    }
    model.load_state_dict(state_dict, strict=False)

    # plot model weights
    weights = model(x, ndim=x.ndim, output="weight", to_dataframe="pandas")
    weights["distance"] = pd.cut(weights["distance"], bins=500)
    weights["W"] = np.abs(weights["W"])
    print(
        weights.groupby(["presynaptic_cell_type", "postsynaptic_cell_type"])["W"].sum()
        / 2500
    )  # check that it should be the same as the "gW" parameter
    viz.figplot(  # note that this just calls sns.relplot under the hood
        weights,
        "relplot",
        x="distance",
        y="W",
        hue="postsynaptic_cell_type",
        col="presynaptic_cell_type",
        errorbar="se",
        kind="line",
        grid="yzero",
    )
    plt.show()

    # plot model response
    response = model(x, ndim=x.ndim, to_dataframe="pandas")
    viz.figplot(
        response,
        "relplot",
        x="distance",
        y="dr",
        hue="cell_type",
        kind="line",
        grid="yzero",
    )
    plt.show()


if __name__ == "__main__":
    main()
