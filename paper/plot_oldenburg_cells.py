import argparse
from pathlib import Path
import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch

from niarb import viz
from niarb.distributions import RectLinePicking
from mpl_config import FIGSIZE, RECT, set_rcParams

logger = logging.getLogger(__name__)


MAPPING = {
    "distance_2d": "Distance (μm)",
    "srel_ori": "Δ pref. ori. (°)",
    "osi": "OSI",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--log-level", "--ll", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(format="%(lineno)d:%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.log_level))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # matplotlib is noisy

    set_rcParams()

    data_dir = args.dir / "ExperimentalResults" / "compressedData"

    # get unique cell properties from cell_table
    cell_df = pd.read_csv(data_dir / "cell_table_250622_2d.csv")
    cell_df = cell_df.groupby("cellID", as_index=False).first()

    # rename to fit python snake case convention
    cell_df = cell_df.rename(columns={"cellOSI": "osi"})

    # plot distribution of x and y coordinates
    df = cell_df.melt(
        id_vars=["cellID"],
        value_vars=["x", "y"],
        var_name="kind",
        value_name="Coordinate (μm)",
    )
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    viz.mapped(sns.histplot, MAPPING)(
        df,
        x="Coordinate (μm)",
        hue="kind",
        stat="density",
        bins=20,
        palette=["#F27189", "#51B148"],
        ax=ax,
    )
    ax.set_xticks([0, 250, 500, 750])
    if args.out:
        fig.savefig(
            args.out / "S2a.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
        )
    plt.show() if args.show else plt.close()

    # # calculate pairwise distances
    # cell_pairs_df = cell_df.merge(cell_df, on=["expNum"], suffixes=("_0", "_1"))
    # cell_pairs_df["distance_2d"] = cell_pairs_df.eval(
    #     "((x_1 - x_0)**2 + (y_1 - y_0)**2)**0.5"
    # )
    # cell_pairs_df["distance_3d"] = cell_pairs_df.eval(
    #     "((x_1 - x_0)**2 + (y_1 - y_0)**2 + (z_1 - z_0)**2)**0.5"
    # )
    # # filter out distances between a cell and itself and remove duplicates
    # cell_pairs_df = cell_pairs_df.query("distance_3d > 0 and x_1 > x_0")

    # q = 0.01
    # a, b = [
    #     round(cell_df[k].quantile(1 - q) - cell_df[k].quantile(q), 1)
    #     for k in ["x", "y"]
    # ]
    # fig = plt.figure(figsize=FIGSIZE)
    # ax = fig.add_axes(RECT)
    # viz.mapped(sns.histplot, MAPPING)(
    #     cell_pairs_df,
    #     x="distance_2d",
    #     stat="density",
    #     common_norm=False,
    #     legend=False,
    #     ax=ax,
    # )
    # print(f"{a=:.4g}, {b=:.4g}")
    # dist = RectLinePicking(a, b)
    # x = torch.linspace(dist.support.lower_bound, dist.support.upper_bound, 100)
    # ax.plot(x, torch.exp(dist.log_prob(x)))
    # if args.out:
    #     fig.savefig(
    #         args.out / "S2b.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
    #     )
    # plt.show() if args.show else plt.close()

    # Plot overall distribution of OSI
    sf = cell_df.query("~osi.isna()")
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    viz.mapped(sns.histplot, MAPPING)(
        sf, x="osi", stat="density", bins=20, ax=ax, color="grey"
    )
    x = np.linspace(0, 1, 50)
    a, b, _, _ = stats.beta.fit(sf["osi"], floc=0, fscale=1)
    ax.plot(x, stats.beta.pdf(x, a, b), color="black")
    ax.set_title(f"$\\alpha$={a:.2f}, $\\beta$={b:.2f}")
    if args.out:
        fig.savefig(
            args.out / "S2b.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
        )
    plt.show() if args.show else plt.close()


if __name__ == "__main__":
    main()
