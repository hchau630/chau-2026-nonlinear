import argparse
from pathlib import Path
import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from scipy import stats, optimize
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


def logpdf(a, b, k0, k1, k2, dori, osi):
    return stats.beta.logpdf(osi, a, b) + stats.vonmises.logpdf(
        dori, np.exp(k0 + k1 * osi + k2 * osi**2)
    )


# def logpdf(a, b, k0, k1, k2, dori, osi):
#     kappa = k0 + k1 * osi + k2 * osi**2
#     out = stats.beta.logpdf(osi, a, b)
#     out += np.where(
#         kappa >= 0,
#         stats.vonmises.logpdf(dori, np.abs(kappa)),
#         stats.vonmises.logpdf(dori, np.abs(kappa), loc=np.pi),
#     )
#     return out


def neg_log_likelihood(args, dori, osi):
    return -np.sum(logpdf(*args, dori, osi))


def fit_joint(dori, osi):
    mask = ~np.isnan(dori) & ~np.isnan(osi)
    dori, osi = dori[mask], osi[mask]

    x0 = np.array([*stats.beta._fitstart(osi)[:2], 0.0, 0.0, 0.0])
    bounds = [(0.0, None)] * 2 + [(None, None)] * (len(x0) - 2)
    res = optimize.minimize(neg_log_likelihood, x0, args=(dori, osi), bounds=bounds)
    if not res.success:
        raise ValueError(res.message)

    logger.info(f"Negative log likelihood: {neg_log_likelihood(res.x, dori, osi)}")
    return res.x


def plot_joint(params, osi_bins, scale_x=90 / np.pi, num=50, ax=None, colors=None):
    if ax is None:
        ax = plt.gca()

    if colors is None:
        colors = [None] * len(osi_bins)

    x = np.linspace(-np.pi, np.pi / 2, num)
    for osi_bin, color in zip(osi_bins, colors):
        y = np.exp(logpdf(*params, x, osi_bin.mid)) * osi_bin.length
        ax.plot(x * scale_x, y / scale_x, color=color)


def categorize_ensembles(df):
    # categorize untuned vs cotuned ensembles
    df["ens_tuning"] = pd.NA
    df.loc[(df["cellMeanEnsOSI"] < 0.5) & (df["cellEnsOSI"] < 0.3), "ens_tuning"] = (
        "untuned"
    )
    df.loc[(df["cellMeanEnsOSI"] > 0.5) & (df["cellEnsOSI"] > 0.7), "ens_tuning"] = (
        "cotuned"
    )

    # categorize diffuse vs compact ensembles
    if "cellEnsMeaD" in df.columns:
        df["density"] = pd.NA
        df.loc[df["cellEnsMeaD"] >= 200, "density"] = "diffuse"
        df.loc[df["cellEnsMeaD"] < 200, "density"] = "compact"

    return df


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

    # Plot target ROI distributions
    roi_df = categorize_ensembles(pd.read_csv(data_dir / "roi_table_250622.csv"))
    roi_pairs_df = roi_df.merge(
        roi_df, on=["ensNum", "expNum", "density", "ens_tuning"], suffixes=("_0", "_1")
    )
    roi_pairs_df["distance_2d"] = roi_pairs_df.eval(
        "((x_1 - x_0)**2 + (y_1 - y_0)**2)**0.5"
    )
    roi_pairs_df["distance_3d"] = roi_pairs_df.eval(
        "((x_1 - x_0)**2 + (y_1 - y_0)**2 + (z_1 - z_0)**2)**0.5"
    )
    # filter out distances between a cell and itself and remove duplicates
    roi_pairs_df = roi_pairs_df.query("distance_3d > 0 and x_1 > x_0")

    # g = sns.displot(
    #     roi_df.melt(
    #         id_vars=["ensNum", "expNum", "density", "ens_tuning"],
    #         value_vars=["x", "y", "z"],
    #     ),
    #     x="value",
    #     hue="density",
    #     col="variable",
    #     row="ens_tuning",
    #     stat="proportion",
    #     bins=20,
    #     common_norm=False,
    #     height=2,
    # )
    # if args.out:
    #     g.savefig(
    #         args.out / "S2a.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
    #     )
    # plt.show() if args.show else plt.close()

    # g = sns.displot(roi_df, x="x", y="y", col="density", row="ens_tuning", height=2)
    # if args.out:
    #     g.savefig(
    #         args.out / "S2a.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
    #     )
    # plt.show() if args.show else plt.close()

    qs = {"diffuse": 0.05, "compact": 0.1}
    params = {
        g: [
            round(sf[k].quantile(1 - qs[g]) - sf[k].quantile(qs[g]), 1)
            for k in ["x", "y"]
        ]
        for g, sf in roi_df.groupby("density")
    }
    hue_order = ["diffuse", "compact"]
    palette = {"diffuse": "#939598", "compact": "#000000"}
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    viz.mapped(sns.histplot, MAPPING)(
        roi_pairs_df,
        x="distance_2d",
        hue="density",
        hue_order=hue_order,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        legend=False,
        palette=list(palette.values()),
        ax=ax,
    )
    for density, sf in roi_pairs_df.groupby("density"):
        a, b = params[density]
        print(f"{a=:.4g}, {b=:.4g}")
        dist = RectLinePicking(a, b)
        x = torch.linspace(dist.support.lower_bound, dist.support.upper_bound, 100)
        ax.plot(x, torch.exp(dist.log_prob(x)), c=palette[density])
    if args.out:
        fig.savefig(
            args.out / "S2e.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
        )
    plt.show() if args.show else plt.close()

    # bins = np.histogram_bin_edges(roi_pairs_df["distance_2d"], bins=30)
    # g = viz.figplot(
    #     roi_pairs_df,
    #     func="displot",
    #     x="distance_2d",
    #     hue="density",
    #     hue_order=["diffuse", "compact"],
    #     col="ens_tuning",
    #     col_order=["untuned", "cotuned"],
    #     stat="proportion",
    #     common_norm=False,
    #     bins=bins,
    #     estimator="mean",
    #     height=2,
    # )
    # if args.out:
    #     g.savefig(
    #         args.out / "S2a.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
    #     )
    # plt.show() if args.show else plt.close()

    # Load target cells dataframe
    target_df = categorize_ensembles(pd.read_csv(data_dir / "target_table_250622.csv"))
    # print(target_df.groupby(["expNum", "ensNum"]).count().describe())

    # convert cellPO to direction in degrees
    target_df["cellPO"] = np.r_[pd.NA, 0:360:45][target_df["cellPO"] - 1]
    # convert to orientation in degrees
    target_df["cellPO"] = target_df["cellPO"] % 180
    # compute (signed and unsigned) relative orientation
    target_df["srel_ori"] = ((target_df["cellPO"] - target_df["ensPO"] + 90) % 180) - 90
    target_df["rel_ori"] = target_df["srel_ori"].abs()

    # sf = target_df.groupby(["ensNum", "expNum"], as_index=False)["ensPO"].mean()
    # print(sf.groupby(["expNum", "ensPO"]).count())
    # sns.histplot(sf, x="ensPO")
    # plt.show()

    # validate against cell_table
    cell_df = pd.read_csv(data_dir / "cell_table_250622_2d.csv")
    cell_df = cell_df.set_index(["ensNum", "expNum", "cellID"])
    sf = target_df.query("cellID != -1").set_index(["ensNum", "expNum", "cellID"])
    pd.testing.assert_series_equal(
        cell_df.loc[sf.index, "cellOrisDiff"].astype("Int64"),
        sf["rel_ori"].astype("Int64"),
        check_names=False,
    )

    # convert ori to float
    target_df["srel_ori"] = pd.to_numeric(target_df["srel_ori"])
    target_df["rel_ori"] = pd.to_numeric(target_df["rel_ori"])

    # rename to fit python snake case convention
    target_df = target_df.rename(columns={"cellOSI": "osi"})
    # print(target_df)

    # # compute ensemble OSI using circular variance
    # target_df["_z"] = target_df["osi"] * np.exp(target_df["srel_ori"] / 90 * np.pi * 1j)
    # target_df["ens_osi_cv"] = (
    #     target_df.groupby(["expNum", "ensNum"])["_z"].transform("mean").abs()
    # )
    # viz.figplot(
    #     target_df,
    #     func="displot",
    #     x="ens_osi_cv",
    #     hue="ens_tuning",
    #     hue_order=["untuned", "cotuned"],
    #     col="density",
    #     col_order=["diffuse", "compact"],
    #     stat="density",
    #     element="step",
    #     fill=False,
    #     common_norm=False,
    #     title_verbosity=0,
    # )
    # plt.show() if args.show else plt.close()

    # Fit and plot joint distribution of relative orientation and OSI of ensembles
    target_df["binned_osi"] = pd.cut(target_df["osi"], bins=[0.0, 0.25, 0.5, 0.75, 1.0])
    target_df["binned_osi_val"] = pd.IntervalIndex(target_df["binned_osi"]).mid

    col_order = ["untuned", "cotuned"]
    g = viz.figplot(
        target_df,
        func="displot",
        x="srel_ori",
        hue="binned_osi_val",
        col="ens_tuning",
        col_order=col_order,
        stat="density",
        element="step",
        fill=False,
        bins=[-112.5, -67.5, -22.5, 22.5, 67.5],
        mapping=MAPPING,
        height=FIGSIZE[1],
        aspect=0.9,
    )
    for ax in g.axes.flat:
        ax.set_xticks([-90, 0, 90])

    # obtain colors used by seaborn for hue, code with help from ChatGPT-5.4
    values = target_df["binned_osi_val"][~target_df["binned_osi_val"].isna()].unique()
    cmap = sns.color_palette("ch:", as_cmap=True)  # seaborn's default numeric palette
    norm = colors.Normalize(vmin=min(values), vmax=max(values))
    hex_colors = [colors.to_hex(cmap(norm(v))) for v in sorted(values)]

    # plot fitted joint distributions
    names = ["a", "b", "k0", "k1", "k2"]
    for ens_tuning, sf in target_df.groupby("ens_tuning"):
        print(f"Fitting joint distribution for {ens_tuning} ensembles...")
        params = fit_joint(sf["srel_ori"] / 90 * np.pi, sf["osi"])
        print(", ".join(f"{name}={xi:.2g}" for name, xi in zip(names, params)))

        ax = g.axes[0, col_order.index(ens_tuning)]
        plot_joint(
            params, target_df["binned_osi"].cat.categories, ax=ax, colors=hex_colors
        )
    g.set_titles(template="{col_name}")

    if args.out:
        g.savefig(
            args.out / "S2f.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
        )
    plt.show() if args.show else plt.close()


if __name__ == "__main__":
    main()
