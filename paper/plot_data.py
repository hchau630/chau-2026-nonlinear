import argparse
from pathlib import Path
from itertools import product

import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from niarb import viz


def plot_yzero(ax):
    ax.axhline(
        0,
        linestyle="--",
        color=rcParams["grid.color"],
        linewidth=rcParams["grid.linewidth"],
    )


def set_titles(g, both=True):
    row_temp = "{row_var}: {row_name}"
    col_temp = "{col_var}: {col_name}"
    temp = f"{row_temp}\n{col_temp}" if both else None
    g.set_titles(template=temp, row_template=row_temp, col_template=col_temp)


def load_ref(filename):
    ref = torch.load(filename, weights_only=True)
    ref = {k: v.squeeze().numpy() for k, v in ref.items()}
    ref = pd.DataFrame(ref).rename(
        columns={"x": "distance", "y": "dr", "yerr": "dr_se"}
    )
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--xy", action="store_true")
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    ref_dir = args.dir / "reference"

    xy = "_xy" if args.xy else ""

    df = pd.read_pickle(args.dir / f"single_cell_space{xy}_data.pkl")
    df["distance"] = pd.IntervalIndex(df["distance"]).mid
    ref = load_ref(ref_dir / "space_resp_1_cell.pt")
    df = pd.concat({"data": df, "ref": ref}, names=["source"]).reset_index(0)
    df = viz.sample_df(df, y="dr", yerr="dr_se")
    g = sns.relplot(
        df, kind="line", x="distance", y="dr", style="source", errorbar="se", height=2
    )
    for ax in g.axes.flat:
        plot_yzero(ax)
    if args.out:
        plt.savefig(args.out / f"single_cell_space{xy}.pdf")
    if args.show:
        plt.show()
    plt.clf()

    df = pd.read_pickle(args.dir / f"ensemble_space_ori{xy}_data.pkl")
    df["distance"] = pd.IntervalIndex(df["distance"]).mid
    refs = {
        ("low", "spreadout"): load_ref(
            ref_dir / "space_resp_10_cell_mean_geq200_untuned.pt"
        ),
        ("low", "compact"): load_ref(
            ref_dir / "space_resp_10_cell_mean_leq200_untuned.pt"
        ),
        ("high", "spreadout"): load_ref(
            ref_dir / "space_resp_10_cell_mean_geq200_cotuned.pt"
        ),
        ("high", "compact"): load_ref(
            ref_dir / "space_resp_10_cell_mean_leq200_cotuned.pt"
        ),
    }
    ref = pd.concat(refs, names=["holo_osi", "density"]).reset_index([0, 1])
    df = pd.concat({"data": df, "ref": ref}, names=["source"]).reset_index(0)
    df = viz.sample_df(df, y="dr", yerr="dr_se")
    g = sns.relplot(
        df,
        kind="line",
        x="distance",
        y="dr",
        col="holo_osi",
        col_order=["high", "low"],
        row="density",
        row_order=["spreadout", "compact"],
        style="source",
        errorbar="se",
        height=2,
    )
    set_titles(g)
    for ax in g.axes.flat:
        plot_yzero(ax)
    if args.out:
        plt.savefig(args.out / f"ensemble_space_ori{xy}.pdf")
    if args.show:
        plt.show()
    plt.clf()

    df = pd.read_pickle(args.dir / f"ensemble_space_v_ori{xy}_data.pkl")
    df["distance"] = pd.IntervalIndex(df["distance"]).mid
    refs = {
        (density, dori): load_ref(
            ref_dir / f"space_resp_10_cell_{density}_cotuned_dori_{dori}.pt"
        )
        for density, dori in product(["spreadout", "compact"], [0, 45, 90])
    }
    ref = pd.concat(refs, names=["density", "rel_ori"]).reset_index([0, 1])
    ref["rel_ori"] = (
        ref["rel_ori"]
        .astype("category")
        .cat.rename_categories(
            {
                0: pd.Interval(0.0, 22.5, closed="left"),
                45: pd.Interval(22.5, 67.5, closed="left"),
                90: pd.Interval(67.5, 90.0, closed="left"),
            }
        )
    )
    df = pd.concat({"data": df, "ref": ref}, names=["source"]).reset_index(0)
    df = viz.sample_df(df, y="dr", yerr="dr_se")
    g = sns.relplot(
        df,
        kind="line",
        x="distance",
        y="dr",
        col="rel_ori",
        row="density",
        row_order=["spreadout", "compact"],
        style="source",
        errorbar="se",
        height=2,
    )
    set_titles(g)
    for ax in g.axes.flat:
        plot_yzero(ax)
    if args.out:
        plt.savefig(args.out / f"ensemble_space_v_ori{xy}.pdf")
    if args.show:
        plt.show()
    plt.clf()


if __name__ == "__main__":
    main()
