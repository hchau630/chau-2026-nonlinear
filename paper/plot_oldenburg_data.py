import argparse
from pathlib import Path
from itertools import product
import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from niarb import viz
from mpl_config import FIGSIZE, GRID_COLOR, GRID_WIDTH, set_rcParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--old-dir", type=Path)
    parser.add_argument("--2d", dest="two_d", action="store_true")
    parser.add_argument("--rel-ori", action="store_true")
    parser.add_argument("--out-data-dir", type=Path)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--log-level", "--ll", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(format="%(lineno)d:%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.log_level))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # matplotlib is noisy

    set_rcParams()

    suffix = "_2d" if args.two_d else ""
    cell_table_dir = args.dir / "ExperimentalResults" / "compressedData"
    cell_table_path = cell_table_dir / f"cell_table_250622{suffix}.csv"
    figure_data_dir = args.dir / "ExperimentalResults" / f"figure_data{suffix}"

    df = pd.read_csv(cell_table_path)
    n_cells = df.groupby("expNum")["cellID"].nunique()
    print(n_cells.describe())

    df["density"] = pd.cut(
        df["cellEnsMeaD"],
        bins=[0, 200, np.inf],
        labels=["compact", "spreadout"],
        right=False,
        ordered=False,
    )
    df["holo_osi"] = pd.NA
    df.loc[(df["cellMeanEnsOSI"] < 0.5) & (df["cellEnsOSI"] < 0.3), "holo_osi"] = "low"
    df.loc[(df["cellMeanEnsOSI"] > 0.5) & (df["cellEnsOSI"] > 0.7), "holo_osi"] = "high"
    df["distance"] = pd.cut(df["cellDist"], bins=np.arange(15, 301, 15.0), right=False)
    df = df.query("offTarget == 0")

    baseline = (
        df.query("~holo_osi.isna()").groupby(["ensNum", "expNum"])["baselineEst"].mean()
    )
    # print(baseline.to_string(max_rows=100))
    print(baseline.mean(), baseline.sem())

    if args.rel_ori:
        df = df.query("visP < 0.05 and cellOSI > 0.25 and holo_osi == 'high'")
        df = df.loc[pd.IntervalIndex(df["distance"]).right < 151].reset_index(drop=True)
        df["distance"] = df["distance"].cat.remove_unused_categories()
        df = df.rename(columns={"cellOrisDiff": "rel_ori"})
        df_dist = df.groupby(
            ["ensNum", "expNum", "density", "rel_ori", "distance"],
            as_index=False,
            observed=True,
        )["dff"].mean()
        out = (
            df_dist.groupby(["density", "rel_ori", "distance"], observed=True)["dff"]
            .agg(dr="mean", dr_se="sem")
            .reset_index()
        )
        out1 = out.copy()
    else:
        df_dist = df.groupby(
            ["ensNum", "expNum", "density", "holo_osi", "distance"],
            as_index=False,
            observed=True,
        )["dff"].mean()
        df_mean = (
            df.query("cellDist > 15 and cellDist < 75")
            .groupby(
                ["ensNum", "expNum", "density", "holo_osi"],
                as_index=False,
                observed=True,
            )["dff"]
            .mean()
        )
        out = (
            df_dist.groupby(["density", "holo_osi", "distance"], observed=True)["dff"]
            .agg(dr="mean", dr_se="sem")
            .reset_index()
        )
        out1 = out.loc[pd.IntervalIndex(out["distance"]).right < 151].reset_index(
            drop=True
        )
        out1["distance"] = out1["distance"].cat.remove_unused_categories()

    # Verify correctness against their figure data
    if args.rel_ori:
        expected = {}
        for density, rel_ori in product(["compact", "spreadout"], [0, 45, 90]):
            expected[(density, float(rel_ori))] = pd.DataFrame(
                np.loadtxt(
                    figure_data_dir / "fig_4b" / f"{density}_{rel_ori}.csv",
                    delimiter=",",
                ),
                columns=["distance", "dr", "dr_se"],
            )
        expected = (
            pd.concat(expected, names=["density", "rel_ori"])
            .reset_index([0, 1])
            .reset_index(drop=True)
        )
    else:
        expected = {}
        for density, holo_osi in product(["compact", "spreadout"], ["high", "low"]):
            holo_osi_ = "untuned" if holo_osi == "low" else "cotuned"
            expected[(density, holo_osi)] = pd.DataFrame(
                np.loadtxt(
                    figure_data_dir / "fig_4a" / f"{density}_{holo_osi_}.csv",
                    delimiter=",",
                ),
                columns=["distance", "dr", "dr_se"],
            )
        expected = (
            pd.concat(expected, names=["density", "holo_osi"])
            .reset_index([0, 1])
            .reset_index(drop=True)
        )
    expected["density"] = expected["density"].astype("category")
    expected["distance"] = pd.cut(
        expected["distance"], bins=np.arange(15, 151, 15.0), right=False
    )
    pd.testing.assert_frame_equal(out1, expected)

    # Some extra processing for fitting
    if args.old_dir or args.out_data_dir:
        out2 = out.copy()
        out2["cell_type"] = pd.Categorical(["PYR"] * len(out2))
        out2["N"] = 10
        out2[["dr", "dr_se"]] = out2[["dr", "dr_se"]].astype(np.float32)

        suffix = "_xy" if args.two_d else ""
        suffix = f"v_ori_osi{suffix}" if args.rel_ori else f"ori{suffix}"
        if args.rel_ori:
            out2["holo_osi"] = "high"
            out2["osi"] = pd.Interval(0.25, 1.0, closed="left")
            out2["osi"] = out2["osi"].astype("category")
            out2["rel_ori"] = out2["rel_ori"].astype("category")
            out2["rel_ori"] = out2["rel_ori"].cat.rename_categories(
                {
                    0.0: pd.Interval(0.0, 22.5, closed="left"),
                    45.0: pd.Interval(22.5, 67.5, closed="left"),
                    90.0: pd.Interval(67.5, 90.0, closed="left"),
                }
            )

    suffix = "_xy" if args.two_d else ""
    suffix = f"v_ori_osi{suffix}" if args.rel_ori else f"ori{suffix}"
    # Validate against old data
    if args.old_dir:
        if args.rel_ori:
            expected = pd.read_pickle(
                args.old_dir / f"ensemble_space_{suffix}_data_old.pkl"
            )
            expected = expected.sort_values(
                ["density", "rel_ori", "distance"]
            ).reset_index(drop=True)

            # Unfortunately, the old data used a different method of computing preferred
            # orientation (max(ori) instead of max(dir)), so the dr and dr_se values are
            # slightly different. We therefore simply verify that out2 correlates well
            # with expected visually.
            plt.scatter(out2["dr"], expected["dr"])
            plt.show()
            plt.scatter(out2["dr_se"], expected["dr_se"])
            plt.show()
            expected["dr"] = out2["dr"]
            expected["dr_se"] = out2["dr_se"]
        else:
            expected = pd.read_pickle(
                args.old_dir / f"ensemble_space_{suffix}_data_old.pkl"
            )
            expected = expected.sort_values(
                ["density", "holo_osi", "distance"]
            ).reset_index(drop=True)

        expected["distance"] = expected["distance"].cat.as_ordered()
        pd.testing.assert_frame_equal(
            out2, expected, check_like=True, check_exact=False, rtol=5e-3, atol=1e-5
        )

    # Save processed data for fitting
    if args.out_data_dir:
        out2.attrs["command"] = " ".join(["python"] + sys.argv)
        out2.to_pickle(args.out_data_dir / f"ensemble_space_{suffix}_data.pkl")

    # Plot the data
    mapping = {
        "distance": "Distance (µm)",
        "dff": "Response (ΔF/F)",
        "holo_osi": "Ens. Tuning",
        # "density": "Spatial Density",
        "density": "",
    }
    df_dist["density"] = df_dist["density"].cat.rename_categories(
        {"spreadout": "Diffuse", "compact": "Compact"}
    )
    if args.rel_ori:
        df_dist["rel_ori"] = df_dist["rel_ori"].astype("category")
        logger.info("Plotting figure S2d...")
        g = viz.figplot(
            df_dist,
            "relplot",
            kind="line",
            x="distance",
            y="dff",
            hue="rel_ori",
            col="density",
            col_order=["Diffuse", "Compact"],
            palette=["#93B8E2", "#46BC99", "#F69CA0"],
            height=FIGSIZE[1],
            aspect=0.8,
            title_verbosity=0,
            errorbar="se",
            statannot=True,
            mapping=mapping,
        )
        g.refline(y=0, linestyle="-", color=GRID_COLOR, linewidth=GRID_WIDTH)
        if args.out:
            g.savefig(
                args.out / "S2d.pdf",
                metadata={"Subject": " ".join(["python"] + sys.argv)},
            )
        plt.show() if args.show else plt.close()
        return

    df_dist["holo_osi"] = df_dist["holo_osi"].replace(
        {"low": "Untuned", "high": "Cotuned"}
    )
    df_mean["density"] = df_mean["density"].cat.rename_categories(
        {"spreadout": "Diffuse", "compact": "Compact"}
    )
    df_mean["holo_osi"] = df_mean["holo_osi"].replace(
        {"low": "Untuned", "high": "Cotuned"}
    )

    logger.info("Plotting figure S2c...")
    g = viz.figplot(
        df_dist,
        "relplot",
        x="distance",
        y="dff",
        hue="holo_osi",
        col="density",
        kind="line",
        col_order=["Diffuse", "Compact"],
        height=FIGSIZE[1],
        aspect=0.9,
        title_verbosity=0,
        errorbar="se",
        statannot=True,
        mapping=mapping,
    )
    g.refline(y=0, linestyle="-", color=GRID_COLOR, linewidth=GRID_WIDTH)
    if args.out:
        g.savefig(
            args.out / "S2c.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
        )
    plt.show() if args.show else plt.close()

    df_dist = df_dist.loc[pd.IntervalIndex(df_dist["distance"]).right < 151].copy()
    logger.info("Plotting figure 2d...")
    g = viz.figplot(
        df_dist,
        "relplot",
        x="distance",
        y="dff",
        hue="holo_osi",
        col="density",
        kind="line",
        col_order=["Diffuse", "Compact"],
        height=FIGSIZE[1],
        aspect=0.8,
        title_verbosity=0,
        errorbar="se",
        statannot=True,
        mapping=mapping,
    )
    g.refline(y=0, linestyle="-", color=GRID_COLOR, linewidth=GRID_WIDTH)
    if args.out:
        g.savefig(
            args.out / "2d.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
        )
    plt.show() if args.show else plt.close()

    logger.info("Plotting figure 2e...")
    for density, sf in df_mean.groupby("density", observed=True):
        out = stats.ttest_ind(
            sf.query("holo_osi == 'Untuned'")["dff"],
            sf.query("holo_osi == 'Cotuned'")["dff"],
            alternative="greater",
            equal_var=False,
            permutations=999999,
        )
        print(f"{density=}: {out}")
    for holo_osi, sf in df_mean.groupby("holo_osi", observed=True):
        out = stats.ttest_ind(
            sf.query("density == 'Diffuse'")["dff"],
            sf.query("density == 'Compact'")["dff"],
            alternative="greater",
            equal_var=False,
            permutations=999999,
        )
        print(f"{holo_osi=}: {out}")
    g = viz.figplot(
        df_mean,
        "catplot",
        x="density",
        y="dff",
        hue="holo_osi",
        kind="bar",
        order=["Diffuse", "Compact"],
        hue_order=["Untuned", "Cotuned"],
        height=FIGSIZE[1],
        aspect=1.3,
        title_verbosity=0,
        errorbar="se",
        mapping=mapping,
    )
    # g.refline(y=0, linestyle="-", color=GRID_COLOR, linewidth=GRID_WIDTH)
    # g.tick_params(axis="x", rotation=30)

    # display x-axis on top at y = 0
    g.tick_params(axis="x", bottom=False, labeltop=True, labelbottom=False)
    g.despine(bottom=True, top=False)
    for ax in g.axes.flat:
        ax.spines["top"].set_position("zero")

    g.figure.suptitle("Nearby response (< 75 μm)", y=0.86)
    g.tight_layout()
    if args.out:
        g.savefig(
            args.out / "2e.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
        )
    plt.show() if args.show else plt.close()


if __name__ == "__main__":
    main()
