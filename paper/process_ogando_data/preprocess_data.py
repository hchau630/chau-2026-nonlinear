import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--ens-sizes", nargs="+", type=int, default=[0, 10, 30, 50])
    parser.add_argument("--min-dist", type=float, default=30.0)
    parser.add_argument("--max-dist", type=float, default=181.0)
    parser.add_argument("--2d", dest="two_d", action="store_true")
    parser.add_argument("--windowed", "-w", action="store_true")
    parser.add_argument("--inp", "-i", default="normalized")
    parser.add_argument("--out", "-o", nargs="?", const="preprocessed.pkl")
    args = parser.parse_args()

    dfs = {}
    for path in (args.data_path / args.inp).glob("*.pkl"):
        dfs[path.stem] = pd.read_pickle(path)

    # # convert x, y, z from pixels to microns using conversion provided Mora in her email
    # units = {"x": 1.78, "y": 1.76, "z": 1.8}
    # for k, v in units.items():
    #     dfs["cell"][k] = dfs["cell"][k] * v

    # # parse session dates
    # dfs["session"]["date"] = pd.to_datetime(dfs["session"]["date"], format="%Y%m%d")
    dfs["session"] = dfs["session"].sort_values("date")

    # # create a cell_type column for convenience
    # cell_df = dfs["cell"].merge(dfs["mouse"], how="left")
    # ct = pd.Series(dtype=str, index=cell_df.index)
    # ct[cell_df.eval('mouse_group == "PV" and is_red_cell')] = "PV"
    # ct[cell_df.eval('mouse_group == "PV" and not is_red_cell')] = "non-PV"
    # ct[cell_df.eval('mouse_group == "SST" and is_red_cell')] = "SST"
    # ct[cell_df.eval('mouse_group == "SST" and not is_red_cell')] = "non-SST"
    # dfs["cell"]["cell_type"] = ct.astype("category")
    dfs["cell"]["cell_type"] = dfs["cell"]["cell_type"].cat.rename_categories(
        {"PC": "PYR"}
    )

    # # identify stimmable targets
    # dfs["cell_holo"]["is_stimmable_target"] = dfs["cell_holo"]["is_target"] & (
    #     dfs["cell_holo"]["holo_resp_pval"] < 0.05
    # )

    # # check that is_stimmable_target is consistent with stimmable_ensemble_size
    # s1 = dfs["cell_holo"].groupby("holo_id", observed=True)["is_stimmable_target"]
    # s1 = s1.sum().astype("int16").sort_index()
    # s2 = dfs["holo"].set_index("holo_id")["stimmable_ensemble_size"].sort_index()
    # print(s1[s1 != s2], s2[s1 != s2])  # should only have 1 error
    # # pd.testing.assert_series_equal(s1, s2, check_names=False)

    # # compute min distance to any target location for each cell-holo combination
    # cell_holo = dfs["cell_holo"].merge(dfs["cell"], how="left", on="cell_id")
    # targets = cell_holo.query("is_stimmable_target")
    # cols = ["cell_id", "holo_id", "x", "y", "z"]
    # cell_targets = cell_holo[cols].merge(
    #     targets[cols], on="holo_id", suffixes=("_cell", "_target")
    # )
    # cell_targets["xy_dist"] = cell_targets.eval(
    #     "((x_target - x_cell)**2 + (y_target - y_cell)**2)**0.5"
    # )
    # cell_targets["z_dist"] = cell_targets.eval("((z_target - z_cell)**2)**0.5")
    # cell_targets["xyz_dist"] = cell_targets.eval(
    #     "((x_target - x_cell)**2 + (y_target - y_cell)**2 + (z_target - z_cell)**2)**0.5"
    # )

    # stats = cell_targets.groupby(
    #     ["cell_id_cell", "holo_id"], as_index=False, observed=True
    # ).agg(
    #     min_cell_tg_xy_dist=("xy_dist", "min"),
    #     min_cell_tg_z_dist=("z_dist", "min"),
    #     min_cell_tg_xyz_dist=("xyz_dist", "min"),
    #     nearest_tg_idx=("xyz_dist", "idxmin"),
    # )
    # stats["nearest_tg_cell_id"] = cell_targets.loc[stats["nearest_tg_idx"]][
    #     "cell_id_target"
    # ].reset_index(drop=True)
    # stats = stats.rename(columns={"cell_id_cell": "cell_id"}).drop(
    #     columns="nearest_tg_idx"
    # )
    # dfs["cell_holo"] = dfs["cell_holo"].merge(stats, how="left")

    # # replace old holo_osi and holo_ori columns with new ones
    # df = dfs["holo"]
    # df = df.drop(columns=["holo_osi", "holo_ori"])
    # df["holo_osi"] = df["stim_input_osi"]
    # df["holo_ori"] = df["stim_input_pref_ori_real"]
    # df = df.drop(
    #     columns=["stim_input_osi", "stim_input_pref_ori", "stim_input_pref_ori_real"]
    # )
    # dfs["holo"] = df

    # # replace old osi, pref_ori, and pref_ori_real columns with new ones
    # df = dfs["cell"]
    # df = df.drop(columns=["osi", "pref_ori", "pref_ori_real"])
    # df["osi"] = df["new_osi"]
    # df["pref_ori"] = df["new_pref_ori"]
    # df["pref_ori_real"] = df["new_pref_ori_real"]
    # df = df.drop(columns=["new_osi", "new_pref_ori", "new_pref_ori_real"])
    # dfs["cell"] = df

    # compute rel_ori and rel_ori_real
    df = dfs["cell_holo"].merge(dfs["cell"], how="left").merge(dfs["holo"], how="left")
    df["rel_ori_real"] = (df["pref_ori_real"] - df["holo_ori_real"]) % 180
    bins = [-1e5, (0 + 45) / 2, (45 + 90) / 2, (90 + 135) / 2, (135 + 180) / 2, 180]
    df["rel_ori"] = pd.cut(
        df["rel_ori_real"], bins=bins, labels=[0, 45, 90, 135, 0], ordered=False
    )
    dfs["cell_holo"] = dfs["cell_holo"].merge(
        df[["cell_id", "holo_id", "rel_ori", "rel_ori_real"]], how="left"
    )

    # # compute holo OSI statistics
    # cell_holo = dfs["cell_holo"].merge(dfs["cell"], how="left", on="cell_id")
    # targets = cell_holo.query("is_stimmable_target")

    # dfs["holo"] = dfs["holo"].merge(
    #     targets.groupby("holo_id", observed=True, as_index=False).agg(
    #         mean_osi=("osi", "mean"),
    #         min_osi=("osi", "min"),
    #         max_osi=("osi", "max"),
    #     ),
    #     how="left",
    # )

    # # compute average z_dff
    # dfs["resp"]["z_dff"] = dfs["resp"].eval("(z_dff1 + z_dff2) / 2")
    # dfs["cell_holo"]["z_dff_normalized"] = (
    #     dfs["cell_holo"]["z_dff"] / dfs["cell_holo"]["z_dff_baseline_std"]
    # )
    if args.windowed:
        dfs["cell_holo"]["dr"] = (
            dfs["cell_holo"]["df_windowed"] / dfs["cell_holo"]["f_baseline_std"]
        )
    else:
        dfs["cell_holo"]["dr"] = (
            dfs["cell_holo"]["df"] / dfs["cell_holo"]["f_baseline_std"]
        )

    if args.two_d:
        dfs["cell_holo"]["distance"] = dfs["cell_holo"]["distance_xy"]

    df = preprocess(dfs, args)

    # save preprocessed dataframes
    if args.out:
        df.to_pickle(args.data_path / args.out)
        # out = args.data_path / args.out
        # out.mkdir(exist_ok=True)
        # for k, v in dfs.items():
        # v.reset_index(drop=True).to_pickle(out / f"{k}.pkl")
    else:
        print(df)
        # for k, v in dfs.items():
        # print(f"{k}:\n{v}\n")


def preprocess(dfs, args):
    df = dfs["cell_holo"].query("exclusion_dist_um > 30")
    df = df.merge(dfs["cell"]).merge(dfs["holo"]).reset_index(drop=True)

    print(f"Number of ensembles: {df['holo_id'].nunique()}.")
    print(f"Number of cells: {df['cell_id'].nunique()}.")
    print(
        f"Number of cells by cell type:\n{df.groupby('cell_type', observed=True)['cell_id'].nunique()}."
    )
    rename = {"stimmable_ensemble_size": "N"}
    # if args.d == 3:
    #     rename["min_cell_tg_xyz_dist"] = "distance"
    # elif args.d == 2:
    #     rename["min_cell_tg_xy_dist"] = "distance"
    #     rename["min_cell_tg_z_dist"] = "z_distance"
    # else:
    #     raise ValueError(f"d must be either 2 or 3, but got {args.d=}.")
    df.rename(columns=rename, inplace=True)

    df["N"] = df["N"].astype(int)
    df["_holo_osi"] = df["holo_osi"].copy()
    df["holo_osi"] = pd.cut(df["_holo_osi"], bins=2, right=False)
    df["holo_osi"] = df["holo_osi"].cat.rename_categories(["low", "high"])

    df["_N"] = df["N"].copy()
    df["N"] = pd.cut(df["N"], bins=args.ens_sizes, right=False)
    df["_distance"] = df["distance"].copy()
    df["distance"] = pd.cut(
        df["distance"],
        bins=np.arange(args.min_dist, min(args.max_dist, df["_distance"].max()), 30),
        right=False,
    )
    df.loc[df["rel_ori"] == 135, "rel_ori"] = 45
    if isinstance(df["rel_ori"].dtype, pd.CategoricalDtype):
        df["rel_ori"] = df["rel_ori"].cat.remove_unused_categories()
    else:
        df["rel_ori"] = df["rel_ori"].astype("category")

    return df


if __name__ == "__main__":
    main()
