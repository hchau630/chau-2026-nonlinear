import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pandas import DataFrame


def normalize(
    df: DataFrame,
    primary_keys: Sequence[str],
    foreign_keys: Sequence[str] = (),
    other_keys: Sequence[str] = (),
) -> tuple[DataFrame, DataFrame]:
    dependent_keys = [*foreign_keys, *other_keys]

    # check that primary_keys uniquely determines foreign_keys and other_keys
    g = df.groupby(primary_keys, observed=True)
    for col in dependent_keys:
        # set dropna=False to ensure there it has 1 unique value including NaNs
        if not (g[col].nunique(dropna=False) == 1).all():
            raise ValueError(f"{primary_keys} does not uniquely determine {col}")

    new_df = g[[*primary_keys, *dependent_keys]].nth[0]

    # remove the above columns from df since they are now redundant
    df = df.drop(columns=other_keys)

    return new_df, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--inp", "-i", default="compressed.pkl")
    parser.add_argument("--out", "-o", nargs="?", const="normalized")
    args = parser.parse_args()

    df = pd.read_pickle(args.data_path / args.inp)

    # normalize resp_df
    dfs = {}
    configs = {
        "mouse": {
            "primary_keys": ["mouse_id"],
            "other_keys": ["mouse_group"],
        },
        "session": {
            "primary_keys": ["session_id"],
            "foreign_keys": ["mouse_id"],
            "other_keys": ["date"],
        },
        "holo": {
            "primary_keys": ["holo_id"],
            "foreign_keys": ["mouse_id", "session_id"],
            "other_keys": [
                "exp_type",
                "stimmable_ensemble_size",
                "holo_osi",
                "holo_ori",
                "holo_ori_real",
                "percentages",
            ],
        },
        "cell": {
            "primary_keys": ["cell_id"],
            "foreign_keys": ["mouse_id", "session_id"],
            "other_keys": [
                "cell_type",
                "osi",
                "pref_ori",
                "pref_ori_real",
                "vis_resp_pval",
                "__pval_resp",
            ],
        },
        "cell_holo": {
            "primary_keys": ["cell_id", "holo_id"],
            "foreign_keys": ["mouse_id", "session_id"],
            "other_keys": [
                "holo_resp_pval",
                "exclusion_dist_um",
                "distance",
                "distance_xy",
                "distance_z",
                "z_dff",
                "z_dff_windowed",
                # "z_dff_baseline",
                # "z_dff_baseline_std",
                # "z_dff_normalized",
                "f_baseline_std",
                "df",
                "df_windowed",
            ],
        },
    }
    for name, config in configs.items():
        dfs[name], df = normalize(df, **config)

    if set(df.columns) != {"mouse_id", "session_id", "holo_id", "cell_id"}:
        raise ValueError(f"df has unexpected columns: {df.columns=}")

    # save normalized dataframes
    if args.out:
        out = args.data_path / args.out
        out.mkdir(exist_ok=True)
        for k, v in dfs.items():
            v.reset_index(drop=True).to_pickle(out / f"{k}.pkl")
    else:
        for k, v in dfs.items():
            print(f"{k}:\n{v}\n")


if __name__ == "__main__":
    main()
