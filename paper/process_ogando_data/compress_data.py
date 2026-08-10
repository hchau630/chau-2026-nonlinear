import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--inp", "-i")
    parser.add_argument("--out", "-o", nargs="?", const="compressed.pkl")
    args = parser.parse_args()

    if args.inp is None:
        paths = list(args.data_path.glob("*.csv"))
        if len(paths) > 1:
            raise ValueError(
                f"Multiple CSV files found in {args.data_path}. Please specify one "
                "with --inp."
            )
        args.inp = paths[0].name

    df = pd.read_csv(args.data_path / args.inp)
    print(df.columns)
    df.drop(
        columns=[
            "index",
            "dataset",
            "osi_cat",
            # "activity_BL",
            # "activity_STD",
        ],
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"], format=r"%Y%m%d")

    df = df.astype(
        {
            "holoID_s": "category",
            # "dataset": "category",
            "expType": "category",
            "activity": "float32",
            "activity_0p5to1p5": "float32",
            # "activity_BL": "float32",
            # "activity_STD": "float32",
            # "activity_std_per_trial": "float32",
            "baselineSTD_rawF_concatTrials": "float32",
            "respMinusBaseline_rawF": "float32",
            "respMinusBaseline_rawF_0p5_1p5": "float32",
            "pval_holo": "float32",
            "percentages": "float32",
            "Cell_uID": "category",
            "peak_holo": "float32",
            "peak_cell": "float32",
            "__pval_resp_cell": "float32",
            "__anova_pval_cell": "float32",
            "osis_h": "float32",
            "osis_cell": "float32",
            "maxOri_holo": "int16",
            "maxOri_cell": "int16",
            "ntargs": "int16",
            "mouse_type": "category",
            "cell_type": "category",
            "mouse": "category",
            "date": "category",
            "sessionID": "category",
            "distCT_PSF": "float32",
            # "osi_cat": "category",
            "distCT_um": "float32",
            "distCT_um_lat": "float32",
            "distCT_um_z": "float32",
        },
        copy=False,
    )

    df = df.rename(
        columns={
            "holoID_s": "holo_id",
            "expType": "exp_type",
            "activity": "z_dff",
            "activity_0p5to1p5": "z_dff_windowed",
            # "activity_BL": "z_dff_baseline",
            # "activity_STD": "z_dff_baseline_std",
            # "activity_std_per_trial": "z_dff_normalized",
            "baselineSTD_rawF_concatTrials": "f_baseline_std",
            "respMinusBaseline_rawF": "df",
            "respMinusBaseline_rawF_0p5_1p5": "df_windowed",
            "pval_holo": "holo_resp_pval",
            "Cell_uID": "cell_id",
            "peak_holo": "holo_ori_real",
            "peak_cell": "pref_ori_real",
            "__pval_resp_cell": "__pval_resp",
            "__anova_pval_cell": "vis_resp_pval",
            "osis_h": "holo_osi",
            "osis_cell": "osi",
            "maxOri_holo": "holo_ori",
            "maxOri_cell": "pref_ori",
            "ntargs": "stimmable_ensemble_size",
            "mouse_type": "mouse_group",
            "mouse": "mouse_id",
            "sessionID": "session_id",
            "distCT_PSF": "exclusion_dist_um",
            "distCT_um": "distance",
            "distCT_um_lat": "distance_xy",
            "distCT_um_z": "distance_z",
        },
        copy=False,
    )

    # save compressed data
    if args.out:
        df.to_pickle(args.data_path / args.out)
    else:
        print(df)
        print(df.dtypes)


if __name__ == "__main__":
    main()
