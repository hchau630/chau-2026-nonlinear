import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=Path)
    parser.add_argument("--max-retinotopy", "-R", type=float, default=float("inf"))
    parser.add_argument("--cmf", type=float, default=30.0)
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    df = pd.read_pickle(args.filename)
    assert isinstance(df, pd.DataFrame)
    df["dr"] = df["y"]
    for _, sf in df.groupby("cell_type"):
        y0 = sf.query("stim_type == 'spontaneous'")["y"].mean()
        df.loc[sf.index, "dr"] = sf["y"] - y0
    df = df.query("stim_type != 'spontaneous'").reset_index(drop=True)
    df = df.drop(columns=["y"])
    df["stim_type"] = pd.Categorical(
        df["stim_type"], categories=["center", "iso", "cross"]
    )
    df["cell_type"] = pd.Categorical(
        df["cell_type"], categories=["PYR", "PV", "SST", "VIP", "L4"]
    )
    df["space"] = pd.Categorical(
        df["space"], categories=["center", "middle", "surround"]
    )
    df["ori"] = pd.Categorical(df["ori"], categories=["iso", "ortho"])
    df["dr"] = df["dr"].astype("float32")
    df["distance"] = df["space"].cat.rename_categories(
        pd.IntervalIndex.from_tuples(
            [
                (0.0, 10 * args.cmf),
                (10 * args.cmf, 15 * args.cmf),
                (15 * args.cmf, args.max_retinotopy * args.cmf),
            ],
            closed="left",
        )
    )
    df["rel_ori"] = df["ori"].cat.rename_categories(
        pd.IntervalIndex.from_tuples([(0.0, 45.0), (45.0, 90.0)], closed="left")
    )
    if args.query:
        df = df.query(args.query).reset_index(drop=True)
    df = df.drop(columns=["space", "ori", "space-ori"])

    print(df.to_string(max_rows=100))
    print(df.dtypes)
    for col in ["cell_type", "stim_type", "distance", "rel_ori"]:
        print(f"{col}: {df[col].cat.categories}")

    if args.show:
        g = sns.catplot(
            df,
            x="stim_type",
            y="dr",
            col="space-ori",
            row="cell_type",
            height=1.5,
            aspect=1.1,
            sharey="row",
        )
        g.refline(y=0)
        plt.show()
    if args.out:
        df.attrs["command"] = " ".join(["python"] + sys.argv)
        df.to_pickle(args.out)


if __name__ == "__main__":
    main()
