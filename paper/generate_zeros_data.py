import argparse
import sys
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("distance", type=float, nargs="+")
    parser.add_argument("se", type=float)
    parser.add_argument("--cell-types", "-c", nargs="+", default=["PYR", "PV", "SST"])
    parser.add_argument("--rel-ori", action="store_true")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()

    distance = pd.IntervalIndex.from_breaks([*args.distance, np.inf], closed="left")
    keys = ["cell_type", "N", "density", "holo_osi", "distance"]
    values = [
        args.cell_types,
        [10],
        ["spreadout", "compact"],
        ["low", "high"],
        distance,
    ]
    if args.rel_ori:
        keys.append("rel_ori")
        values.append([0, 1, 2])
    df = pd.DataFrame([dict(zip(keys, v)) for v in product(*values)])
    for k in ["cell_type", "density", "holo_osi", "distance"]:
        df[k] = df[k].astype("category")
    if args.rel_ori:
        rel_ori = pd.IntervalIndex.from_breaks([0.0, 22.5, 67.5, 90.0], closed="left")
        df["rel_ori"] = pd.Categorical.from_codes(df["rel_ori"], categories=rel_ori)
    df["dr"] = 0.0
    df["dr_se"] = args.se
    for k in ["dr", "dr_se"]:
        df[k] = df[k].astype("float32")

    df.attrs["command"] = " ".join(["python"] + sys.argv)
    if args.out:
        df.to_pickle(args.out)
    else:
        print(df)
        print(df.dtypes)
        print(df.attrs)


if __name__ == "__main__":
    main()

