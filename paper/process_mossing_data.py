import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from niarb import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--as-tensor", "-t", action="store_true")
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()

    dfs: list[pd.DataFrame] = []
    for filename in args.path.glob("figure_1_contrast/L23/*.csv"):
        data = np.loadtxt(filename, delimiter=",")
        x, y = data[:, 0], data[:, 1]
        y = (y - y[0]) / y[0]
        df = pd.DataFrame({"cell_type": filename.stem, "contrast": x, "drf/rf": y})
        dfs.append(df)
    df = pd.concat(dfs)
    df["cell_type"] = pd.Categorical(
        df["cell_type"], categories=["PYR", "PV", "SST", "VIP"], ordered=True
    )
    df = df.sort_values(by=["cell_type", "contrast"])

    if args.query:
        df = df.query(args.query)

    df = df.reset_index(drop=True)

    if args.as_tensor:
        df = df.pivot(index="contrast", columns="cell_type", values="drf/rf")
        df = torch.from_numpy(df.to_numpy()).float()

    print(df)
    if args.out:
        if isinstance(df, pd.DataFrame):
            io.save_dataframe(df, args.out)
        elif isinstance(df, torch.Tensor):
            torch.save(df, args.out)
        else:
            raise RuntimeError()


if __name__ == "__main__":
    main()
