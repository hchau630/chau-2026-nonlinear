import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from niarb import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--index", "-i", type=int, default=0)
    args = parser.parse_args()

    data = pd.read_pickle(args.data)
    run = pd.read_pickle(io.iterdir(args.runs, pattern="*.pkl", indices=args.index))
    assert isinstance(run, pd.DataFrame)
    assert isinstance(data, pd.DataFrame)
    run["distance"] = pd.cut(
        run["distance"], bins=np.arange(15, 151, 15.0), right=False
    )
    run = run[[c for c in data.columns if c in run.columns]]
    run = run.groupby(
        ["N", "cell_type", "density", "holo_osi", "distance"],
        observed=True,
        as_index=False,
    )["dr"].mean()
    df = data.merge(
        run, on=["N", "cell_type", "density", "holo_osi", "distance"], validate="1:1"
    )
    w, x, y = df["dr_se"] ** -2, df["dr_x"], df["dr_y"]
    res = np.sum(w * (x - y) ** 2)
    var = np.sum(w * (x - x.mean()) ** 2)
    print(f"R^2: {1 - res / var:.3f}")


if __name__ == "__main__":
    main()
