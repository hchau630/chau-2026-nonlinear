import argparse
from pathlib import Path
from itertools import product
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize, io


def func(x, a0, a1, s0, s1, b):
    return a0 * np.exp(-(x**2) / (2 * s0**2)) - a1 * np.exp(-(x**2) / (2 * s1**2)) + b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()

    data = io.loadmat(args.data)["averaged_rate_field"]
    n_sizes, n_locs = 9, 361
    assert data.shape == (n_sizes * 2, n_locs)  # classical or inverse
    x = np.linspace(-180, 180, n_locs)
    stim_types = ["classical", "inverse"]
    stim_sizes = [5, 15, 25, 35, 45, 55, 65, 75, 85]
    p0 = [0.1, 0.0, 20.0, 20.0]
    # p0 = [0.1, 0.0]
    b0 = np.mean(data[:, 0])
    func_ = partial(func, b=b0)  # fix baseline
    # func_ = partial(func, s0=19.0, s1=20.0, b=b0)  # fix baseline

    df = {}
    for label, y in zip(product(stim_types, stim_sizes), data, strict=True):
        bounds = (0, [(np.max(y) - b0) * 4] * 2 + [50.0] * 2)
        # bounds = (-(np.max(y) - b0) * 4, (np.max(y) - b0) * 4)
        popt, _ = optimize.curve_fit(func_, x, y, p0=p0, bounds=bounds)
        # popt, _ = optimize.curve_fit(func_, x, y, p0=p0)
        popt[:2], popt[2:] = np.round(popt[:2], 2), np.round(popt[2:], 1)
        y_pred = func_(x, *popt)
        df[(*label, "data")] = pd.DataFrame({"x": x, "y": y})
        df[(*label, "fit")] = pd.DataFrame({"x": x, "y": y_pred})
        with np.printoptions(suppress=True):
            print(popt, label)
    df = pd.concat(df, names=["stim_type", "stim_size", "kind"]).reset_index([0, 1, 2])

    g = sns.relplot(
        df.query("x.abs() < 90.1"),
        x="x",
        y="y",
        col="stim_size",
        row="stim_type",
        style="kind",
        kind="line",
        height=2,
    )
    g.set_titles("{col_var}: {col_name}\n{row_var}: {row_name}")
    for ax in g.axes.flat:
        ax.axvline(20.0)
        ax.axvline(-20.0)
        ax.axvline(40.0)
        ax.axvline(-40.0)
    plt.show()


if __name__ == "__main__":
    main()
