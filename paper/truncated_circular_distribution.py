from itertools import product

import numpy as np
from scipy import stats, special
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def truncated_von_mises(n, kappa, theta):
    if n == "inf":
        return stats.vonmises.pdf(theta, kappa)
    summed = sum(special.iv(k, kappa) * np.cos(k * theta) for k in range(1, n))
    return (1 + 2 / special.i0(kappa) * summed) / (2 * np.pi)


def cos_tuning(kappa, theta):
    return (1 + 2 * kappa * np.cos(theta)) / (2 * np.pi)


def main():
    n = [2, 3, 4, 8, 16]
    # kappa = [0, 0.5, 1, 2, 4, 8]
    # kappa = [0, 0.5, 1, 2, 3]
    kappa = [0, -0.5, -1, -2, -3]
    # kappa = [0.5, 1, 2, 4, 8, 16, 32]
    # for k in kappa:
    #     x = list(range(1, 10))
    #     y = [(special.iv(i, k) / special.i0(k)).item() for i in x]
    #     plt.plot(x, y, label=f"k={k}")
    # plt.legend()
    # plt.show()
    theta = np.linspace(-np.pi, np.pi, 100)
    dfs: dict[tuple[str, str], pd.DataFrame] = {}
    for param in product(n, kappa):
        prob = truncated_von_mises(*param, theta)
        dfs[param] = pd.DataFrame({"theta": theta, "prob": prob})
    df = pd.concat(dfs, names=["n", "kappa"]).reset_index(level=[0, 1])
    df = df.reset_index(drop=True)
    # g = sns.relplot(df, x="theta", y="prob", hue="n", col="kappa", kind="line", height=2)
    g = sns.relplot(
        df, x="theta", y="prob", col="n", hue="kappa", kind="line", height=2
    )
    g.refline(y=0)
    for ax in g.axes.flat:
        ax.plot(theta, cos_tuning(0.5, theta), ls="--", color="black")
    plt.show()


if __name__ == "__main__":
    main()
