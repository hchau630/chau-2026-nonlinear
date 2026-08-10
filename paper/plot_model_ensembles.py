from collections.abc import Callable, Sequence
import operator
import warnings
from pathlib import Path
import argparse
from functools import partial
from math import prod
import sys

import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns

from niarb.distributions import RectLinePicking
from mpl_config import FIGSIZE, RECT, set_rcParams

HISTPLOT_KWARGS = dict(stat="density", element="step", fill=False)
OPERATOR_MAPPING = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def dori_pdf(dori, osi, k0, k1, k2, scale=90 / np.pi):
    dori = dori / scale
    kappa = np.exp(k0 + k1 * osi + k2 * osi**2)
    out = stats.vonmises.pdf(dori, kappa)
    # kappa = k0 + k1 * osi + k2 * osi**2
    # out = np.where(
    #     kappa >= 0,
    #     stats.vonmises.pdf(dori, np.abs(kappa)),
    #     stats.vonmises.pdf(dori, np.abs(kappa), loc=np.pi),
    # )
    return out / scale


def rejection_sampling(
    pdf: Callable[[*tuple[np.ndarray, ...]], np.ndarray],
    M: float,
    proposals: Sequence[stats.rv_continuous],
    shape: Sequence[int] = (),
    batch_shape: Sequence[int] = (),
):
    count = 0
    out = []
    while count < prod(shape):
        samples = [p.rvs(size=batch_shape) for p in proposals]
        U = np.random.rand(*batch_shape)
        W = pdf(*samples) / np.prod(
            [p.pdf(s) for s, p in zip(samples, proposals)], axis=0
        )
        if (W > M).any():
            raise ValueError(
                f"Max likelihood ratio {W.max()=} exceeds ratio upper bound {M=}."
            )
        if (M > 2 * W).any():
            warnings.warn(
                f"Max likelihood ratio {W.max()=} is much smaller than ratio upper "
                f"bound {M=}. Consider reducing M for more efficient sampling."
            )
        accept = W >= M * U
        count += np.count_nonzero(accept)
        out.append(np.stack(samples, axis=-1)[accept, ...])
    out = np.concatenate(out)[: prod(shape)].reshape((*shape, len(proposals)))
    return np.unstack(out, axis=-1)


def dori_osi_pdf(a, b, k0, k1, k2, dori, osi):
    return stats.beta.pdf(osi, a, b) * dori_pdf(dori, osi, k0, k1, k2)


def plot_ensemble_space(a, b, m, n, mean_dist_cond, eps=1e-5):
    L = np.array([a, b])
    samples = np.random.rand(m, n, len(L)) * L
    distances = np.linalg.vector_norm(
        samples[:, :, None, :] - samples[:, None, :, :], axis=-1
    )
    distances = distances[(slice(None), *np.triu_indices(n, k=1))]
    mean_distances = np.mean(distances, axis=-1)
    if mean_dist_cond:
        op, threshold = mean_dist_cond[0], float(mean_dist_cond[1:])
        op = OPERATOR_MAPPING[op]
        mask = op(mean_distances, threshold)
        distances = distances[mask]
        mean_distances = mean_distances[mask]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    df = pd.DataFrame({"distance": distances.flatten()})
    sns.histplot(df, x="distance", stat="density", ax=ax)
    x = torch.linspace(0, np.linalg.vector_norm(L) - eps, 100)
    ax.plot(x, RectLinePicking(a, b).log_prob(x).exp())
    plt.show()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    df = pd.DataFrame({"mean_distance": mean_distances})
    sns.histplot(df, x="mean_distance", stat="density", ax=ax)
    plt.show()


def plot_ensemble_ori(a, b, k0, k1, k2, max_ens_osi, m, n, out, show, eps=1e-2):
    pdf = partial(dori_osi_pdf, a, b, k0, k1, k2)
    proposals = [stats.uniform(loc=-90, scale=180), stats.beta(a, b)]
    dori, osi = np.ogrid[-90:90:100j, 0:1:100j]
    M = 180 * dori_pdf(dori, osi, k0, k1, k2).max() * (1 + eps)
    dori, osi = rejection_sampling(
        pdf, M, proposals, shape=(m, n), batch_shape=(m * n,)
    )
    ens_idx, cell_idx = np.mgrid[0:m:1, 0:n:1]
    df = pd.DataFrame(
        {
            "ens_idx": ens_idx.flatten(),
            "cell_idx": cell_idx.flatten(),
            "dori_raw": dori.flatten(),
            "osi": osi.flatten(),
        }
    )
    df["mean_osi"] = df.groupby("ens_idx")["osi"].transform("mean")
    df["binned_osi"] = pd.cut(df["osi"], bins=[0, 0.25, 0.5, 0.75, 1])
    df["binned_osi_val"] = pd.IntervalIndex(df["binned_osi"]).mid

    # # calculate ensemble preferred orientation and OSI
    # theta = np.array([-90, -45, 0, 45])
    # tuning_curves = 1 + osi[..., None] * np.cos((theta - dori[..., None]) / 90 * np.pi)
    # ens_tuning_curves = np.mean(tuning_curves, axis=1)  # (m, 4)
    # ens_ori_idx = np.argmax(ens_tuning_curves, axis=-1, keepdims=True)  # (m, 1)
    # ens_ori = theta[ens_ori_idx]
    # po = np.take_along_axis(ens_tuning_curves, ens_ori_idx, axis=-1)  # (m, 1)
    # oo = np.take_along_axis(ens_tuning_curves, (ens_ori_idx + 2) % 4, axis=-1)  # (m, 1)
    # ens_osi = (po - oo) / (po + oo)
    # df["ens_ori"] = np.broadcast_to(ens_ori, (m, n)).flatten()
    # df["ens_osi"] = np.broadcast_to(ens_osi, (m, n)).flatten()

    # calculate ensemble preferred orientation and OSI
    df["z"] = df["osi"] * np.exp(df["dori_raw"] / 90 * np.pi * 1j)
    df["ens_z"] = df.groupby("ens_idx")["z"].transform("mean")
    df["ens_ori"] = np.angle(df["ens_z"]) / np.pi * 90
    df["ens_osi"] = np.abs(df["ens_z"])
    # mimic their analysis where ens_ori is discretized
    df["ens_ori"] = pd.cut(
        df["ens_ori"],
        bins=np.linspace(-112.5, 112.5, 6),
        labels=[-90, -45, 0, 45, -90],
        ordered=False,
    ).astype(float)

    # calculate delta ori
    df["dori"] = (df["dori_raw"] - df["ens_ori"] + 90) % 180 - 90

    # # Plot histograms of mean OSI and ensemble OSI
    # grouped_df = df.groupby("ens_idx", as_index=False)[["mean_osi", "ens_osi"]].mean()

    # fig = plt.figure(figsize=FIGSIZE)
    # ax = fig.add_axes(RECT)
    # sns.histplot(grouped_df, x="mean_osi", ax=ax, **HISTPLOT_KWARGS)
    # plt.show()

    # fig = plt.figure(figsize=FIGSIZE)
    # ax = fig.add_axes(RECT)
    # sns.histplot(grouped_df, x="ens_osi", ax=ax, **HISTPLOT_KWARGS)
    # plt.show()

    # Plot histogram of delta ori for different OSI bins
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    sf = df.query(f"ens_osi <= {max_ens_osi}").copy()
    sf = sf.rename(columns={"dori": "Δ pref. ori. (°)"})
    sns.histplot(
        sf,
        x="Δ pref. ori. (°)",
        hue="binned_osi_val",
        legend=False,
        ax=ax,
        **HISTPLOT_KWARGS,
    )

    # obtain colors used by seaborn for hue, code with help from ChatGPT-5.4
    values = df["binned_osi_val"][~df["binned_osi_val"].isna()].unique()
    cmap = sns.color_palette("ch:", as_cmap=True)  # seaborn's default numeric palette
    norm = colors.Normalize(vmin=min(values), vmax=max(values))
    hex_colors = [colors.to_hex(cmap(norm(v))) for v in sorted(values)]

    # plot probability density functions
    x = np.linspace(-90, 90, 100)
    for osi_bin, c in zip(df["binned_osi"].cat.categories, hex_colors):
        ax.plot(x, dori_osi_pdf(a, b, k0, k1, k2, x, osi_bin.mid) * osi_bin.length, c=c)
    ax.set_xticks([-90, 0, 90])
    if out:
        plt.savefig(out, metadata={"Subject": " ".join(["python"] + sys.argv)})
    plt.show() if show else plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["space", "ori"])
    parser.add_argument("-a", type=float)
    parser.add_argument("-b", type=float)
    parser.add_argument("-k", type=float, nargs=3)
    parser.add_argument("--mean-dist-cond", type=str)
    parser.add_argument("--max-ens-osi", type=float, default=1.0)
    parser.add_argument("-m", type=int, default=100000)
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    set_rcParams()

    if args.mode == "space":
        plot_ensemble_space(args.a, args.b, args.m, args.n, args.mean_dist_cond)
    else:
        plot_ensemble_ori(
            args.a,
            args.b,
            *args.k,
            args.max_ens_osi,
            args.m,
            args.n,
            args.out,
            args.show,
        )


if __name__ == "__main__":
    main()
