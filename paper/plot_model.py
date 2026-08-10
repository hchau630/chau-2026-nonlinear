import argparse
from pathlib import Path
import sys
import math

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
from hsluv import hsluv_to_rgb

from niarb import neurons
from mpl_config import set_rcParams, CM, AXSIZE, GRID_WIDTH

FIGSIZE = (2.7 * CM, 2.7 * CM)
CBAR_FIGSIZE = (2.7 * CM, 0.35 * CM)
CBAR_AXSIZE = (AXSIZE[0], 0.25 * CM)
RECT = (
    0.4 * CM / FIGSIZE[0],
    0.4 * CM / FIGSIZE[1],
    AXSIZE[0] / FIGSIZE[0],
    AXSIZE[1] / FIGSIZE[1],
)
CBAR_RECT = (
    0.4 * CM / CBAR_FIGSIZE[0],
    0.05 * CM / CBAR_FIGSIZE[1],
    CBAR_AXSIZE[0] / CBAR_FIGSIZE[0],
    CBAR_AXSIZE[1] / CBAR_FIGSIZE[1],
)
set_rcParams()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=float, default=29642.6)
    parser.add_argument("--space-extent", type=int, nargs=2, default=[1150, 1000])
    parser.add_argument("-L", type=int, default=50)
    parser.add_argument("-l", type=float, default=75.0)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    N = round(args.density * math.prod(args.space_extent) / 1e6)
    print(f"{N=}")
    halfL = args.L / 2
    x = neurons.sample(
        N=N,
        variables=["cell_type", "space", "ori", "osi"],
        cell_types=["PYR", "PV"],
        space_extent=args.space_extent,
        osi_prob=["Beta", 0.88, 1.07],
        min_dist=3.0,
        w_dims=[],
    )
    df = x.to_framelike()
    hsluv = np.concatenate(
        [df[["ori[0]", "osi"]].to_numpy(), np.full((len(df), 1), args.l)], axis=-1
    )
    hsluv[:, 0] = (hsluv[:, 0] + 90) * 2
    hsluv[:, 1] = hsluv[:, 1] * 100
    df.loc[:, ["r", "g", "b"]] = np.array([hsluv_to_rgb(v) for v in hsluv])
    markers = {"PYR": "^", "PV": "o"}

    # Custom orientation tuning preference colorbar
    hue_cmap = colors.ListedColormap(
        [hsluv_to_rgb([h, 100, args.l]) for h in np.linspace(0, 360, 256)]
    )
    ori_norm = colors.Normalize(vmin=-90, vmax=90)

    fig = plt.figure(figsize=CBAR_FIGSIZE)
    ax = fig.add_axes(CBAR_RECT)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=ori_norm, cmap=hue_cmap),
        cax=ax,
        orientation="horizontal",
    )
    cbar.ax.set_xticks([], [])
    if args.out:
        plt.savefig(
            args.out / "2f_ori_cmap.pdf",
            metadata={"Subject": " ".join(["python"] + sys.argv)},
        )
    if args.show:
        plt.show()

    # Custom OSI colorbar
    sat_cmap = colors.ListedColormap(
        [hsluv_to_rgb([0, s, args.l]) for s in np.linspace(0, 100, 256)]
    )
    osi_norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=CBAR_FIGSIZE)
    ax = fig.add_axes(CBAR_RECT)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=osi_norm, cmap=sat_cmap),
        cax=ax,
        orientation="horizontal",
    )
    cbar.ax.set_xticks([], [])
    if args.out:
        plt.savefig(
            args.out / "2f_osi_cmap.pdf",
            metadata={"Subject": " ".join(["python"] + sys.argv)},
        )
    if args.show:
        plt.show()

    # Main panel
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    for cell_type, sf in df.groupby("cell_type", observed=True):
        ax.scatter(
            sf["space[0]"],
            sf["space[1]"],
            s=0.05,
            edgecolors="none",
            c=sf[["r", "g", "b"]].to_numpy(),
            marker=markers[cell_type],
        )
    ax.add_artist(
        plt.Rectangle((-halfL, -halfL), args.L, args.L, fill=False, lw=GRID_WIDTH / 2)
    )
    ax.set_xlim(-args.space_extent[0] / 2, args.space_extent[0] / 2)
    ax.set_ylim(-args.space_extent[1] / 2, args.space_extent[1] / 2)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel(f"{args.space_extent[0]} µm", labelpad=2)
    ax.set_ylabel(f"{args.space_extent[1]} µm", labelpad=2)
    ax.xaxis.set_label_position("top")
    ax.set_aspect("equal")

    if args.out:
        plt.savefig(
            args.out / "2f.pdf",
            metadata={"Subject": " ".join(["python"] + sys.argv)},
        )
    if args.show:
        plt.show()

    # Inset panel
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    df = df.query(f"`space[0]`.abs() < {halfL} and `space[1]`.abs() < {halfL}")
    for cell_type, sf in df.groupby("cell_type", observed=True):
        ax.scatter(
            sf["space[0]"],
            sf["space[1]"],
            s=16,
            edgecolors="none",
            c=sf[["r", "g", "b"]].to_numpy(),
            marker=markers[cell_type],
        )
    ax.set_xlim(-halfL, halfL)
    ax.set_ylim(-halfL, halfL)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel(f"{args.L} µm", labelpad=2)
    ax.set_ylabel(f"{args.L} µm", labelpad=2)

    if args.out:
        plt.savefig(
            args.out / "2f_inset.pdf",
            metadata={"Subject": " ".join(["python"] + sys.argv)},
        )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
