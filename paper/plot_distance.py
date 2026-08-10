import argparse
from functools import partial
from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

from niarb import neurons, perturbation
from mpl_config import set_rcParams, FIGSIZE, RECT

set_rcParams()


def cond(min_inter_target_dist, x):
    out = perturbation.inter_target_distance_statistics(x, ["min"])["min"].item()
    return out >= min_inter_target_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=float, default=29642.6)
    parser.add_argument("-L", type=int, default=50)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--n-targets", type=int, default=3)
    parser.add_argument("--min-dist", type=float, default=3.0)
    parser.add_argument(
        "--min-inter-target-dist", "--target-dist", type=float, default=25.0
    )
    parser.add_argument("--levels", type=int, default=[10, 15, 20, 25, 30, 35])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("-s", type=float)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    args.L = args.L * args.scale
    args.min_dist = args.min_dist * args.scale
    args.min_inter_target_dist = args.min_inter_target_dist * args.scale
    args.levels = [level * args.scale for level in args.levels]
    args.density = args.density / (args.scale**2)

    N = round(args.density * args.L**2 / 1e6)
    print(f"{N=}")
    halfL = args.L / 2
    xx = torch.linspace(-halfL, halfL, steps=args.steps)
    yy = torch.linspace(-halfL, halfL, steps=args.steps)
    xx, yy = torch.meshgrid(xx, yy, indexing="ij")
    loc = torch.stack([xx, yy], dim=-1)
    x = neurons.sample(
        N=N,
        variables=["cell_type", "space"],
        cell_types=["PYR", "PV"],
        space_extent=[args.L, args.L],
        min_dist=args.min_dist,
        w_dims=[],
    )
    x["dh"] = perturbation.sample(
        x,
        N=args.n_targets,
        cell_probs={"PYR": 1.0},
        cond=partial(cond, args.min_inter_target_dist),
    )
    distances = perturbation.min_distance(x[x["dh"] > 0]["space"], loc)
    df = x.to_framelike()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    cs = ax.contour(xx, yy, distances, levels=args.levels)
    ax.clabel(cs, fmt="%d μm", manual=True)

    markers = {"PYR": (3, 0, 0), "PV": "o"}
    for cell_type, sf in df.groupby("cell_type", observed=True):
        ax.scatter(
            sf["space[0]"],
            sf["space[1]"],
            facecolors="grey",
            edgecolors="none",
            marker=markers[cell_type],
            s=args.s,
        )
    sf = df.query("dh > 0")
    ax.scatter(
        sf["space[0]"],
        sf["space[1]"],
        facecolors="none",
        edgecolors="#BE1E2D",
        linestyle="--",
        marker="o",
        s=args.s,
    )
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    if args.out:
        plt.savefig(
            args.out / "2b.pdf",
            metadata={"Subject": " ".join(["python"] + sys.argv)},
        )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
