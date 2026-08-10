import argparse
from pathlib import Path
import sys

# import matplotlib.cm as cm
# import matplotlib.colors as colors
# import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_config import set_rcParams, CM, AXSIZE, GRID_COLOR, GRID_WIDTH, FIGSIZE, RECT
import torch
import numpy as np

from niarb import nn, numerics
from niarb.optimize import elementwise

CBAR_FIGSIZE = (1.0 * CM, 2.7 * CM)
CBAR_AXSIZE = (0.25 * CM, AXSIZE[1])
CBAR_RECT = (
    0.05 * CM / CBAR_FIGSIZE[0],
    0.4 * CM / CBAR_FIGSIZE[1],
    CBAR_AXSIZE[0] / CBAR_FIGSIZE[0],
    CBAR_AXSIZE[1] / CBAR_FIGSIZE[1],
)
SMALL_AXSIZE = (0.7 * CM, 0.5 * CM)
SMALL_FIGSIZE = (1.1 * CM, 0.8 * CM)
SMALL_RECT = (
    0.3 * CM / SMALL_FIGSIZE[0],
    0.25 * CM / SMALL_FIGSIZE[1],
    SMALL_AXSIZE[0] / SMALL_FIGSIZE[0],
    SMALL_AXSIZE[1] / SMALL_FIGSIZE[1],
)
set_rcParams()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf", type=float, default=1.0)
    parser.add_argument("--dv-min", type=float, default=-0.75)
    parser.add_argument("--dv-max", type=float, default=1.0)
    parser.add_argument("--color", "-c", type=str, default="black")
    parser.add_argument("--colormap", "--cmap", type=str, default="coolwarm")
    parser.add_argument("--vmax", type=float, default=3.0)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # plot ricciardi
    f = nn.Ricciardi(scale=1.0)
    a, b = torch.tensor(-1.0), torch.tensor(10.0)
    vf = elementwise.bisect(lambda x: f(x) - args.rf, a, b)
    dfdv = numerics.compute_nth_deriv(f, vf)
    d2fdv2 = numerics.compute_nth_deriv(f, vf, n=2)
    cv = 0.5 * d2fdv2 / dfdv**2
    print(
        f"rf={args.rf}, vf={vf.item():.4f}, dfdv={dfdv.item():.4f}, "
        f"d2fdv2={d2fdv2.item():.4f}, cv={cv.item():.4f}"
    )

    x = torch.linspace(vf + args.dv_min, vf + args.dv_max, 100)
    y = f(x)
    y_approx1 = dfdv * (x - vf) + args.rf
    y_approx2 = d2fdv2 / 2 * (x - vf) ** 2 + dfdv * (x - vf) + args.rf
    grid_kwargs = {"linewidth": GRID_WIDTH, "color": GRID_COLOR}
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    ax.plot(x, y, c=args.color, label="Ricciardi")
    ax.plot(x, y_approx1, ls="--", c=args.color, label="Linear")
    ax.plot(x, y_approx2, ls=":", c=args.color, label="Quadratic")

    ax.axhline(0.0, **grid_kwargs)
    ax.axvline(vf, ls="--", **grid_kwargs)
    ax.set_xticks(
        [vf.item(), vf.item() + args.dv_max], [r"$v_j^*$", r"$v_j^* + \delta v_j$"]
    )
    ax.set_yticks([0])
    # ax.set_ylabel("f(v)")
    ax.set_title("Transfer function")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # fig.legend()
    if args.out:
        plt.savefig(
            args.out / "3d.pdf", metadata={"Subject": " ".join(["python"] + sys.argv)}
        )
    if args.show:
        plt.show()

    # plot x^2
    x = np.linspace(-1, 1)
    fig = plt.figure(figsize=SMALL_FIGSIZE)
    ax = fig.add_axes(SMALL_RECT)
    ax.plot(x, x**2, c=args.color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$\delta v_j$", fontsize=5, labelpad=1)
    ax.set_ylabel(r"$H_{jj} \delta v_j^2$", fontsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # fig.legend()
    if args.out:
        plt.savefig(
            args.out / "3f_inset.pdf",
            metadata={"Subject": " ".join(["python"] + sys.argv)},
        )
    if args.show:
        plt.show()

    # # plot colormap
    # fig = plt.figure(figsize=CBAR_FIGSIZE)
    # ax = fig.add_axes(CBAR_RECT)
    # cmap = mpl.colormaps[args.colormap]
    # norm = colors.Normalize(-args.vmax, args.vmax)
    # cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    # # cbar.ax.set_xticks([], [])
    # cbar.ax.set_yticks([-args.vmax, 0, args.vmax])
    # if args.out:
    #     plt.savefig(
    #         args.out / "3d_cmap.pdf",
    #         metadata={"Subject": " ".join(["python"] + sys.argv)},
    #     )
    # if args.show:
    #     plt.show()


if __name__ == "__main__":
    main()
