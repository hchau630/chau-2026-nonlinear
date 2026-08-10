import argparse
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from zero_crossing import bisect, first_zero


def angle1(l, rho):
    return np.angle(1j * (rho * l - 1))


def angle2(l, rho):
    return np.angle(-((1 - rho * l) ** 2) / np.sqrt(l))


def modulus(l):
    sqrtl = np.sqrt(l)
    return 2 * np.abs(sqrtl) * sqrtl.imag / l.imag


def L1(r, l, rho, prefactor=True, mul_r=False):
    sqrtl = np.sqrt(l)
    ir = sqrtl.imag * r
    out = np.cos(ir - angle1(l, rho))
    if prefactor:
        rr = sqrtl.real * r
        out = out * np.exp(-rr) / (2 * np.pi)
    if mul_r:
        return out
    return out / r


def L2(r, l, rho, prefactor=True):
    sqrtl = np.sqrt(l)
    ir = sqrtl.imag * r
    comp1 = np.cos(ir - angle2(l, rho))
    comp2 = modulus(l) * np.sinc(ir / np.pi)
    out = comp1 + comp2
    if prefactor:
        rr = sqrtl.real * r
        out = out * np.exp(-rr) / (4 * np.pi * np.abs(sqrtl))
    return out


def L1_crossing(l, rho, tol=1e-8):
    rmin = np.zeros(())
    rmax = -np.pi / np.sqrt(l).imag

    return bisect(
        partial(L1, prefactor=False, mul_r=True), rmin, rmax, args=(l, rho), tol=tol
    )


# def L2_crossing(l, rho, n=50, tol=1e-8):
#     rmin = np.zeros(())
#     rmax = -1.5 * np.pi / np.sqrt(l).imag
#     segments = np.linspace(rmin, rmax, num=n + 1)
#     rmin = segments[:-1]
#     rmax = segments[1:]

#     return np.nanmin(
#         bisect(partial(L2, prefactor=False), rmin, rmax, args=(l, rho), tol=tol), axis=0
#     )


def L2_crossing(l, rho, tol=1e-8):
    a = modulus(l)
    θ = -angle2(l, rho)
    k = -np.sqrt(l).imag
    return first_zero(a, θ, tol=tol) / k


def L1_crossing_expected(l, rho):
    return (angle1(l, rho) - np.pi / 2) / np.sqrt(l).imag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-real-l", type=float, default=-2.0)
    parser.add_argument("--max-real-l", type=float, default=2.0)
    parser.add_argument("--min-imag-l", type=float, default=-2.0)
    parser.add_argument("--max-imag-l", type=float, default=0.0)
    parser.add_argument("--N-real", type=int, default=100)
    parser.add_argument("--N-imag", type=int, default=100)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("-n", type=int, default=50)
    args = parser.parse_args()

    rho = args.rho
    levels = args.levels
    real_l = np.linspace(args.min_real_l, args.max_real_l, num=args.N_real)
    imag_l = np.linspace(args.min_imag_l, args.max_imag_l, num=args.N_imag + 1)[:-1]
    real_ll, imag_ll = np.meshgrid(real_l, imag_l)
    ll = real_ll + 1j * imag_ll

    angle1_ = np.rad2deg(angle1(ll, rho))
    angle2_ = np.rad2deg(angle2(ll, rho))

    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # im = axes[0].contourf(real_ll, imag_ll, angle1_, levels=levels, cmap="hsv")
    # fig.colorbar(im, ax=axes[0])
    # axes[0].set_ylim(args.min_imag_l, 0)
    # im = axes[1].contourf(real_ll, imag_ll, angle2_, levels=levels, cmap="hsv")
    # fig.colorbar(im, ax=axes[1])
    # axes[1].set_ylim(args.min_imag_l, 0)
    # plt.show()

    # plt.contourf(real_ll, imag_ll, angle1_ - angle2_, levels=levels, cmap="hsv")
    # plt.colorbar()
    # plt.ylim(args.min_imag_l, 0)
    # plt.show()

    # plt.contourf(real_ll, imag_ll, modulus(ll), levels=levels, norm="log", cmap="Reds")
    # cbar = plt.colorbar(ticks=[1, 2, 5, 10, 20, 50, 100])
    # cbar.ax.set_yticklabels([1, 2, 5, 10, 20, 50, 100])
    # plt.show()

    L1_cross = L1_crossing(ll, rho)
    # L1_expected = L1_crossing_expected(ll, rho)
    L2_cross = L2_crossing(ll, rho)

    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # im = axes[0].contourf(
    #     real_ll, imag_ll, L1_cross, levels=levels, norm="log", cmap="Reds"
    # )
    # fig.colorbar(im, ax=axes[0])
    # # im = axes[1].contourf(
    # #     real_ll, imag_ll, L1_expected, levels=levels, norm="log", cmap="Reds"
    # # )
    # im = axes[1].contourf(
    #     real_ll, imag_ll, L2_cross, levels=levels, norm="log", cmap="Reds"
    # )
    # fig.colorbar(im, ax=axes[1])
    # plt.show()

    ratio = L2_cross / L1_cross
    print(np.min(ratio), np.max(ratio))

    # plt.contourf(real_ll, imag_ll, ratio, levels=levels, cmap="Reds")
    # plt.colorbar()
    # plt.show()

    x, y = np.linspace(-6, 5, num=100), np.linspace(0, 10, num=100)
    xx, yy = np.meshgrid(x, y)
    tr = 2 - xx
    det = yy - 1 + tr
    l = 0.5 * (tr - np.sqrt(tr**2 - 4 * det + 0.0j))
    zz = np.full_like(xx, np.nan)
    mask = yy > xx**2 / 4
    zz[mask] = L2_crossing(l[mask], 1.0) / L1_crossing(l[mask], 1.0)
    plt.contourf(xx, yy, zz, levels=levels, cmap="Reds")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
