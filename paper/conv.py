import argparse
from functools import partial
import math
import cmath

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch

from niarb.special.resolvent import laplace_r


def laplace_r_conv(d, l0, l1, r):
    r = torch.as_tensor(r)
    if l0 == l1:
        out = laplace_r(d - 2, l0, r) / (4 * torch.pi)
    else:
        out = (laplace_r(d, l0, r) - laplace_r(d, l1, r)) / (l1 - l0)

        if d == 1:
            singularity = (1 / cmath.sqrt(l1) - 1 / cmath.sqrt(l0)) / 2
        elif d == 2:
            singularity = (cmath.log(l0) - cmath.log(l1)) / (4 * cmath.pi)
        elif d == 3:
            singularity = (cmath.sqrt(l0) - cmath.sqrt(l1)) / (4 * cmath.pi)
        else:
            singularity = np.inf
        singularity = singularity / (l0 - l1)
        out = torch.where(r != 0, out, singularity)

    return out.numpy()


def angle1(l, rho):
    return cmath.phase(1j * (rho * l - 1))


def angle2(l, rho):
    return cmath.phase(-((1 - rho * l) ** 2) / cmath.sqrt(l))


def modulus(l):
    sqrtl = cmath.sqrt(l)
    return 2 * abs(sqrtl) * sqrtl.imag / l.imag


def L1(r, l, rho, prefactor=True, mul_r=False):
    sqrtl = cmath.sqrt(l)
    ir = sqrtl.imag * r
    out = np.cos(ir - angle1(l, rho))
    if prefactor:
        rr = sqrtl.real * r
        out = out * np.exp(-rr) / (2 * np.pi)
    if mul_r:
        return out
    return out / r


def L2(r, l, rho, prefactor=True):
    sqrtl = cmath.sqrt(l)
    ir = sqrtl.imag * r
    comp1 = np.cos(ir - angle2(l, rho))
    comp2 = modulus(l) * np.sinc(ir / np.pi)
    print(modulus(l), angle2(l, rho), sqrtl.imag)
    if prefactor:
        rr = sqrtl.real * r
        comp1 = comp1 * np.exp(-rr) / (4 * np.pi * abs(sqrtl))
        comp2 = comp2 * np.exp(-rr) / (4 * np.pi * abs(sqrtl))
    return comp1, comp2


def expected_L2(r, l0: complex, rho: float, d=3):
    l1 = l0.conjugate()
    coef0 = (1 - rho * l0) / (l1 - l0)
    coef0 = coef0 / abs(coef0)
    coef1 = np.conj(coef0)
    return (
        coef0**2 * laplace_r_conv(d, l0, l0, r)
        + coef1**2 * laplace_r_conv(d, l1, l1, r)
        + 2 * coef0 * coef1 * laplace_r_conv(d, l0, l1, r)
    ).real


def integrand(r, l, rho, x1, x2, x3):
    r1 = math.sqrt(x1**2 + x2**2 + x3**2)
    r2 = math.sqrt((x1 - r) ** 2 + x2**2 + x3**2)
    return L1(r1, l, rho) * L1(r2, l, rho)


@np.vectorize
def integral(r, l, rho):
    integrand_ = partial(integrand, r, l, rho)
    return integrate.tplquad(
        integrand_, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf
    )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("real_l", type=float)
    parser.add_argument("imag_l", type=float)
    parser.add_argument("--exp", "-e", action="store_true")
    parser.add_argument("--comp", "-c", action="store_true")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--rmin", type=float, default=0.0)
    parser.add_argument("--rmax", type=float, default=5.0)
    parser.add_argument("-N", type=float, default=1000)
    args = parser.parse_args()

    l = complex(args.real_l, args.imag_l)
    rho = args.rho
    assert l.imag < 0
    assert rho > 0
    assert args.rmin >= 0
    assert args.rmax > 0

    r = np.linspace(args.rmin, args.rmax, num=args.N)
    y1 = L1(r, l, rho, mul_r=True)
    y2 = L2(r, l, rho)
    expected_2d = expected_L2(r, l, rho, d=2)
    expected_3d = expected_L2(r, l, rho)
    if args.exp:
        exp = np.exp(cmath.sqrt(l).real * r)
        y1, y2 = y1 * exp, y2 * exp
        expected_2d, expected_3d = expected_2d * exp, expected_3d * exp

    plt.plot(r, y1, label="r * L1")
    if args.comp:
        plt.plot(r, y2[0], label="L2 (1st comp)")
        plt.plot(r, y2[1], label="L2 (2nd comp)")
    plt.plot(r, sum(y2), label="L2")
    plt.plot(r, expected_2d, ls="--", label="L2 (2D)")
    plt.plot(r, expected_3d, ls="--", label="L2 (expected)")
    plt.gca().axhline(
        0, color=rcParams["grid.color"], linewidth=rcParams["grid.linewidth"]
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
