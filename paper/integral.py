from functools import partial

import torch
from scipy import integrate, special
import numpy as np
import matplotlib.pyplot as plt

from niarb.special import resolvent


def z(r12, r13, r23, l1, l2, l3, u2, u3):
    u1 = 1 - u2 - u3
    ul = u1 * l1 + u2 * l2 + u3 * l3
    uu = u1 * u2 + u1 * u3 + u2 * u3
    ur = u1 * r23**2 + u2 * r13**2 + u3 * r12**2
    return (ur * ul / uu) ** 0.5


def integrand(d, r12, r13, r23, l1, l2, l3, u2, u3, cplx=False):
    u1 = 1 - u2 - u3
    # def integrand(d, r12, r13, r23, l1, l2, l3, u2, u1, cplx=False):
    # u3 = 1 - u1 - u2
    ul = u1 * l1 + u2 * l2 + u3 * l3
    uu = u1 * u2 + u1 * u3 + u2 * u3
    ur = u1 * r23**2 + u2 * r13**2 + u3 * r12**2
    nu = 3 - d
    kv = special.kv if cplx else special.kn
    return kv(nu, (ur * ul / uu) ** 0.5) * (ur / ul) ** (nu / 2) * uu**-1.5


@np.vectorize
def integral(d, r12, r13, r23, l1, l2, l3):
    const = 1 / (4 * (2 * np.pi) ** d)
    ub = lambda x: 1 - x
    cplx = any(isinstance(l, complex) for l in [l1, l2, l3])
    integrand_ = partial(integrand, d, r12, r13, r23, l1, l2, l3, cplx=cplx)
    if not cplx:
        out = integrate.dblquad(integrand_, 0, 1, 0, ub)
        # print(out[1])
        return const * out[0]
    return const * (
        integrate.dblquad(lambda u, v: integrand_(u, v).real, 0, 1, 0, ub)[0]
        + 1j * integrate.dblquad(lambda u, v: integrand_(u, v).imag, 0, 1, 0, ub)[0]
    )


def conv(d, r, l1, l2):
    r = torch.tensor(r)
    if l1 == l2:
        return resolvent.laplace_r(d - 2, l1, r).numpy() / (4 * np.pi)
    return (
        (resolvent.laplace_r(d, l1, r) - resolvent.laplace_r(d, l2, r)) / (l2 - l1)
    ).numpy()


def laplace_r_2d(l, r, cplx=False):
    k0 = partial(special.kv, 0) if cplx else special.k0
    return k0(l**0.5 * r) / (2 * np.pi)


def expected_conv_integrand_2d(r, l1, l2, y1, y2, cplx=False):
    r1 = (y1**2 + y2**2) ** 0.5
    r2 = ((y1 - r) ** 2 + y2**2) ** 0.5
    return laplace_r_2d(l1, r1, cplx) * laplace_r_2d(l2, r2, cplx)


def expected_integrand_2d(r, l1, l2, l3, y1, y2, cplx=False):
    r1 = (y1**2 + y2**2) ** 0.5
    r2 = r1
    r3 = ((y1 - r) ** 2 + y2**2) ** 0.5
    return (
        laplace_r_2d(l1, r1, cplx)
        * laplace_r_2d(l2, r2, cplx)
        * laplace_r_2d(l3, r3, cplx)
    )


@np.vectorize
def expected_2d(r, *ls, n=3):
    if n not in {2, 3}:
        raise ValueError(f"n must be 2 or 3, got {n}")

    cplx = any(isinstance(l, complex) for l in ls)
    integrand = expected_conv_integrand_2d if n == 2 else expected_integrand_2d
    integrand = partial(integrand, r, *ls, cplx=cplx)
    if not cplx:
        out = integrate.dblquad(integrand, -np.inf, np.inf, -np.inf, np.inf)
        # print(out[1])
        return out[0]
    return (
        integrate.dblquad(
            lambda u, v: integrand(u, v).real, -np.inf, np.inf, -np.inf, np.inf
        )[0]
        + 1j
        * integrate.dblquad(
            lambda u, v: integrand(u, v).imag, -np.inf, np.inf, -np.inf, np.inf
        )[0]
    )


def exponent(l1, l2, l3, u2, u3):
    return (
        (1 - u3)
        * (l1 + u2 * (l2 - l1) + u3 * (l3 - l1))
        / ((1 - u3) * (u2 + u3) - u2**2)
    )


def approx_integrand_2d(r, l1, l2, l3, u, cplx=False):
    c = ((l1 + l2 - (l1 + l2 - 2 * l3) * u) / 2) ** 0.5
    b = 2 / (1 + 3 * u) ** 0.5
    k1 = partial(special.kv, 1) if cplx else special.k1
    return b**3 / c * r * k1(r * b * c)


@np.vectorize
def approx_integral_2d(r, *ls):
    const = 1 / (4 * (2 * np.pi) ** 2)
    cplx = any(isinstance(l, complex) for l in ls)
    integrand_ = partial(approx_integrand_2d, r, *ls, cplx=cplx)
    # return const * integrate.quad(integrand_, 0, 1)[0]
    return const * integrand_(1 / 3)
    # return const * integrand_(1 / 2)


def approx_integral_2d_2(r, l):
    const = 1 / (4 * (2 * np.pi) ** 2)
    cplx = isinstance(l, complex)
    k1 = partial(special.kv, 1) if cplx else special.k1
    k2 = partial(special.kv, 2) if cplx else partial(special.kn, 2)
    prefactor = r / (72 * l**0.5)
    arg = (2 * l) ** 0.5 * r
    term1 = 18 * 2**0.5 * (14 + l * r**2) * k1(arg)
    term2 = -(l**0.5) * r * (84 + l * r**2) * k2(arg)
    return const * prefactor * (term1 + term2)
    # return const * 2 * 2**0.5 * r / l**0.5 * k1(arg)


def approx_integral_2d_3(r, l1, l2, l3):
    const = 1 / (4 * (2 * np.pi) ** 2)
    cplx = any(isinstance(l, complex) for l in (l1, l2, l3))
    k1 = partial(special.kv, 1) if cplx else special.k1
    k2 = partial(special.kv, 2) if cplx else partial(special.kn, 2)
    sl = l1 + l2 + l3
    prefactor = r / (864 * sl**0.5)
    prefactor1 = (
        6**0.5
        * (
            4
            * (
                (l1 + l2) ** 2 * (71 * l1**2 + 82 * l1 * l2 + 71 * l2**2)
                + 8 * (l1 + l2) * (7 * l1**2 + 26 * l1 * l2 + 7 * l2**2) * l3
                + 3 * (39 * l1**2 - 86 * l1 * l2 + 39 * l2**2) * l3**2
                - 28 * (l1 + l2) * l3**3
                + 2 * l3**4
            )
            * r**2
            + (l1 - l2) ** 2 * (-2 * (l1 + l2) + l3) ** 2 * r**4 * sl
            + 3024 * sl**3
        )
    ) / sl**3
    prefactor2 = (
        4
        * r
        * (
            12
            * (
                -2 * (l1 + l2) ** 2 * (l1**2 - 28 * l1 * l2 + l2**2)
                + 2 * (l1 + l2) * (65 * l1**2 + 82 * l1 * l2 + 65 * l2**2) * l3
                + 3 * (9 * l1**2 + 182 * l1 * l2 + 9 * l2**2) * l3**2
                + 34 * (l1 + l2) * l3**3
                - 23 * l3**4
            )
            + (2 * (l1 + l2) - l3)
            * (
                2 * (l1 + l2) ** 3
                + 6 * (5 * l1**2 - 8 * l1 * l2 + 5 * l2**2) * l3
                - l3**3
            )
            * r**2
            * sl
        )
    ) / sl**3.5
    arg = (2 * sl / 3) ** 0.5 * r
    return const * prefactor * (prefactor1 * k1(arg) - prefactor2 * k2(arg))


def main():
    d = 2
    r = np.linspace(0.1, 5, 50)
    # r = 1
    # l1, l2, l3 = 1, 1 + 2j, 1 - 2j
    # l1, l2, l3 = 2, 1, 5
    # l1, l2, l3 = 1, 3, 3
    # l1, l2, l3 = 1, 3, 1
    l1, l2, l3 = 1, 1, 1
    # l1, l2, l3 = 1, 1, 3
    # l1, l2, l3 = 3, 1, 1
    is_complex = any(isinstance(l, complex) for l in [l1, l2, l3])

    # u = np.linspace(0, 1, 100)
    # for ri in [0.1, 0.5, 1.0]:
    #     plt.plot(u, approx_integrand_2d(ri, l1, l2, l3, u, cplx=is_complex), label=ri)
    # plt.legend(title="r")
    # plt.show()

    # # out = conv(d, r, l1, l2)
    # # expected = expected_2d(r, l1, l2, n=2)
    out = integral(d, 0, r, r, l1, l2, l3)
    # expected = expected_2d(r, l1, l2, l3)
    # approx = approx_integral_2d(r, l1, l2, l3)
    # approx = approx_integral_2d_2(r, l3)
    approx = approx_integral_2d_3(r, l1, l2, l3)

    if is_complex:
        plt.plot(r, out.real, label="integral (real)")
        plt.plot(r, out.imag, label="integral (imag)")
        # plt.plot(r, expected.real, label="expected (real)", linestyle="--")
        # plt.plot(r, expected.imag, label="expected (imag)", linestyle="--")
        plt.plot(r, approx.real, label="approx (real)", linestyle="--")
        plt.plot(r, approx.imag, label="approx (imag)", linestyle="--")
    else:
        plt.plot(r, out, label="integral")
        # plt.plot(r, expected, label="expected", linestyle="--")
        plt.plot(r, approx, label="approx (real)", linestyle="--")
    # plt.plot(r, special.k0(l3**0.5 * r) / (8 * np.pi**2 * l3), label="approx")

    plt.legend()
    plt.show()

    # n = 200
    # r = 1
    # u2, u3 = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    # out = np.zeros_like(u2, dtype=np.float64 if not is_complex else np.complex128)
    # mask = u2 + u3 <= 1
    # out[~mask] = np.nan
    # out[mask] = integrand(d, 0, r, r, l1, l2, l3, u2[mask], u3[mask], cplx=is_complex)
    # # out[mask] = integrand(d, 0, r, r, l1, l2, l3, u3[mask], u2[mask], cplx=is_complex)
    # # out[mask] = np.exp(-(r**2 / 4) * exponent(l1, l2, l3, u2[mask], u3[mask])) / (
    # #     (1 - u3[mask]) * (u2[mask] + u3[mask]) - u2[mask] ** 2
    # # )
    # # out[mask] = z(0, r, r, l1, l2, l3, u2[mask], u3[mask])

    # for i in range(0, n, n // 10):
    #     plt.plot(u2[i], out[i], label=f"{u3[i, 0]:.2f}")
    # plt.legend(title="u3")
    # plt.show()

    # # im = plt.contourf(u2, u3, out.real, cmap="Reds", levels=100, norm="log")
    # im = plt.contourf(u2, u3, out.real, cmap="Reds", levels=50)
    # plt.colorbar(im)
    # plt.show()

    # if is_complex:
    #     print(out[~np.isnan(out)].sum())
    #     im = plt.contourf(u2, u3, out.imag, cmap="bwr", levels=100, norm="symlog")
    #     plt.colorbar(im)
    #     plt.show()


if __name__ == "__main__":
    main()
