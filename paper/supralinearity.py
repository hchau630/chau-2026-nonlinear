import argparse

import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

from niarb import nn, numerics
from niarb.optimize import elementwise

grid_kwargs = {
    "linewidth": rcParams["grid.linewidth"],
    "color": rcParams["grid.color"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf", type=float, default=0.58)
    parser.add_argument("--scale", type=float, default=(0.58 / 0.6))
    parser.add_argument("--vf0", type=float, default=1.0)
    parser.add_argument("--dhp", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--tau-rp", type=float, default=0.002)
    parser.add_argument("-f", type=str, default="ricciardi")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.f == "ricciardi":
        f = nn.Ricciardi(scale=args.scale, tau=args.tau, tau_rp=args.tau_rp)
    else:
        f = nn.SSN(p=int(args.f))

    # a, b = torch.tensor(-1.0), torch.tensor(10.0)
    # vf = elementwise.bisect(lambda x: f(x) - args.rf, a, b)
    if hasattr(f, "inv"):
        vf = f.inv()(torch.tensor(args.rf))
    else:
        vf = elementwise.newton(lambda x: f(x) - args.rf, torch.tensor(args.vf0))
    dfdv = numerics.compute_nth_deriv(f, vf)
    d2fdv2 = numerics.compute_nth_deriv(f, vf, n=2)
    # f(x) = k*(x - x0)^2 + y0, f'(x) = 2*k*(x - x0), f''(x) = 2*k
    # solving yields k = f''(x)/2, x0 = x - f'(x)/(2*k), y0 = f(x) - f'(x)^2/(4*k)
    k = d2fdv2 / 2  # k = f''(x)/2
    x0 = vf - dfdv / (2 * k)  # x0 = x - f'(x)/(2*k)
    y0 = args.rf - dfdv**2 / (4 * k)  # y0 = f'(x)^2/(4*k)
    cv = 0.5 * d2fdv2 / dfdv**2
    print(
        f"rf={args.rf}, vf={vf.item():.4f}, dfdv={dfdv.item():.4f}, "
        f"d2fdv2={d2fdv2.item():.4f}, k={k.item():.4f}, x0={x0.item():.4f}, "
        f"y0={y0.item():.4f}, cv={cv.item():.4f}"
    )
    rf2 = args.rf + args.dhp * args.scale
    # rf2 = args.rf + 17.58
    # rf2 = args.rf + 3 * 17.58
    # vf2 = elementwise.bisect(lambda x: f(x) - rf2, a, b)
    if hasattr(f, "inv"):
        vf2 = f.inv()(torch.tensor(rf2))
    else:
        vf2 = elementwise.newton(lambda x: f(x) - rf2, torch.tensor(args.vf0))
    print(f"rf2={rf2}, vf2={vf2.item():.4f}, dh={(vf2 - vf).item():.4f}")
    # print(((vf2 - vf) / 1.4132).item())
    x = torch.linspace(-1.0, 2.0 + vf, 100)
    plt.plot(x, f(x), label="Ricciardi")
    plt.plot(x, k * (x - x0) ** 2 + y0, label="Quadratic")
    plt.gca().axhline(0.0, **grid_kwargs)
    plt.gca().axhline(args.rf, ls="--", **grid_kwargs)
    plt.gca().axvline(vf, ls="--", **grid_kwargs)
    plt.gca().axhline(rf2, ls="--", **grid_kwargs)
    plt.gca().axvline(vf2, ls="--", **grid_kwargs)
    plt.legend()
    if args.show:
        plt.show()
    else:
        plt.clf()

    # for n in range(2, 4):
    #     f = nn.Ricciardi(scale=0.1, n=n)
    #     vf = torch.linspace(-1.0, 5.0, 100)
    #     plt.plot(vf, f(vf), label=f"Ricciardi(n={n})")
    # plt.legend()
    # plt.show()
    # return
    # rf = 0.58
    # f = nn.Ricciardi(scale=0.1)
    # a, b = torch.tensor(-1.0), torch.tensor(10.0)
    # vf = elementwise.bisect(lambda x: f(x) - rf, a, b)
    # print(vf)

    # vf = torch.linspace(-1, 10, 100)
    # for n, tau in product([1, 2, 5], [0.02, 0.01]):
    #     f = nn.Ricciardi(n=n, tau=tau)
    #     plt.plot(vf, f(vf), label=f"Ricciardi(n={n}, τ={tau})")
    # plt.legend()
    # plt.show()

    # rf = 0.58
    # rate = torch.linspace(0.1, 20, 100)
    # scale = rf / rate
    # for tau in [0.02, 0.01]:
    #     cv = []
    #     for s in scale:
    #         f = nn.Ricciardi(scale=s.item(), tau=tau)
    #         a, b = torch.tensor(-1.0), torch.tensor(10.0)
    #         vf = elementwise.bisect(lambda x: f(x) - rf, a, b)
    #         cv.append(
    #             0.5
    #             * numerics.compute_nth_deriv(f, vf, n=2)
    #             / numerics.compute_nth_deriv(f, vf) ** 2
    #         )
    #     cv = torch.tensor(cv)
    #     plt.plot(rate, cv, label=f"Ricciardi(τ={tau})")
    # for i, p in enumerate([2, 3]):
    #     cv = 0.5 * (p - 1) / (p * rf)
    #     plt.gca().axhline(cv, label=f"SSN(p={p})", color=f"C{i+2}")
    # plt.legend()
    # plt.show()

    # rate = torch.linspace(2, 10, 100)
    # # scale = 0.1
    # scale = 1.0
    # rf = rate * scale
    # for tau in [0.02, 0.01]:
    #     cv = []
    #     for rfi in rf:
    #         f = nn.Ricciardi(scale=scale, tau=tau, n=3)
    #         a, b = torch.tensor(-1.0), torch.tensor(10.0)
    #         vf = elementwise.bisect(lambda x: f(x) - rfi, a, b)
    #         cv.append(
    #             0.5
    #             * numerics.compute_nth_deriv(f, vf, n=2)
    #             / numerics.compute_nth_deriv(f, vf) ** 2
    #         )
    #     cv = torch.tensor(cv)
    #     plt.plot(rate, scale * cv, label=f"Ricciardi(τ={tau})")
    # for i, p in enumerate([2, 3, 4, 5]):
    #     cv = 0.5 * (p - 1) / (p * rf)
    #     plt.plot(rate, scale * cv, label=f"SSN(p={p})")
    # # plt.gca().axvline(0.58 / scale, color=rcParams["grid.color"])
    # plt.gca().axvline(5.8, color=rcParams["grid.color"])
    # plt.gca().axvline(3.1, color=rcParams["grid.color"])
    # plt.gca().axvline(5.8 * 0.4, color=rcParams["grid.color"])
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
