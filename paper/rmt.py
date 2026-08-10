import argparse

import matplotlib.pyplot as plt
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("--sigma", "-s", type=float, default=0.5)
    args = parser.parse_args()

    k = args.n // args.m
    assert args.m * k == args.n

    W = args.sigma * torch.randn((args.n, args.n)) / args.n**0.5
    tL = torch.linalg.inv(torch.eye(args.n) - W) - torch.eye(args.n)
    tL_sub = tL.reshape(k, args.m, k, args.m).diagonal(dim1=0, dim2=2).movedim(-1, 0)
    assert tL_sub.shape == (k, args.m, args.m)
    assert (tL_sub[0] == tL[: args.m, : args.m]).all()
    assert (tL_sub[1] == tL[args.m : 2 * args.m, args.m : 2 * args.m]).all()

    # W_eigvals = torch.linalg.eigvals(W)
    # tL_eigvals = torch.linalg.eigvals(tL)
    tL_sub_eigvals = torch.linalg.eigvals(tL_sub).reshape(-1)
    tL_sub_norm = torch.linalg.matrix_norm(tL_sub, ord=1)
    print(torch.linalg.matrix_norm(tL, ord=1))
    print(tL_sub_eigvals.abs().max())
    print(tL_sub_norm.max())
    # fig, ax = plt.subplots()
    # ax.scatter(W_eigvals.real, W_eigvals.imag, s=1)
    # ax.scatter(tL_eigvals.real, tL_eigvals.imag, s=1)
    # ax.scatter(tL_sub_eigvals.real, tL_sub_eigvals.imag, s=1)
    # ax.set_aspect("equal")
    # ax.add_patch(plt.Circle((0, 0), args.sigma, color="gray", ls="--", fill=False))
    # θ = torch.linspace(-torch.pi, torch.pi, 1000)
    # z = args.sigma * torch.exp(1j * θ)
    # z = 1 / (1 - args.sigma * torch.exp(1j * θ)) - 1
    # ax.plot(z.real, z.imag, color="gray", ls="--")
    plt.show()


if __name__ == "__main__":
    main()
