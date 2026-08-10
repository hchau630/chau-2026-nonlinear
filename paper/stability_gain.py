import argparse

import numpy as np


def compute_is_stable(W, tau):
    n = W.shape[-1]
    J = np.diag(1 / tau) @ (W - np.eye(n))  # (..., n, n)
    eigvals = np.linalg.eigvals(J)  # (..., n)
    is_stable = (eigvals.real < 0).all(axis=-1)  # (...)
    return is_stable, eigvals, J


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=10000)
    parser.add_argument("-K", type=int, default=100)
    parser.add_argument("-n", type=int, default=3)
    parser.add_argument("-s", "--scale", type=float, default=2.0)
    parser.add_argument("--tau", type=float, nargs="+", default=[1.0, 0.5, 1.0])
    parser.add_argument("--gmax", type=float, default=100.0)
    args = parser.parse_args()

    W = args.scale * np.abs(np.random.randn(args.N, args.n, args.n))
    W[:, :, 1:] = -W[:, :, 1:]  # make inhibitory
    print(W[0])
    tau = np.array(args.tau)[: args.n]
    is_stable, _, _ = compute_is_stable(W, tau)  # (N,)
    W = W[is_stable]  # (M, n, n)
    print(np.count_nonzero(is_stable))
    _, _, JW = compute_is_stable(W, tau)  # (M, n, n)
    G = np.broadcast_to(np.eye(args.n), (args.K, args.n, args.n)).copy()
    G[..., -1, -1] = np.logspace(0.0, np.log10(args.gmax), args.K)
    GW = G[:, None, ...] @ W[None, ...]  # (K, M, n, n)
    is_stable, _, _ = compute_is_stable(GW, tau)  # (K, M)
    is_stable = is_stable.all(axis=0)  # (M,)
    print(np.count_nonzero(is_stable))

    # EPV_unstable_det = np.linalg.det(JW[..., :2, :2]) < 0  # (M,)
    # EPV_unstable_tr = np.linalg.trace(JW[..., :2, :2]) > 0  # (M,)
    # sufficient = EPV_unstable_det & EPV_unstable_tr
    # sufficient = EPV_unstable_det

    L = np.linalg.inv(np.eye(args.n) - W)  # (M, n, n)
    val1 = L[:, -1, -1] - 1  # (M,)
    val2 = (
        tau[0] * L[:, 0, 0]
        + tau[1] * L[:, 1, 1]
        - np.linalg.trace(JW[:, :-1, :-1]) / (tau[-1] * np.linalg.det(JW))
    )  # (M,)
    t0, t1, t2 = tau
    W00, W01, W02 = W[:, 0, 0], W[:, 0, 1], W[:, 0, 2]
    W10, W11, W12 = W[:, 1, 0], W[:, 1, 1], W[:, 1, 2]
    W20, W21, W22 = W[:, 2, 0], W[:, 2, 1], W[:, 2, 2]
    val3 = 4 * t0 * t1 * (t1 * (-1 + W00) + t0 * (-1 + W11)) * (
        t0 * (t1 + t2 - t2 * W11)
        + t2 * (t1 + t2 - t1 * W00 - t2 * (W00 + W01 * W10) + t2 * (-1 + W00) * W11)
    ) - (
        -(t1**2 * t2 * (-1 + W00) * (W02 * W20 + W22 - W00 * W22))
        + t0**2
        * (
            (t1 + t2 - t2 * W11) * W12 * W21
            + (-2 * t1 + t2 * (-1 + W11)) * (-1 + W11) * W22
        )
        + t0
        * t1
        * (
            t1 * W02 * W20
            - t2 * W01 * W12 * W20
            - t2 * W02 * W10 * W21
            - 2 * (-1 + W00) * (t1 + t2 - t2 * W11) * W22
        )
    ) ** 2 / (
        W22
        * (
            -(t1 * W02 * W20)
            - t0 * W12 * W21
            + t1 * (-1 + W00) * W22
            + t0 * (-1 + W11) * W22
        )
    )
    val4 = (
        -0.5
        * (
            t1**2 * t2 * (-1 + W00) * (W02 * W20 + W22 - W00 * W22)
            + t0
            * t1
            * (
                -(t1 * W02 * W20)
                + t2 * W01 * W12 * W20
                + t2 * W02 * W10 * W21
                + 2 * (-1 + W00) * (t1 + t2 - t2 * W11) * W22
            )
            + t0**2
            * (
                -((t1 + t2 - t2 * W11) * W12 * W21)
                + (-1 + W11) * (2 * t1 + t2 - t2 * W11) * W22
            )
        )
        / (
            t0
            * t1
            * W22
            * (t1 * (W02 * W20 + W22 - W00 * W22) + t0 * (W12 * W21 + W22 - W11 * W22))
        )
    )
    # Wss = W[:, -1, -1]
    # detWee = np.linalg.det(W[:, 1:, 1:])
    # detWpp = np.linalg.det(W[:, ::2, ::2])
    # val1 = np.linalg.det(W) - detWee - detWpp + Wss
    # val2 = (Wss - detWee) / tau[1] + (Wss - detWpp) / tau[0]
    cond1 = val1 < 0
    cond2 = val2 > 0
    # cond2 = val2 < 0
    cond3 = val3 < 0
    cond4 = val4 < 1
    # iff = cond1 & cond2
    iff = cond1 & cond2 & (cond3 | cond4)
    fails = np.logical_xor(iff, is_stable)
    false_pos = fails & iff
    false_neg = fails & ~iff
    print(
        np.count_nonzero(iff),
        np.count_nonzero(fails),
        np.count_nonzero(false_pos),
        np.count_nonzero(false_neg),
    )
    if np.count_nonzero(false_pos) > 0:
        print("False positive")
        print(W[false_pos][0], val1[false_pos][0], val2[false_pos][0])
    if np.count_nonzero(false_neg) > 0:
        print("False negative")
        print(W[false_neg][0], val1[false_neg][0], val2[false_neg][0])
    # print(np.count_nonzero(sufficient), np.count_nonzero(sufficient & is_stable))
    # idx = np.nonzero(~is_stable)[0][0]
    # print(W[idx])
    # print(eigvals[:, idx])
    # print(compute_is_stable(W[idx], tau))


if __name__ == "__main__":
    main()
