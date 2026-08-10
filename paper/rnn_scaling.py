import argparse
import math
import logging
from pathlib import Path
import sys
from functools import partial
from collections.abc import Callable

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize

from niarb import viz, nn, numerics
from niarb.optimize import elementwise
from mpl_config import GRID_WIDTH, GRID_COLOR


@np.vectorize
def integral(
    phi: Callable[[np.ndarray], np.ndarray], u: np.ndarray, delta: np.ndarray, p=1
) -> np.ndarray:
    """Compute E[phi(u + sqrt(delta) * z)^p]_z where z ~ N(0, 1) via numerical integration."""

    def integrand(z):
        return (
            phi(u + np.sqrt(delta) * z) ** p
            * np.exp(-(z**2) / 2)
            / math.sqrt(2 * math.pi)
        )

    return integrate.quad(integrand, -np.inf, np.inf)[0]


@np.vectorize
def integral2(
    f: Callable[[np.ndarray], np.ndarray],
    g: Callable[[np.ndarray], np.ndarray],
    u: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Compute E[f(u + sqrt(delta) * z)g(u + sqrt(delta) * z)]_z where z ~ N(0, 1) via numerical integration."""

    def integrand(z):
        return (
            f(u + np.sqrt(delta) * z)
            * g(u + np.sqrt(delta) * z)
            * np.exp(-(z**2) / 2)
            / math.sqrt(2 * math.pi)
        )

    return integrate.quad(integrand, -np.inf, np.inf)[0]


def solve_mft(
    phi: Callable[[np.ndarray], np.ndarray],
    u: np.ndarray,
    gbar: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for variance Δ  and input h0 such that mean is u.

    Given nonlinearity φ, and mean and standard deviation parameters of connectivity,
    the MFT equations are
    u = gbar E[φ(u + sqrt(Δ) z)]_z + h0
    Δ = g^2 E[φ(u + sqrt(Δ) z)^2]_z
    """

    def func(delta):
        return delta - g**2 @ integral(phi, u, delta, p=2)

    x0 = np.zeros_like(u) + 0.1
    sol = optimize.root(func, x0)
    print(f"initial residual: {func(np.zeros_like(u))}")
    print(f"final residual: {func(sol.x)}")
    # for x0 in np.linspace(0.05, 0.15, num=15):
    #     delta = np.full_like(u, x0)
    #     print(x0, (g**2 @ integral(phi, u, delta, p=2))[0], func(delta)[0])
    if not sol.success:
        raise RuntimeError(sol.message)

    delta = sol.x
    h0 = u - gbar @ integral(phi, u, delta)
    return delta, h0


def solve_mft2(
    phi: Callable[[np.ndarray], np.ndarray],
    h0: np.ndarray,
    gbar: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for mean u and variance Δ given input h0.

    Given nonlinearity φ, and mean and standard deviation parameters of connectivity,
    the MFT equations are
    u = gbar E[φ(u + sqrt(Δ) z)]_z + h0
    Δ = g^2 E[φ(u + sqrt(Δ) z)^2]_z
    """
    assert h0.ndim == 1
    n = h0.shape[0]

    def func(params):
        u, delta = params[:n], params[n:]
        eq1 = u - gbar @ integral(phi, u, delta) - h0
        eq2 = delta - g**2 @ integral(phi, u, delta, p=2)
        return np.concatenate([eq1, eq2])

    x0 = np.zeros((2 * n,))
    # x0[:n] = 0.05
    # x0[n:] = 0.3
    # x0[1] = -23.0
    # x0[-1] = 7276.0
    # x0[1] = -1.0
    # x0[-1] = 11.0
    x0[1] = -0.5
    x0[-1] = 1.0
    sol = optimize.root(func, x0)
    print(f"initial residual: {func(x0)}")
    print(f"final residual: {func(sol.x)}")
    if not sol.success:
        raise RuntimeError(sol.message)

    u, delta = sol.x[:n], sol.x[n:]
    return u, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--n-time", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=10000)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=50.0)
    parser.add_argument(
        "--exc-frac", "--frac", dest="exc_frac", type=float, default=0.5
    )
    parser.add_argument("--mu-exc", type=float, default=2.0)
    parser.add_argument("--mu-inh", type=float, default=-2.0)
    parser.add_argument("--sigma", "-s", type=float, default=0.5)
    parser.add_argument("--dh-rate", "--dh", dest="dh_rate", type=float, default=10.0)
    parser.add_argument("--n-perturbed", type=int, default=10)
    parser.add_argument("--rf", type=float, default=1.0)
    parser.add_argument("--vf-bounds", type=float, nargs=2, default=[-2.0, 10.0])
    parser.add_argument(
        "-f", type=str, choices=["ssn", "ricciardi", "rectified"], default="ricciardi"
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=[
            "analytical",
            "linear_approx",
            "quasi_linear_approx",
            "second_order_approx_naive",
            "second_order_approx",
        ],
    )
    parser.add_argument(
        "--scaling", type=str, choices=["sqrt", "linear", "mixed"], default="mixed"
    )
    parser.add_argument(
        "--log-level", "--ll", dest="log_level", type=str, default="INFO"
    )
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    metadata = {"Subject": " ".join(["python"] + sys.argv)}
    logging.basicConfig(level=args.log_level)
    logging.getLogger("matplotlib").setLevel("INFO")

    assert args.mu_exc >= 0
    assert args.mu_inh <= 0
    assert args.exc_frac == 0.5  # for now

    f = {
        "ricciardi": nn.Ricciardi(scale=1.0),
        "ssn": nn.SSN(),
        "rectified": nn.Rectified(),
    }[args.f]
    vf_bounds = torch.tensor(args.vf_bounds)
    vf = elementwise.bisect(lambda x: f(x) - args.rf, *vf_bounds)
    if vf.isnan():
        raise ValueError("vf is nan. Try with large vf bounds.")
    _dh = elementwise.bisect(lambda x: f(x) - args.rf - args.dh_rate, *vf_bounds)
    if _dh.isnan():
        raise ValueError("_dh is nan. Try with large vf bounds.")
    G = numerics.compute_nth_deriv(f, vf)
    _dG = numerics.compute_nth_deriv(f, vf + _dh) - G

    sigma = args.sigma / math.sqrt(2)  # due to the normalization by Ne rather than N
    args.n = torch.logspace(
        math.log10(args.n_min), math.log10(args.n_max), steps=args.n_steps
    )
    args.n = args.n.round().long().tolist()
    n_trials = [round(args.n_trials / n) for n in args.n]
    t = torch.linspace(args.t_min, args.t_max, args.n_time)

    g = np.full((2, 2), sigma) / G.item()
    gbar = np.array([[args.mu_exc, args.mu_inh], [args.mu_exc, args.mu_inh]]) / G.item()
    g[:, 0] = 0
    g[0, 1] = 0
    gbar[0, 1] = 0

    def phi(x):
        return f(torch.tensor(x)).numpy()

    # u = np.full((2,), vf.item())
    # delta, h0 = solve_mft(phi, u, gbar, g)
    h0 = np.full((2,), 1.0)
    h0[0] = 0
    u, delta = solve_mft2(phi, h0, gbar, g)
    phi0 = integral(phi, u, delta)
    print(f"{g=}")
    print(f"{u=}, {delta=}, {h0=}")
    print(f"{args.rf=}, {phi0=}")

    def dphi(x):
        return numerics.compute_nth_deriv(f, torch.tensor(x)).numpy()

    def ddphi(x):
        out = numerics.compute_nth_deriv(f, torch.tensor(x), n=2)
        out = 0 if x < -10 and out.isnan() else out.numpy()
        return out

    phi_dphi = integral2(phi, dphi, u, delta)
    phi_ddphi = integral2(phi, ddphi, u, delta)
    dphi2 = integral(dphi, u, delta, p=2)
    if args.f == "rectified":
        ddphi0 = np.exp(-(u**2) / (2 * delta)) / math.sqrt(2 * math.pi)
        ddphi0[delta == 0] = 0.0
    else:
        ddphi0 = integral(ddphi, u, delta)
    # print(args.rf * numerics.compute_nth_deriv(f, vf), phi_dphi)
    # print(args.rf * numerics.compute_nth_deriv(f, vf, n=2), phi_ddphi)
    # print(numerics.compute_nth_deriv(f, vf) ** 2, dphi2)
    eye = np.eye(2)
    A = g**2 * ddphi0[:, None] * phi_dphi + eye * integral(dphi, u, delta)
    B = g**2 * ddphi0[:, None] * (dphi2 + phi_ddphi) / 2
    D = g**2 * (dphi2 + phi_ddphi)
    E = 2 * g**2 * phi_dphi
    pop_stab = -np.block([[eye - gbar @ A, -B], [-gbar @ E, eye - D]])
    pop_stab_eigvals = np.linalg.eigvals(pop_stab)
    loc_stab = g**2 * dphi2
    loc_stab_eigvals = np.linalg.eigvals(loc_stab)
    print(pop_stab)
    print(np.max(pop_stab_eigvals.real))
    print(loc_stab)
    print(np.max(np.abs(loc_stab_eigvals)))

    df_eigvals = []
    df_norms = []
    df_resps = []
    df_time_resps = []
    for n, b in zip(args.n, n_trials, strict=True):
        n_e = round(n * args.exc_frac)
        n_i = n - n_e
        GW = sigma * torch.randn((b, n, n))
        GW[..., :n_e] /= n_e if args.scaling == "linear" else n_e**0.5
        GW[..., n_e:] /= n_i if args.scaling == "linear" else n_i**0.5
        GW[..., :n_e] += args.mu_exc / (n_e**0.5 if args.scaling == "sqrt" else n_e)
        GW[..., n_e:] += args.mu_inh / (n_i**0.5 if args.scaling == "sqrt" else n_i)
        W = GW / G
        dh, dG = torch.zeros((n,)), torch.zeros((n,))
        dh[: args.n_perturbed] = _dh
        dG[: args.n_perturbed] = _dG
        L = torch.linalg.inv(torch.eye(n) - GW)
        tL = L - torch.eye(n)

        eigvals = torch.linalg.eigvals(GW)
        eigvals = pd.DataFrame(
            {"n": n, "real": eigvals.real.flatten(), "imag": eigvals.imag.flatten()}
        )
        df_eigvals.append(eigvals)

        # GW_1norm = torch.linalg.matrix_norm(GW, ord=1)
        # GW_2norm = torch.linalg.matrix_norm(GW, ord=2)

        # LdGW = L @ dG.diag() @ W
        # LdGW_1norm = torch.linalg.matrix_norm(LdGW, ord=1)
        # LdGW_2norm = torch.linalg.matrix_norm(LdGW, ord=2)
        # LdGW_infnorm = torch.linalg.matrix_norm(LdGW, ord=torch.inf)
        # LdGW_SR = torch.linalg.eigvals(LdGW).abs().amax(dim=-1)
        # tL_1norm = torch.linalg.matrix_norm(tL, ord=1)
        # tL_2norm = torch.linalg.matrix_norm(tL, ord=2)
        # tL_infnorm = torch.linalg.matrix_norm(tL, ord=torch.inf)
        # tL_mean = tL.abs().mean(dim=(-2, -1))
        # tL_max = tL.abs().amax(dim=(-2, -1))
        # norms = pd.DataFrame(
        #     {
        #         "n": n,
        #         "GW_1norm": GW_1norm,
        #         "GW_2norm": GW_2norm,
        #         "tL_1norm": tL_1norm,
        #         "tL_2norm": tL_2norm,
        #         "tL_infnorm": tL_infnorm,
        #         "tL_mean": tL_mean,
        #         "tL_max": tL_max,
        #         "LdGW_1norm": LdGW_1norm,
        #         "LdGW_2norm": LdGW_2norm,
        #         "LdGW_infnorm": LdGW_infnorm,
        #         "LdGW_SR": LdGW_SR,
        #     }
        # )
        # df_norms.append(norms)

        # batch_idx = torch.arange(b).unsqueeze(-1).broadcast_to((b, n)).flatten()
        # sim = numerics.perturbed_response(
        #     vf,
        #     W,
        #     f,
        #     dh,
        #     t=t,
        #     # options={"max_num_steps": 10000},
        #     # max_t=500.0,
        #     # dx_rtol=1e-5,
        # ).x
        # sim = sim.mean(dim=0).flatten()
        # # sim = sim.flatten()
        # for mode in args.modes:
        #     theory = numerics.perturbed_steady_state_approx(
        #         vf,
        #         tL,
        #         f,
        #         dh,
        #         mode=mode,
        #         min_dv_frac=None,
        #         max_dr_frac=torch.inf,
        #         assert_convergence=False,
        #         max_num_steps=100,
        #     ).flatten()
        #     resps = pd.DataFrame(
        #         {
        #             "mode": mode,
        #             "n": n,
        #             "batch_idx": batch_idx,
        #             "dh": dh.broadcast_to((b, n)).flatten(),
        #             "sim": sim,
        #             "theory": theory,
        #             "rel_err": ((theory - sim) / sim).abs(),
        #         }
        #     )
        #     df_resps.append(resps)

        batch_idx = torch.arange(b)[:, None].broadcast_to((args.n_time, b, n)).flatten()
        unit_idx = torch.arange(n).broadcast_to((args.n_time, b, n)).flatten()
        # dx0 = torch.randn((n,)) * 1e-5
        hf = torch.empty((b, n))
        hf[:, :n_e] = h0[0]
        hf[:, n_e:] = h0[1]
        # x0 = torch.empty((b, n))
        # x0[:, :n_e] = phi0[0]
        # x0[:, n_e:] = phi0[1]
        x0 = torch.randn((b, n))
        x0[:, :n_e] = u[0] + delta[0] ** 0.5 * x0[:, :n_e]
        x0[:, n_e:] = u[1] + delta[1] ** 0.5 * x0[:, n_e:]
        sim0 = numerics.simulate(
            W,
            f,
            hf,
            t=t,
            x0=x0,
            options={"max_num_steps": 10000},
            # max_t=500.0,
            # dx_rtol=1e-5,
            kind="voltage",
        ).x
        # print(phi0, sim0.mean(dim=0)[:n_e].mean(), sim0.mean(dim=0)[n_e:].mean())
        print(
            u,
            delta,
            sim0[args.n_time // 2 :, :, n_e:].mean(),
            sim0[args.n_time // 2 :, :, n_e:].std(),
            phi0,
            f(sim0[args.n_time // 2 :, :, n_e:]).mean(),
            f(sim0[args.n_time // 2 :, :, n_e:]).std(),
        )
        sim = numerics.simulate(
            W, f, hf + dh, t=t, x0=x0, options={"max_num_steps": 10000}
        ).x
        sim = sim - sim0
        # sim0 = numerics.simulate(
        #     vf, W, f, torch.zeros((n,)), t=t, options={"max_num_steps": 10000}
        # ).x
        # sim = numerics.perturbed_response(
        #     vf, W, f, dh, t=t, options={"max_num_steps": 10000}
        # ).x
        assert sim.shape == (args.n_time, b, n)
        cell_type = torch.cat([torch.zeros((n_e,)), torch.ones((n_i,))])
        time_resps = pd.DataFrame(
            {
                "n": n,
                "t": t[:, None, None].broadcast_to((args.n_time, b, n)).flatten(),
                "cell_type": cell_type.broadcast_to((args.n_time, b, n)).flatten(),
                "batch_idx": batch_idx,
                "unit_idx": unit_idx,
                "dh": dh.broadcast_to((args.n_time, b, n)).flatten(),
                "sim0": sim0.flatten(),
                "sim": sim.flatten(),
            }
        )
        df_time_resps.append(time_resps)

    df_eigvals = pd.concat(df_eigvals, ignore_index=True)
    # df_norms = pd.concat(df_norms, ignore_index=True)
    # df_resps = pd.concat(df_resps, ignore_index=True)
    df_time_resps = pd.concat(df_time_resps, ignore_index=True)

    g = viz.figplot(
        # df_time_resps,
        df_time_resps.query("cell_type == 1"),
        func="relplot",
        kind="line",
        x="t",
        y="sim0",
        row="n",
        col="batch_idx",
        units="unit_idx",
        estimator=None,
        height=2,
        aspect=1.0,
        facet_kws={"sharey": False},
    )
    if args.show:
        plt.show()
    elif args.out:
        g.savefig(args.out / "response_v_time_no_input.pdf", metadata=metadata)
        plt.clf()

    g = viz.figplot(
        # df_time_resps.query(f"dh < {_dh.item() / 2}"),
        df_time_resps.query(f"dh < {_dh.item() / 2} and cell_type == 1"),
        func="relplot",
        kind="line",
        x="t",
        y="sim",
        row="n",
        col="batch_idx",
        units="unit_idx",
        estimator=None,
        height=2,
        aspect=1.0,
        facet_kws={"sharey": False},
    )
    if args.show:
        plt.show()
    elif args.out:
        g.savefig(args.out / "response_v_time.pdf", metadata=metadata)
        plt.clf()

    # g = viz.figplot(
    #     df_resps.query("mode != 'analytical'"),
    #     func="relplot",
    #     kind="line",
    #     x="n",
    #     y="rel_err",
    #     hue="mode",
    #     col="dh",
    #     hue_order=[mode for mode in args.modes if mode != "analytical"],
    #     xscale="log",
    #     yscale="log",
    #     errorbar="se",
    #     height=2,
    #     aspect=1.0,
    #     facet_kws={"sharey": False},
    # )

    # if args.show:
    #     plt.show()
    # elif args.out:
    #     g.savefig(args.out / "rel_err.pdf", metadata=metadata)
    #     plt.clf()

    # g = viz.figplot(
    #     df_resps.query(f"dh < {_dh.item() / 2}"),
    #     func="relplot",
    #     x="theory",
    #     y="sim",
    #     col="n",
    #     hue="mode",
    #     hue_order=args.modes,
    #     col_wrap=5,
    #     height=2,
    #     aspect=1.0,
    #     facet_kws={"sharey": False, "sharex": False},
    # )
    # for ax in g.axes.flat:
    #     xlim, ylim = ax.get_xlim(), ax.get_ylim()
    #     lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    #     ax.plot(lim, lim, ls="--", linewidth=GRID_WIDTH, color=GRID_COLOR)

    # if args.show:
    #     plt.show()
    # elif args.out:
    #     g.savefig(args.out / "response.pdf", metadata=metadata)
    #     plt.clf()

    # df_norms["log10(n)"] = np.log10(df_norms["n"])
    # for y in [
    #     "GW_1norm",
    #     "GW_2norm",
    #     "tL_1norm",
    #     "tL_2norm",
    #     "tL_infnorm",
    #     "tL_mean",
    #     "tL_max",
    #     "LdGW_1norm",
    #     "LdGW_2norm",
    #     "LdGW_infnorm",
    #     "LdGW_SR",
    # ]:
    #     df_norms[f"log10({y})"] = np.log10(df_norms[y])
    #     g = viz.figplot(
    #         df_norms.query("n >= 100") if y.startswith("LdGW") else df_norms,
    #         func="lmplot",
    #         x="log10(n)",
    #         y=f"log10({y})",
    #         x_estimator=np.median,
    #         statannot=True,
    #     )

    #     if args.show:
    #         plt.show()
    #     elif args.out:
    #         g.savefig(args.out / f"{y}.pdf", metadata=metadata)
    #         plt.clf()

    lim = df_eigvals[["real", "imag"]].abs().max().max() * 1.05
    g = viz.figplot(
        df_eigvals,
        func="relplot",
        x="real",
        y="imag",
        col="n",
        height=3,
        aspect=1,
        col_wrap=5,
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        s=4,
    )
    for ax in g.axes.flat:
        ax.set_aspect("equal")
        ax.add_patch(plt.Circle((0, 0), 1.0, color="gray", fill=False))
        ax.add_patch(plt.Circle((0, 0), args.sigma, color="gray", ls="--", fill=False))

    if args.show:
        plt.show()
    elif args.out:
        g.savefig(args.out / "eigvals.pdf", metadata=metadata)
        plt.clf()


if __name__ == "__main__":
    main()
