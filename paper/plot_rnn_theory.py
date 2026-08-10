import argparse
import math
import logging
from pathlib import Path
import sys
from itertools import product

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from niarb import viz, nn, numerics, exceptions, random
from niarb.optimize import elementwise
from mpl_config import set_rcParams, GRID_WIDTH, GRID_COLOR

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_rcParams()


def eigvals(GW):
    eigvals = torch.linalg.eigvals(GW)
    return pd.DataFrame(
        {"real": eigvals.real.flatten().cpu(), "imag": eigvals.imag.flatten().cpu()}
    )


def norm(W, GW, dG, L, tL):
    GW_1norm = torch.linalg.matrix_norm(GW, ord=1)
    GW_2norm = torch.linalg.matrix_norm(GW, ord=2)

    LdGW = L @ dG.diag() @ W
    LdGW_1norm = torch.linalg.matrix_norm(LdGW, ord=1)
    LdGW_2norm = torch.linalg.matrix_norm(LdGW, ord=2)
    LdGW_infnorm = torch.linalg.matrix_norm(LdGW, ord=torch.inf)
    LdGW_SR = torch.linalg.eigvals(LdGW).abs().amax(dim=-1)
    tL_1norm = torch.linalg.matrix_norm(tL, ord=1)
    tL_2norm = torch.linalg.matrix_norm(tL, ord=2)
    tL_infnorm = torch.linalg.matrix_norm(tL, ord=torch.inf)
    tL_mean = tL.abs().mean(dim=(-2, -1))
    tL_max = tL.abs().amax(dim=(-2, -1))
    return pd.DataFrame(
        {
            "GW_1norm": GW_1norm.cpu(),
            "GW_2norm": GW_2norm.cpu(),
            "tL_1norm": tL_1norm.cpu(),
            "tL_2norm": tL_2norm.cpu(),
            "tL_infnorm": tL_infnorm.cpu(),
            "tL_mean": tL_mean.cpu(),
            "tL_max": tL_max.cpu(),
            "LdGW_1norm": LdGW_1norm.cpu(),
            "LdGW_2norm": LdGW_2norm.cpu(),
            "LdGW_infnorm": LdGW_infnorm.cpu(),
            "LdGW_SR": LdGW_SR.cpu(),
        }
    )


def resp(
    W, tL, vf, f, dh, modes, t, max_num_steps=10000, dx_rtol=1e-4, dx_atol=5e-6,
):
    b, n = W.shape[0], W.shape[-1]
    batch_idx = torch.arange(b, device=W.device).unsqueeze(-1).broadcast_to((b, n))
    options = {"max_num_steps": max_num_steps}
    if t is None:
        sim = numerics.perturbed_response(
            vf, W, f, dh, options=options, dx_rtol=dx_rtol, dx_atol=dx_atol
        ).x
        converged = torch.ones((b,), dtype=torch.bool, device=device)
    else:
        t = torch.tensor([0.0, t], device=device)
        out = numerics.perturbed_response(vf, W, f, dh, t=t, options=options)
        x, dxdt = out.x[-1], out.dxdt[-1]
        # converged = dxdt.mean(dim=-1).abs() <= 1e-5
        logger.info(
            f"median median: {x.abs().quantile(0.5, dim=-1).quantile(0.5).item():.4e}, "
            f"median min: {x.abs().amin(dim=-1).quantile(0.5).item():.4e}, "
        )
        converged = (dxdt.abs() <= x.abs() * dx_rtol + dx_atol).all(dim=-1)
        assert converged.shape == (b,)
        logger.info(f"Num batches converged: {converged.count_nonzero().item()}/{b}")
        sim = x[converged, :]
        assert sim.shape == (converged.count_nonzero().item(), n)

    dfs = []
    for mode in modes:
        theory = numerics.perturbed_steady_state_approx(
            vf,
            tL,
            f,
            dh,
            mode=mode,
            min_dv_frac=None,
            max_dr_frac=torch.inf,
            assert_convergence=False,
            max_num_steps=100,
        )
        theory = theory[converged, :]
        assert theory.shape == (converged.count_nonzero().item(), n)
        df = pd.DataFrame(
            {
                "mode": mode,
                "batch_idx": batch_idx[converged, :].flatten().cpu(),
                "dh": dh.broadcast_to((b, n))[converged, :].flatten().cpu(),
                "sim": sim.flatten().cpu(),
                "theory": theory.flatten().cpu(),
                "rel_err": ((theory - sim) / sim).abs().flatten().cpu(),
            }
        )
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def time_resp(W, vf, f, dh, t):
    b, n = W.shape[0], W.shape[-1]
    n_time = len(t)
    batch_idx = torch.arange(b)[:, None].broadcast_to((n_time, b, n)).flatten()
    unit_idx = torch.arange(n).broadcast_to((n_time, b, n)).flatten()
    # dx0 = torch.randn((n,)) * 1e-5
    # hf = torch.empty((b, n))
    # hf[:, :n_e] = h0[0]
    # hf[:, n_e:] = h0[1]
    # x0 = torch.empty((b, n))
    # x0[:, :n_e] = phi0[0]
    # x0[:, n_e:] = phi0[1]
    # sim0 = numerics.simulate(
    #     W,
    #     f,
    #     hf,
    #     t=t,
    #     x0=x0,
    #     options={"max_num_steps": 10000},
    #     # max_t=500.0,
    #     # dx_rtol=1e-5,
    # ).x
    # sim = numerics.simulate(
    #     W, f, hf + dh, t=t, x0=x0, options={"max_num_steps": 10000}
    # ).x
    sim0 = numerics.simulate(
        vf, W, f, torch.zeros((n,)), t=t, options={"max_num_steps": 10000}
    ).x
    sim = numerics.perturbed_response(
        vf, W, f, dh, t=t, options={"max_num_steps": 10000}
    ).x
    assert sim.shape == (n_time, b, n)
    sim = sim - sim0
    # cell_type = torch.cat([torch.zeros((n_e,)), torch.ones((n_i,))])
    return pd.DataFrame(
        {
            "t": t[:, None, None].broadcast_to((n_time, b, n)).flatten().cpu(),
            # "cell_type": cell_type.broadcast_to((n_time, b, n)).flatten(),
            "batch_idx": batch_idx,
            "unit_idx": unit_idx,
            "dh": dh.broadcast_to((n_time, b, n)).flatten().cpu(),
            "sim0": sim0.flatten().cpu(),
            "sim": sim.flatten().cpu(),
        }
    )


def plot_eigvals(df, sigma, axes, **kwargs):
    assert axes == ["n"]
    lim = df[["real", "imag"]].abs().max().max() * 1.05
    gs = {}

    gs["eigvals"] = viz.figplot(
        df,
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
        **kwargs,
    )
    for ax in gs["eigvals"].axes.flat:
        ax.set_aspect("equal")
        ax.add_patch(plt.Circle((0, 0), 1.0, color="gray", fill=False))
        ax.add_patch(plt.Circle((0, 0), sigma[0], color="gray", ls="--", fill=False))

    return gs


def plot_norm(df, axes, **kwargs):
    assert axes == ["n"]
    df["log10(n)"] = np.log10(df["n"])
    gs = {}

    for y in [
        "GW_1norm",
        "GW_2norm",
        "tL_1norm",
        "tL_2norm",
        "tL_infnorm",
        "tL_mean",
        "tL_max",
        "LdGW_1norm",
        "LdGW_2norm",
        "LdGW_infnorm",
        "LdGW_SR",
    ]:
        df[f"log10({y})"] = np.log10(df[y])
        gs[y] = viz.figplot(
            df.query("n >= 100") if y.startswith("LdGW") else df,
            func="lmplot",
            x="log10(n)",
            y=f"log10({y})",
            x_estimator=np.median,
            statannot=True,
            **kwargs,
        )
    return gs


def plot_resp(df: pd.DataFrame, modes, axes, kind, **kwargs):
    mode_mapping = {
        "linear_approx": "Linear",
        "quasi_linear_approx": "Quasi-linear",
        "second_order_approx_naive": "2nd order (naive)",
        "second_order_approx": "2nd order",
    }
    df["perturbed"] = df["dh"] > 0
    df["mode"] = df["mode"].replace(mode_mapping)
    modes = [mode_mapping.get(m, m) for m in modes]
    gs = {}

    for perturbed, sf in df.groupby("perturbed", observed=True):
        suffix = "perturbed" if perturbed else "unperturbed"

        if "n" in axes and kind == "rel_err":
            assert all(k not in kwargs for k in ["col", "row"])
            ax_kwargs = dict(zip(["col", "row"], [ax for ax in axes if ax != "n"]))
            gs[f"rel_err_n_{suffix}"] = viz.figplot(
                sf,
                func="relplot",
                kind="line",
                x="n",
                y="rel_err",
                hue="mode",
                hue_order=modes,
                xscale="log",
                yscale="log",
                estimator="median",
                errordim="batch_idx",
                height=1.5,
                aspect=1.0,
                # facet_kws={"sharey": False},
                **ax_kwargs,
                **kwargs,
            )

        if len(axes) >= 2 and kind == "rel_err":
            assert all(k not in kwargs for k in ["x", "y", "row"])
            ax_kwargs = dict(zip(["x", "y", "row"], axes))
            gs[f"rel_err_grid_{suffix}"] = viz.figplot(
                sf,
                func="heatplot",
                hue="rel_err",
                col="mode",
                col_order=modes,
                estimator="median",
                hue_scale="log",
                height=1.5,
                aspect=1.0,
                **ax_kwargs,
                **kwargs,
            )

        if len(axes) <= 2 and kind == "response":
            ax_kwargs = dict(zip(["col", "row"], axes))
            gs[f"response_{suffix}"] = viz.figplot(
                sf,
                func="relplot",
                x="theory",
                y="sim",
                hue="mode",
                hue_order=modes,
                height=1.5,
                aspect=1.0,
                facet_kws={"sharey": False, "sharex": False},
                s=4,
                **ax_kwargs,
                **kwargs,
            )
            for ax in gs[f"response_{suffix}"].axes.flat:
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
                ax.plot(lim, lim, ls="--", linewidth=GRID_WIDTH, color=GRID_COLOR)

    return gs


def plot_time_resp(df, axes, **kwargs):
    assert axes == ["n"]
    gs = {}

    gs["response_v_time_no_input"] = viz.figplot(
        df,
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
        **kwargs,
    )

    gs["response_v_time"] = viz.figplot(
        df.query("dh == 0"),
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
        **kwargs,
    )

    return gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "plot", type=str, choices=["eigvals", "norm", "resp", "time_resp"]
    )
    parser.add_argument("-n", type=int, nargs="+")
    parser.add_argument("--n-min", type=int, default=300)
    parser.add_argument("--n-max", type=int, default=3000)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--n-time", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=10000)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=50.0)
    parser.add_argument("--use-t", action="store_true")
    parser.add_argument(
        "--exc-frac", "--frac", dest="exc_frac", type=float, default=0.8
    )
    parser.add_argument("--mu", type=float, nargs="+", default=[1.0])
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.5])
    parser.add_argument("--sigma", "-s", type=float, nargs="+", default=[0.15])
    parser.add_argument(
        "--dh-rate", "--dh", dest="dh_rate", type=float, nargs="+", default=[10.0]
    )
    parser.add_argument("--n-perturbed", type=int, nargs="+", default=[10])
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
            # "analytical",
            "linear_approx",
            "quasi_linear_approx",
            "second_order_approx_naive",
            "second_order_approx",
        ],
    )
    parser.add_argument(
        "--scaling", type=str, choices=["sqrt", "linear", "mixed"], default="mixed"
    )
    parser.add_argument("--log-normal", "--ln", dest="log_normal", action="store_true")
    parser.add_argument("--kind", type=str, choices=["rel_err", "response"], default="rel_err")
    parser.add_argument(
        "--log-level", "--ll", dest="log_level", type=str, default="INFO"
    )
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    metadata = {"Subject": " ".join(["python"] + sys.argv)}
    logging.basicConfig(level=args.log_level)
    logging.getLogger("matplotlib").setLevel("INFO")

    assert all(mu >= 0 for mu in args.mu)
    assert all(alpha >= 0 for alpha in args.alpha)

    f = {
        "ricciardi": nn.Ricciardi(scale=1.0),
        "ssn": nn.SSN(),
        "rectified": nn.Rectified(),
    }[args.f]
    vf_bounds = torch.tensor(args.vf_bounds)
    vf = elementwise.bisect(lambda x: f(x) - args.rf, *vf_bounds)
    if vf.isnan():
        raise ValueError("vf is nan. Try with large vf bounds.")

    if args.n is None:
        args.n = torch.logspace(
            math.log10(args.n_min), math.log10(args.n_max), steps=args.n_steps
        )
        args.n = args.n.round().long().tolist()
    t = torch.linspace(args.t_min, args.t_max, args.n_time, device=device)

    names = ["n", "mu", "alpha", "sigma", "dh_rate", "n_perturbed"]
    axes = [name for name in names if len(getattr(args, name)) > 1]
    if len(axes) > 3:
        raise ValueError("At most 3 axes allowed.")

    dfs = []
    for params in product(*[getattr(args, name) for name in names]):
        logger.info(f"Running params: {dict(zip(names, params))}")
        n, mu, alpha, sigma, dh_rate, n_perturbed = params
        b = round(args.n_trials / n)
        n_e = round(n * args.exc_frac)
        mu_exc = mu / args.exc_frac
        mu_inh = -mu * alpha / (1 - args.exc_frac)

        dh_v = elementwise.bisect(lambda x: f(x) - args.rf - dh_rate, *vf_bounds)
        if dh_v.isnan():
            raise ValueError("dh_v is nan. Try with large vf bounds.")
        dh_v = dh_v - vf
        G = numerics.compute_nth_deriv(f, vf)
        _dG = numerics.compute_nth_deriv(f, vf + dh_v) - G

        if args.log_normal:
            mean_exc = mu_exc / (n**0.5 if args.scaling == "sqrt" else n)
            mean_inh = mu_inh / (n**0.5 if args.scaling == "sqrt" else n)
            std = sigma / (n if args.scaling == "linear" else n**0.5)
            mean = torch.cat(
                [torch.full((n_e,), mean_exc), torch.full((n - n_e,), -mean_inh)]
            )
            mean = mean.to(device)
            GW = random.log_normal_strict(
                mean, torch.tensor(std, device=device), size=(b, n)
            )
            GW[..., n_e:] = -GW[..., n_e:]

        else:
            GW = sigma * torch.randn((b, n, n), device=device)
            GW = GW / (n if args.scaling == "linear" else n**0.5)
            GW[..., :n_e] += mu_exc / (n**0.5 if args.scaling == "sqrt" else n)
            GW[..., n_e:] += mu_inh / (n**0.5 if args.scaling == "sqrt" else n)
        W = GW / G
        dh, dG = torch.zeros((n,), device=device), torch.zeros((n,), device=device)
        dh[:n_perturbed] = dh_v.item()
        dG[:n_perturbed] = _dG.item()
        eye = torch.eye(n, device=device)
        L = torch.linalg.inv(eye - GW)
        tL = L - eye

        if args.plot == "eigvals":
            df = eigvals(GW)
        elif args.plot == "norm":
            df = norm(W, GW, dG, L, tL)
        elif args.plot == "resp":
            T = args.t_max if args.use_t else None
            try:
                df = resp(W, tL, vf.to(device), f, dh, args.modes, T)
            except exceptions.SimulationError as err:
                logger.warning(f"{err}. Skipping this param set.")
                continue
        elif args.plot == "time_resp":
            df = time_resp(W, vf.to(device), f, dh, t)

        for name, param in zip(names, params, strict=True):
            df[name] = param

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    mapping = {
        "rel_err": "Rel. error",
        "n": "N",
        "alpha": r"\alpha",
        "mu": r"$\mu$",
        "sigma": r"$\sigma$",
        "dh_rate": r"$\Delta h'$",
        "n_perturbed": r"$N_\mathrm{perturbed}$",
        "real": r"$\mathrm{Re}(\lambda)$",
        "imag": r"$\mathrm{Im}(\lambda)$",
        "mode": "Approx.",
    }
    if args.plot == "eigvals":
        gs = plot_eigvals(df, args.sigma, axes, mapping=mapping)
    elif args.plot == "norm":
        gs = plot_norm(df, axes, mapping=mapping)
    elif args.plot == "resp":
        gs = plot_resp(df, args.modes, axes, args.kind, mapping=mapping)
    elif args.plot == "time_resp":
        gs = plot_time_resp(df, axes, mapping=mapping)

    if args.out:
        args.out.mkdir(exist_ok=True, parents=True)
        for name, g in gs.items():
            g.savefig(args.out / f"{name}.pdf", metadata=metadata)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
