import argparse
from pathlib import Path
import random

from tqdm import trange
import pandas as pd
import numpy as np
from scipy import optimize
import seaborn as sns
import matplotlib.pyplot as plt


def sigma_con(params: np.ndarray) -> float:
    _, _, s0, s1, _, _ = params
    return s1 - s0  # s1 >= s0


def kappa_con(params: np.ndarray) -> float:
    _, _, _, _, k0, k1 = params
    return k0 - k1  # k0 == k1


def response(a0, a1, s0, s1, k0, k1, r, theta):
    # fmt: off
    out = (
        a0 * (1 + 2 * k0 * np.cos(theta)) * np.exp(-(r**2) / (2 * s0**2))
        - a1 * (1 + 2 * k1 * np.cos(theta)) * np.exp(-(r**2) / (2 * s1**2))
    )
    # fmt: on
    return out


def func(params: np.ndarray, x: pd.DataFrame) -> pd.DataFrame:
    x["dr"] = response(*params, x["_distance"], x["_rel_ori"] / 90.0 * np.pi)
    x = x.groupby(["distance", "rel_ori"], as_index=False, observed=True)["dr"].mean()
    return x


def loss_func(params: np.ndarray, x: pd.DataFrame, y: pd.DataFrame) -> float:
    out = func(params, x).merge(y, on=["distance", "rel_ori"], validate="1:1")
    return np.linalg.vector_norm(out["dr_x"] - out["dr_y"]) ** 2


def callback(params: np.ndarray):
    print(sigma_con(params))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--n-trials", "-n", type=int, default=10)
    parser.add_argument("-R", type=float, default=600.0)
    parser.add_argument("--density", type=float, default=29643.0)  # per mm^2
    parser.add_argument("--a-bounds", type=float, nargs=2, default=[0.0, 5.0])
    parser.add_argument("--s-bounds", type=float, nargs=2, default=[100.0, 300.0])
    parser.add_argument("--k0-bounds", type=float, nargs=2, default=[0.0, 0.5])
    parser.add_argument("--k1-bounds", type=float, nargs=2, default=[-0.5, 0.5])
    parser.add_argument("--equal-kappa", type=str, nargs="*")
    parser.add_argument("--ftol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--verbosity", "-v", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    data = pd.read_pickle(args.data).query("cell_type == 'L4'")
    print(data.attrs["command"])
    # print(data)

    if args.equal_kappa is not None and len(args.equal_kappa) == 0:
        args.equal_kappa = data["stim_type"].unique().tolist()
    elif args.equal_kappa is None:
        args.equal_kappa = []

    N = round(args.density * np.pi * (args.R / 1000.0) ** 2)
    print(f"Total neurons within {args.R} μm: {N}")
    df_x = pd.DataFrame(
        {
            "_distance": np.sqrt(np.random.rand(N)) * args.R,
            "_rel_ori": np.random.uniform(0.0, 90.0, size=(N,)),
        }
    )
    df_x["distance"] = pd.cut(df_x["_distance"], data["distance"].cat.categories)
    df_x["rel_ori"] = pd.cut(df_x["_rel_ori"], data["rel_ori"].cat.categories)

    r, theta = np.linspace(0, args.R), np.array([0.0, np.pi])
    r, theta = np.meshgrid(r, theta, indexing="ij")
    r, theta = r.flatten(), theta.flatten()

    bounds = (
        [args.a_bounds] * 2 + [args.s_bounds] * 2 + [args.k0_bounds, args.k1_bounds]
    )
    df = {}
    for stim_type, df_y in data.groupby("stim_type", observed=True):
        print(f"Fitting for {stim_type}...")
        constraints = [{"type": "ineq", "fun": sigma_con}]
        if stim_type in args.equal_kappa:
            constraints.append({"type": "eq", "fun": kappa_con})

        loss, res = float("inf"), None
        for _ in trange(args.n_trials, disable=not args.progress):
            p0 = [
                random.uniform(*args.a_bounds),
                random.uniform(*args.a_bounds),
                random.uniform(*args.s_bounds),
                random.uniform(*args.s_bounds),
                random.uniform(*args.k0_bounds),
                random.uniform(*args.k1_bounds),
            ]
            new_res = optimize.minimize(
                loss_func,
                p0,
                args=(df_x, df_y),
                bounds=bounds,
                constraints=constraints,
                # callback=callback,
                options={"ftol": args.ftol, "maxiter": args.maxiter},
            )
            new_loss = loss_func(new_res.x, df_x, df_y)
            if not new_res.success:
                print(new_res.message)
            if args.verbosity > 0:
                with np.printoptions(formatter={"float_kind": "{:.3g}".format}):
                    print(f"Loss: {new_loss:.3g}, Result: {new_res.x}")
            if new_res.success and new_loss < loss:
                loss, res = new_loss, new_res
        with np.printoptions(formatter={"float_kind": "{:.3g}".format}):
            print(f"Final loss: {loss:.3g}, Final result: {res.x}")
        dr = response(*res.x, r, theta)
        df[stim_type] = pd.DataFrame({"r": r, "theta": theta, "dr": dr})
    df = pd.concat(df, names=["stim_type"]).reset_index(0).reset_index(drop=True)
    g = sns.relplot(
        df, x="r", y="dr", hue="theta", col="stim_type", height=2, kind="line"
    )
    g.refline(y=0.0)
    plt.show()


if __name__ == "__main__":
    main()
