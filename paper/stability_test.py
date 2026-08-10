from argparse import ArgumentParser
from pathlib import Path

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

from niarb import nn, neurons, exceptions


def main():
    parser = ArgumentParser()
    parser.add_argument("mode", type=str, choices=["run", "plot"])
    parser.add_argument("-p", "--path", type=Path, default=Path("stability_test.pkl"))
    parser.add_argument("-M", type=int, default=1000)
    parser.add_argument("-K", type=int, default=100)
    parser.add_argument("-T", type=float, default=500.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max-gW", type=float, default=20.0)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--max-num-steps", type=int, default=10000)
    parser.add_argument("-s", type=float, default=4.0)
    args = parser.parse_args()
    if args.mode == "run":
        run(
            args.path,
            args.M,
            args.K,
            args.T,
            args.tau,
            args.max_gW,
            args.rtol,
            args.atol,
            args.max_num_steps,
        )
    else:
        plot(args.path, args.s)


def run(path: Path, M, K, T, tau, max_gW, rtol, atol, max_num_steps):
    cell_types = ["PYR", "PV", "SST"]
    n = len(cell_types)

    x = neurons.as_grid(n=n, cell_types=cell_types)  # (n,)
    x = x.unsqueeze(0)  # (1, n)
    model = nn.V1(
        ["cell_type"],
        cell_types=cell_types,
        tau=[1.0, tau, tau],
        f="SSN",
        null_connections=[],
        init_gW_bounds=[0.0, max_gW],
        mode="numerical",
        simulation_kwargs={"options": {"max_num_steps": max_num_steps}},
    )

    df = []
    for _ in trange(M):
        model.reset_parameters()
        x["dh"] = torch.randn(K, n)
        with torch.inference_mode():
            try:
                y = model(x, t=[T - 1, T])  # (2, K, n)
            except exceptions.SimulationError:
                success = False
            else:
                success = torch.allclose(y["dr"][0], y["dr"][1], rtol=rtol, atol=atol)
        W = model.W()
        df.append(
            {
                "det(W)": W.det().item(),
                "Mpp": W[::2, ::2].det().item(),
                "Mss": W[:2, :2].det().item(),
                "Mes": W[1:, :2].det().item(),
                "Mep": W[1:, ::2].det().item(),
                "success": success,
            }
        )
    df = pd.DataFrame(df)
    if path.is_file():
        df = pd.concat([pd.read_pickle(path), df], ignore_index=True)
    df.to_pickle(path)


def plot(path, s):
    df = pd.read_pickle(path)
    print(
        (
            df.eval(
                "`det(W)` > 0 and ((Mpp > 0 and Mes > 0 and Mep < 0) or (Mss > 0 and Mes < 0 and Mep > 0))"
            )
            & ~df.eval("`det(W)` > 0 and Mpp > 0 and Mss > 0")
        ).sum()
    )
    for cond in [
        "`det(W)` > 0",
        "Mpp > 0 or Mss > 0",
        "`det(W)` > 0 and (Mpp > 0 or Mss > 0)",
        "`det(W)` > 0 and Mpp > 0",
        "`det(W)` > 0 and Mss > 0",
        "Mpp > 0 and Mss > 0",
        "`det(W)` > 0 and Mpp > 0 and Mss > 0",
        "`det(W)` > 0 and ((Mpp > 0 and Mes > 0 and Mep < 0) or (Mss > 0 and Mes < 0 and Mep > 0))",
    ]:
        df[cond] = df.eval(cond)
        N_failed = (df[cond] & ~df["success"]).sum()
        N_total = df[cond].sum()
        df[cond] = df[cond].replace({True: "True", False: "False"})
        df[cond] = pd.Categorical(df[cond], categories=["False", "True"], ordered=True)
        g = sns.displot(df, x=cond, hue="success", multiple="stack")
        g.figure.suptitle(f"Number of failed predictions: {N_failed}/{N_total}")
        plt.show()
    g = sns.relplot(df, x="det(W)", y="Mss", hue="success", s=s)
    g.refline(x=0, y=0)
    plt.show()
    g = sns.relplot(df, x="det(W)", y="Mpp", hue="success", s=s)
    g.refline(x=0, y=0)
    plt.show()
    g = sns.relplot(df, x="Mss", y="Mes", hue="success", s=s)
    g.refline(x=0, y=0)
    plt.show()
    g = sns.relplot(df, x="Mpp", y="Mes", hue="success", s=s)
    g.refline(x=0, y=0)
    plt.show()
    g = sns.relplot(df, x="Mes", y="Mep", hue="success", s=s)
    g.refline(x=0, y=0)
    plt.show()


if __name__ == "__main__":
    main()
