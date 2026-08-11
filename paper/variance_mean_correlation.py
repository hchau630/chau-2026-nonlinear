import argparse
import logging
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from niarb import viz


def residual(x, y, keep_mean=True):
    isnan = np.isnan(x) | np.isnan(y)
    reg = stats.linregress(x[~isnan], y[~isnan])
    y_res = y - (reg.intercept + reg.slope * x)
    if keep_mean:
        y_res = y_res - y_res.mean() + y.mean()
    return y_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", "-f", type=Path, default="ogando_data/preprocessed.pkl"
    )
    parser.add_argument("--estimator", type=str, default="median")
    parser.add_argument("--kind", type=str, choices=["var", "std"], default="var")
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--ens-size", "-s", type=int, nargs=2, default=(9, 11))
    parser.add_argument("--min-N", type=int, default=2)
    parser.add_argument("--max-dist", type=float, default=75.0)
    parser.add_argument("--partial-ens-size", "-p", action="store_true")
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--method", "-m", type=str, default="bootstrap")
    parser.add_argument("--n-resamples", "-n", type=int, default=99999)
    parser.add_argument("--log-level", "-l", type=str, default="INFO")
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger()

    if args.out:
        args.out.mkdir(exist_ok=True, parents=True)

    kind = args.kind
    df: pd.DataFrame = pd.read_pickle(args.filename)

    if args.query:
        df = df.query(args.query).copy()

    df["is_nearby_dr"] = df["dr"]
    df.loc[df["_distance"] >= args.max_dist, "is_nearby_dr"] = pd.NA
    df = df.groupby(
        [
            "mouse_id",
            "session_id",
            "exp_type",
            "holo_id",
            "cell_type",
            "_holo_osi",
            "_N",
        ],
        as_index=False,
        observed=True,
    ).agg(
        **{
            "dr": ("is_nearby_dr", args.estimator),
            # kind: ("dr", kind),
            kind: ("is_nearby_dr", kind),
            "count": ("dr", "count"),
        }
    )
    df = df.query(f"~dr.isna() and count >= {args.min_N}").copy()
    df = df.pivot(
        index=["exp_type", "holo_id", "_holo_osi", "_N"],
        columns="cell_type",
        values=["dr", kind],
    )
    df.columns = ["_".join(c) for c in df.columns.to_flat_index()]
    df = df.reset_index()

    df = df.query(f"_N >= {args.ens_size[0]}").copy()
    if args.ens_size[1] >= 0:
        df = df.query(f"_N <= {args.ens_size[1]}").copy()
    if args.partial_ens_size:
        for cell_type, statistic in product(["PYR", "PV", "SST"], ["dr", kind]):
            col = f"{statistic}_{cell_type}"
            if statistic == kind and args.logx:
                df[col] = np.log10(df[col])
            df[col] = residual(df["_N"], df[col])
            if statistic == kind and args.logx:
                df[col] = 10 ** df[col]

    mapping = {
        "std_PV": "PV response s.d.",
        "std_SST": "SST response s.d.",
        "var_PV": r"PV variance",
        "var_SST": r"SST variance",
        "dr_PYR": "Pyr response",
        "dr_PV": "PV response",
        "dr_SST": "SST response",
    }
    if args.partial_ens_size:
        mapping = {k: f"{v} res." for k, v in mapping.items()}

    kwargs = {
        "logx": args.logx,
        "xscale": "log" if args.logx else "linear",
        "n_boot": args.n_resamples,
        "seed": 0,
        "statannot": True,
        "statannot_kws": {
            "method": args.method,
            "n_resamples": args.n_resamples,
            "rng": 0,
            "verbosity": -1,
            "frameon": False,
        },
        "rc_params": {
            "figure.titlesize": 7.25,
            "font.size": 7.25,  # default: 10 pts
            "axes.labelpad": 0.0,  # default: 4.0 pts
            "axes.titlepad": 4.0,  # default: 6.0 pts
            "lines.markersize": 3,  # default: 6 pts
            "lines.markeredgewidth": 0.5,  # default: 1.0
        },
        "scatter_kws": {"edgecolor": "none", "facecolor": "black"},
        "line_kws": {"color": "black"},
        "height": "paper",
        "aspect": 1.1,
        "mapping": mapping,
    }

    for cell_type in ["PV", "SST"]:
        logger.info(f"Plotting dr_PYR vs {kind}_{cell_type}...")
        viz.figplot(
            df.query(f"~{kind}_{cell_type}.isna() and ~dr_PYR.isna()").copy(),
            func="lmplot",
            x=f"{kind}_{cell_type}",
            y="dr_PYR",
            **kwargs,
        )
        if args.out:
            plt.savefig(
                args.out / f"{cell_type}.pdf",
                metadata={"Subject": " ".join(["python"] + sys.argv)},
            )
        if args.show:
            plt.show()
        else:
            plt.close()

    for cell_type in ["PV", "SST"]:
        logger.info(f"Plotting dr_{cell_type} vs {kind}_{cell_type}...")
        viz.figplot(
            df.query(f"~{kind}_{cell_type}.isna() and ~dr_{cell_type}.isna()").copy(),
            func="lmplot",
            x=f"{kind}_{cell_type}",
            y=f"dr_{cell_type}",
            **kwargs,
        )
        if args.out:
            plt.savefig(
                args.out / f"{cell_type}_self.pdf",
                metadata={"Subject": " ".join(["python"] + sys.argv)},
            )
        if args.show:
            plt.show()
        else:
            plt.close()

    if args.logx:
        # mean-mean plots make no sense with log x-axis
        return

    for cell_type in ["PV", "SST"]:
        logger.info(f"Plotting dr_PYR vs dr_{cell_type}...")
        viz.figplot(
            df.query(f"~dr_{cell_type}.isna() and ~dr_PYR.isna()").copy(),
            func="lmplot",
            x=f"dr_{cell_type}",
            y="dr_PYR",
            **kwargs,
        )
        if args.out:
            plt.savefig(
                args.out / f"{cell_type}_mean.pdf",
                metadata={"Subject": " ".join(["python"] + sys.argv)},
            )
        if args.show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()
