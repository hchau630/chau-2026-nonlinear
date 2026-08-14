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
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--use-old-N", action="store_true")
    parser.add_argument("--use-old-distance", action="store_true")
    parser.add_argument("--estimator", type=str, default="median")
    parser.add_argument("--kind", type=str, choices=["var", "std"], default="var")
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--ens-size", "-s", type=int, nargs=2, default=(9, 11))
    parser.add_argument("--min-N", type=int, default=2)
    parser.add_argument("--max-dist", type=float, default=75.0)
    parser.add_argument("--partial-ens-size", "-p", action="store_true")
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--method", "-m", type=str, default="permutation")
    parser.add_argument("--n-boot", "-b", type=int, default=9_999)
    parser.add_argument("--n-resamples", "-n", type=int, default=999_999)
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

    if args.subset:
        mapping = pd.read_csv("ogando_data/check/holo_id_mapping.csv")
        if args.filename.stem == "new_late_mora_mean_2d":
            rename = {"old_holo_id": "holo_id"}
        else:
            rename = {"best_guess_new_holo_id": "holo_id"}
            assert (df["holo_id"].str[-2:] == ".0").all()
            df["holo_id"] = df["holo_id"].str[:-2]
        mapping = mapping.rename(columns=rename).dropna(subset=["holo_id"])
        mapping = mapping[["holo_id", "confidence"]]
        print(len(df))
        df["holo_id"] = df["holo_id"].str.replace("_+", "_", regex=True)
        df = df.merge(mapping, how="left", on="holo_id", validate="m:1")
        df = df.query("confidence > 0.9").copy()
        print(len(df))

    if args.use_old_N or args.use_old_distance:
        df_old = pd.read_pickle("ogando_data/new_late_mora_mean_2d.pkl")
        mapping = pd.read_csv("ogando_data/check/holo_id_mapping.csv")
        mapping = mapping.rename(columns={"old_holo_id": "holo_id"})
        mapping = mapping[["holo_id", "best_guess_new_holo_id", "confidence"]]
        df_old["holo_id"] = df_old["holo_id"].str.replace("_+", "_", regex=True)
        df_old = df_old.merge(mapping, how="left", on="holo_id", validate="m:1")
        df_old = df_old.query("confidence > 0.9").copy()
        df_old = df_old.drop(columns=["holo_id"])
        df_old = df_old.rename(columns={"best_guess_new_holo_id": "holo_id"})
        df_old = df_old[["holo_id", "_N"]].drop_duplicates(subset=["holo_id"])
        df = df.merge(df_old, how="left", on="holo_id", validate="m:1")
        df["_N"] = df["_N_y"]

    df = df.query(f"_distance < {args.max_dist}").copy()
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
            "dr": ("dr", args.estimator),
            kind: ("dr", kind),
            "count": ("dr", "count"),
        }
    )
    df = df.query(f"~dr.isna() and count >= {args.min_N}").copy()
    df = df.pivot(
        index=["exp_type", "holo_id", "_holo_osi", "_N"],
        columns="cell_type",
        values=["dr", kind, "count"],
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

    # new_holo_ids = pd.read_pickle("ogando_data/new_holo_ids.pkl")
    # print(df.query("holo_id.isin(@new_holo_ids)"))
    # return

    mapping = {
        "std_PV": "PV response s.d.",
        "std_SST": "SST response s.d.",
        "var_PV": "PV variance",
        "var_SST": "SST variance",
        "dr_PYR": "Pyr response",
        "dr_PV": "PV response",
        "dr_SST": "SST response",
        "std_PV_res": "PV response s.d. res.",
        "std_SST_res": "SST response s.d. res.",
        "var_PV_res": r"PV variance res.",
        "var_SST_res": r"SST variance res.",
        "dr_PYR_res": "Pyr response res.",
        "dr_PV_res": "PV response res.",
        "dr_SST_res": "SST response res.",
    }
    if args.partial_ens_size:
        mapping = {k: f"{v} res." for k, v in mapping.items() if not k.endswith("res")}

    kwargs = {
        "n_boot": args.n_boot,
        "seed": 0,
        "statannot": True,
        "statannot_kws": {
            "method": args.method,
            "n_resamples": args.n_resamples,
            "rng": 0,
            "format_spec": ".4g",
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

    # E mean vs I var/std plots
    for cell_type in ["PV", "SST"]:
        logger.info(f"Plotting dr_PYR vs {kind}_{cell_type}...")
        viz.figplot(
            df.query(f"~{kind}_{cell_type}.isna() and ~dr_PYR.isna()").copy(),
            func="lmplot",
            x=f"{kind}_{cell_type}",
            y="dr_PYR",
            logx=args.logx,
            xscale="log" if args.logx else "linear",
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

    # E mean vs I var/std plots with I mean partialled out
    for cell_type in ["PV", "SST"]:
        x, y, z = f"{kind}_{cell_type}", "dr_PYR", f"dr_{cell_type}"
        logger.info(f"Plotting {y} vs {x} with {z} partialled out...")
        sf = df.query(f"~{x}.isna() and ~{y}.isna() and ~{z}.isna()").copy()
        if args.logx:
            sf[x] = np.log(sf[x])
        sf[f"{x}_res"] = residual(sf[z], sf[x])
        sf[f"{y}_res"] = residual(sf[z], sf[y])
        if args.logx:
            sf[f"{x}_res"] = np.exp(sf[f"{x}_res"])
        viz.figplot(
            sf,
            func="lmplot",
            x=f"{x}_res",
            y=f"{y}_res",
            logx=args.logx,
            xscale="log" if args.logx else "linear",
            **kwargs,
        )
        if args.out:
            plt.savefig(
                args.out / f"{cell_type}_partial_mean.pdf",
                metadata={"Subject": " ".join(["python"] + sys.argv)},
            )
        if args.show:
            plt.show()
        else:
            plt.close()

    # I mean vs I var/std plots
    for cell_type in ["PV", "SST"]:
        logger.info(f"Plotting dr_{cell_type} vs {kind}_{cell_type}...")
        viz.figplot(
            df.query(f"~{kind}_{cell_type}.isna() and ~dr_{cell_type}.isna()").copy(),
            func="lmplot",
            x=f"{kind}_{cell_type}",
            y=f"dr_{cell_type}",
            logx=args.logx,
            xscale="log" if args.logx else "linear",
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

    # E mean vs I mean plots
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

    # E mean vs I mean plots with I var/std partialled out
    for cell_type in ["PV", "SST"]:
        x, y, z = f"dr_{cell_type}", "dr_PYR", f"{kind}_{cell_type}"
        if args.logx:
            df[f"log({z})"] = np.log10(df[z])
            z = f"log({z})"
        logger.info(f"Plotting {y} vs {x} with {z} partialled out...")
        sf = df.query(f"~{x}.isna() and ~{y}.isna() and ~`{z}`.isna()").copy()
        sf[f"{x}_res"] = residual(sf[z], sf[x])
        sf[f"{y}_res"] = residual(sf[z], sf[y])
        viz.figplot(
            sf,
            func="lmplot",
            x=f"{x}_res",
            y=f"{y}_res",
            **kwargs,
        )
        if args.out:
            plt.savefig(
                args.out / f"{cell_type}_mean_partial_{kind}.pdf",
                metadata={"Subject": " ".join(["python"] + sys.argv)},
            )
        if args.show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()
