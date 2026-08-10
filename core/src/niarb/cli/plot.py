import importlib
import logging
import pprint
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pandas import DataFrame
from seaborn import FacetGrid
from tqdm import tqdm

from niarb import utils, viz

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_parser_arguments(parser):
    parser.add_argument("--out", "-o", type=Path, help="path to output directory")
    parser.add_argument("--show", action="store_true", help="display plots")

    return parser


def run_from_conf_args(conf, args):
    conf = conf.copy()
    for k in ["out", "show", "progress"]:
        if v := getattr(args, k):
            conf[k] = v

    logger.debug(f"config:\n{pprint.pformat(conf)}")
    run(**conf)


def run(
    dataframes: Iterable[DataFrame | dict[str]],
    plots: Iterable[dict[str]],
    tag_names: Sequence[str] = (),
    out: Path | str | None = None,
    file_type: str = "pdf",
    dpi: float | Literal["figure"] = "figure",
    progress: bool = False,
    show: bool = False,
) -> dict[str, FacetGrid]:
    logger.info("Generating dataframe...")
    dfs = []
    for i, df in tqdm(list(enumerate(dataframes)), desc="df", disable=not progress):
        if isinstance(df, dict):
            tags = {k: df.pop(k) for k in tag_names}
            df = dataframe(**df, tags=tags)
        logger.debug(f"df{i}:\n{df}")
        dfs.append(df)

    df = pd.concat(dict(enumerate(dfs))).reset_index(0, names="dataframe_idx")
    logger.debug(f"df:\n{df}")

    logger.info("Plotting results...")
    figs = {}
    for p in tqdm(plots, desc="plot", disable=not progress):
        logger.debug(f"plot config: {p}")
        for k, v in plot(df, **p, progress=progress, leave=False).items():
            figs[k] = v

    if out:
        out = Path(out)
        for k, v in figs.items():
            path = out / f"{k}.{file_type}"
            path.parent.mkdir(exist_ok=True, parents=True)
            v.savefig(path, dpi=dpi, metadata={"Subject": " ".join(sys.argv)})
    if show:
        plt.show()

    return figs


def dataframe(
    func: Callable[..., DataFrame] | Sequence[str | Sequence[str]],
    tags: dict[str, str] | None = None,
    min_instances: int = 0,
    query: str | None = None,
    cuts: dict[str, int | Sequence[int | float]] | None = None,
    rolling: dict[str, tuple[Sequence[int | float], int | float]] | None = None,
    reset_index: tuple[int | str | Sequence[int | str], str | Sequence[str]]
    | None = None,
    columns: dict[str, str | tuple[str, dict]] | None = None,
    pivot_table: dict[str, str | Sequence[str]] | None = None,
    **kwargs,
) -> DataFrame:
    if isinstance(func, Sequence):
        if len(func) == 2 and isinstance(func[0], str) and isinstance(func[1], str):
            func = getattr(importlib.import_module(func[0]), func[1])
        elif all(isinstance(f, Sequence) and len(f) == 2 for f in func):
            func = utils.compose(
                *[getattr(importlib.import_module(f0), f1) for f0, f1 in func]
            )
        else:
            raise ValueError(f"Invalid func: {func}.")

    logger.debug(f"func kwargs:\n{pprint.pformat(kwargs)}")
    df = func(**kwargs)

    if min_instances > 0 and df["instance_idx"].nunique() < min_instances:
        logger.info(f"Skipping: {df['instance_idx'].nunique()=} < {min_instances=}.")
        return pd.DataFrame()

    if tags:
        for k, v in tags.items():
            df[k] = v

    df = process_dataframe(
        df,
        query=query,
        cuts=cuts,
        rolling=rolling,
        reset_index=reset_index,
        columns=columns,
        pivot_table=pivot_table,
    )

    return df


def plot(
    df: DataFrame,
    name: str = "figure",
    query: str | None = None,
    cuts: dict[str, int | Sequence[int | float]] | None = None,
    rolling: dict[str, tuple[Sequence[int | float], int | float]] | None = None,
    reset_index: tuple[int | str | Sequence[int | str], str | Sequence[str]]
    | None = None,
    columns: dict[str, str | tuple[str, dict]] | None = None,
    pivot_table: dict[str, str | Sequence[str]] | None = None,
    groupby: str | Sequence[str] | None = None,
    progress: bool = False,
    leave: bool = True,
    **kwargs,
) -> dict[str, FacetGrid]:
    logger.info(f"Plotting {name}...")
    df = process_dataframe(
        df,
        query=query,
        cuts=cuts,
        rolling=rolling,
        reset_index=reset_index,
        columns=columns,
        pivot_table=pivot_table,
    )

    figs = {}
    if groupby:
        g = df.groupby(groupby, observed=True)
        for group, sf in tqdm(g, desc="subplot", disable=not progress, leave=leave):
            if isinstance(groupby, str):
                group_name = f"{groupby}={group}"
            else:
                group_name = "-".join(f"{k}={v}" for k, v in zip(groupby, group))
            logger.info(f"Plotting {group_name}...")
            figs[f"{name}-{group_name}"] = viz.figplot(sf, **kwargs)
    elif len(df) > 0:
        figs[name] = viz.figplot(df, **kwargs)

    return figs


def process_dataframe(
    df: DataFrame,
    query: str | None = None,
    cuts: dict[str, int | Sequence[int | float]] | None = None,
    rolling: dict[str, tuple[Sequence[int | float], int | float]] | None = None,
    reset_index: tuple[int | str | Sequence[int | str], str | Sequence[str]]
    | None = None,
    columns: dict[str, str | tuple[str, dict]] | None = None,
    pivot_table: dict[str, str | Sequence[str]] | None = None,
) -> DataFrame:
    if (
        len(df) == 0
        or (query is None and cuts is None and rolling is None)
        and (reset_index is None and columns is None)
    ):
        return df

    df = df.copy()

    if reset_index:
        # need to split into two steps because reset_index is dumb
        df.index = df.index.set_names(reset_index[1], level=reset_index[0])
        df = df.reset_index(level=reset_index[0])

    if query:
        df = df.query(query).copy()
        if len(df) == 0:
            return df

    if pivot_table:
        df = df.pivot_table(**pivot_table)
        # flatten MultiIndex columns in case multiple aggfuncs are passed to pivot_table
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c) for c in df.columns.to_flat_index()]
        df = df.reset_index()

    if columns:
        for k, v in columns.items():
            df[k] = df.eval(v) if isinstance(v, str) else df.eval(v[0], local_dict=v[1])

    if rolling and cuts and set(rolling) & set(cuts):
        raise ValueError("rolling and cuts cannot share keys.")

    if cuts:
        for k, v in cuts.items():
            if k in df.columns:
                df[k] = pd.cut(df[k], bins=v, right=False)

    if rolling:
        for k, (centers, window) in rolling.items():
            if k in df.columns:
                df = utils.rolling(df, k, centers, window)

    return df
