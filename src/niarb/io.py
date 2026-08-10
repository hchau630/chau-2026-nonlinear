import importlib
import json
import logging
import os
import pprint
import tomllib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, overload

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor

from niarb import parsing, utils
from niarb.cell_type import CellType

logger = logging.getLogger(__name__)

_load_mappings = {
    ".pkl": pd.read_pickle,
    ".json": pd.read_json,
    ".h5": pd.read_hdf,
    ".hdf5": pd.read_hdf,
    ".feather": pd.read_feather,
    ".parquet": pd.read_parquet,
}

_save_mappings = {
    ".pkl": "to_pickle",
    ".json": "to_json",
    ".h5": "to_hdf",
    ".hdf5": "to_hdf",
    ".feather": "to_feather",
    ".parquet": "to_parquet",
}


def load_dataframe(filename: str | Path, **kwargs) -> DataFrame:
    logger.debug(f"Loading {filename}.")
    return _load_mappings[Path(filename).suffix](filename, **kwargs)


def save_dataframe(df: DataFrame, filename: str | Path, **kwargs):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pd.DataFrame, but got {type(df)=}.")

    getattr(df, _save_mappings[Path(filename).suffix])(filename, **kwargs)


def ref_hook(d):
    if "__ref__" in d:
        # reference an object from a different configuration file
        filename, *keys = d.pop("__ref__")
        out = load_config(filename, parse_objects=False)
        for key in keys:
            out = out[key]
        if isinstance(out, dict):
            out = out | d
        elif len(d) > 0:
            raise ValueError(f"Unknown keys: {d.keys()}")

    else:
        out = d

    return out


def object_hook(d):
    if "__matrix__" in d:
        # create a cartesian product of values with some special handling for
        # additional inclusions
        out = parsing.matrix(
            d.pop("__matrix__"),
            include=d.pop("__include__", []),
            ignore=d.pop("__ignore__", []),
        )

    elif "__call__" in d:
        # dynamically evaluate a callable object in any module with arguments
        module, callable_, *args = d.pop("__call__")
        module = importlib.import_module(module)
        out = getattr(module, callable_)(*args, **d)
        d = {}

    elif "__get__" in d:
        # import module and get an object from it
        module, obj = d.pop("__get__")
        module = importlib.import_module(module)
        out = getattr(module, obj)

    elif "__array__" in d:
        # create a numpy array from a string representation
        out = parsing.array(d.pop("__array__"))

    elif "__indices__" in d:
        # create a list of non-negative integers from a string representation
        out = parsing.indices(d.pop("__indices__"))

    elif "__environ__" in d:
        # substitute value with environment variable
        out = os.environ[d.pop("__environ__")]

    elif "__const__" in d:
        # built-in Python constants that cannot be specified within JSON or TOML
        const = d.pop("__const__")
        if const == "None":
            out = None
        elif const == "True":
            out = True
        elif const == "False":
            out = False
        elif const in {"Ellipsis", "..."}:
            out = Ellipsis
        elif const == "NotImplemented":
            out = NotImplemented
        else:
            raise ValueError(f"Unknown constant: {const}")

    elif "__format__" in d:
        # format a string with values from the dictionary
        fmt = d.pop("__format__")
        d = dict(os.environ) | d  # include environment variables for convenience
        if isinstance(fmt, str):
            out = fmt.format(**d)
        elif isinstance(fmt, Sequence):
            fmt, *args = fmt
            out = fmt.format(*args, **d)
        else:
            raise TypeError(
                f"__format__ must be a string or a sequence, but {type(fmt)=}."
            )
        d = {}

    else:
        out = d
        d = {}

    if len(d) > 0:
        raise ValueError(f"Unknown keys: {d.keys()}")

    return out


def load_config(
    filename: str | Path,
    format: str | None = None,
    parse_refs: bool = True,
    parse_objects: bool = True,
):
    """Load a TOML or JSON file with support for references and object creation.

    Note that references are resolved before Python objects are created, in the same spirit
    as YAML processing order (https://yaml.org/spec/1.2.2/#31-processes), where anchors/aliases
    are resolved before tags are processed. Unlike YAML, however, references are external
    rather than internal, i.e. they refer to objects in other configuration files.

    Args:
        filename: Path to the configuration file.
        format (optional): File format. If None, the format is inferred from the filename.
        parse_refs (optional): Whether to parse references.
        parse_objects (optional): Whether to parse objects.

    Returns:
        Parsed configurations.

    """
    filename = Path(filename)
    if format is None:
        format = filename.suffix.strip(".")

    if format == "json":
        with open(filename, "r") as f:
            out = json.load(f, object_hook=ref_hook if parse_refs else None)
    elif format == "toml":
        with open(filename, "rb") as f:
            out = tomllib.load(f)
        out = json.loads(json.dumps(out), object_hook=ref_hook if parse_refs else None)
    else:
        raise ValueError(f"Unsupported file format: {format}.")

    logger.debug(f"config (first pass):\n{pprint.pformat(out)}")

    if parse_objects:
        out = utils.tree_map(out, object_hook, exclude={"__include__"})
        logger.debug(f"config (second pass):\n{pprint.pformat(out)}")
        out = utils.tree_map(out, object_hook)
        logger.debug(f"config (third pass):\n{pprint.pformat(out)}")

    return out


@overload
def load_data(
    paths: Iterable[str | Path], query: str = ..., y: str = ..., yerr: str = ...
) -> list[DataFrame]: ...


@overload
def load_data(
    paths: str | Path, query: str = ..., y: str = ..., yerr: str = ...
) -> DataFrame: ...


def load_data(
    paths: str | Path | Iterable[str | Path],
    query: str = "",
    uncuts: str | Sequence[str] | None = None,
    y: str = "dr",
    yerr: str = "dr_se",
) -> DataFrame | list[DataFrame]:
    return_list = True
    if isinstance(paths, (str, Path)):
        paths = [paths]
        return_list = False

    if isinstance(uncuts, str):
        uncuts = [uncuts]

    data = [load_dataframe(path) for path in paths]

    # remove NaN target data and standardize column names
    for i, df in enumerate(data):
        cols = [y]
        if yerr in df.columns:
            cols.append(yerr)
        df = df[~df[cols].isna().any(axis=1)]
        df = df.rename(columns=dict(zip(cols, ["dr", "dr_se"])))

        if query:
            df = df.query(query)

        if uncuts:
            for k in uncuts:
                if k in df.columns:
                    df[k] = pd.IntervalIndex(df[k]).mid

        data[i] = df

    if return_list:
        return data
    return data[0]


@overload
def iterdir(
    path: str | Path,
    pattern: str = ...,
    indices: int = ...,
    sort_key: Callable[[Path], Any] | None = ...,
    sort_reverse: bool = ...,
    stem: bool = ...,
) -> str | Path: ...


@overload
def iterdir(
    path: str | Path,
    pattern: str = ...,
    indices: Iterable[int] | None = ...,
    sort_key: Callable[[Path], Any] | None = ...,
    sort_reverse: bool = ...,
    stem: bool = ...,
) -> list[str] | list[Path]: ...


def iterdir(
    path: str | Path,
    pattern: str = "*",
    indices: int | Iterable[int] | None = None,
    sort_key: Callable[[Path], str] | None = None,
    sort_reverse: bool = False,
    stem: bool = False,
) -> str | Path | list[str] | list[Path]:
    """Iterate over files in a directory.

    Args:
        path: Path to directory.
        pattern (optional): File pattern.
        indices (optional): Index/indices of files to load. If None, all files are loaded.
        sort_key (optional): Key function to sort filenames. If None, filenames are sorted
          as floats if they can be converted, otherwise as strings.
        sort_reverse (optional): Whether to sort in reverse order.
        stem (optional): Whether to filename stems only.

    Returns:
        Filename or list of filenames.

    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory.")

    if sort_key is None:

        def sort_key(f: Path) -> str:
            filename = f.stem
            try:
                return str(float(filename))
            except ValueError:
                return filename

    filenames = sorted(path.glob(pattern), key=sort_key, reverse=sort_reverse)

    if stem:
        filenames = [f.stem for f in filenames]

    if isinstance(indices, int):
        return filenames[indices]

    if indices:
        return [filenames[i] for i in indices]

    return filenames


def load_state_dict(
    path: str | Path,
    cell_types: Sequence[str | CellType] = tuple(CellType),
    new_cell_types: Sequence[str | CellType] = tuple(CellType),
    fill: dict[str, float | Sequence[str]] | None = None,
    expand_vf: int = -1,
    modify: dict[str, Sequence[tuple[Sequence[str], str | float]]] | None = None,
    **kwargs,
) -> dict[str, Tensor]:
    if modify is None:
        modify = {}
    cell_types = tuple(ct.name for ct in cell_types)
    new_cell_types = tuple(ct.name for ct in new_cell_types)

    state_dict = torch.load(path, **kwargs)
    logger.debug(f"Loaded state dict:\n{pprint.pformat(state_dict)}")

    n = len(new_cell_types)
    if fill:
        for k, v in state_dict.items():
            if v.ndim == 0:
                continue
            fill_val = fill[k]
            if isinstance(fill_val, Sequence):
                fill_val = v[tuple(cell_types.index(i) for i in fill_val)]

            new_v = torch.full((n,) * v.ndim, fill_val, dtype=v.dtype, device=v.device)
            for idx in np.ndindex(v.shape):
                new_idx = tuple(new_cell_types.index(cell_types[i]) for i in idx)
                new_v[new_idx] = v[idx]
            state_dict[k] = new_v

    if expand_vf != -1:
        vf = state_dict["vf"]
        if vf.ndim != 0:
            raise ValueError("vf must be zero-dimensional if expand_vf is provided.")

        state_dict["vf"] = torch.full((expand_vf,), vf.item(), device=vf.device)

    for name, v in modify.items():
        for idx, value in v:
            idx = tuple(new_cell_types.index(i) for i in idx)
            state_dict[name][idx] = (
                eval(value, {}, state_dict) if isinstance(value, str) else value
            )

    logger.debug(f"Modified state dict:\n{pprint.pformat(state_dict)}")
    return state_dict
