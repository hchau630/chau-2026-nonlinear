import logging
import pprint
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from niarb import exceptions, nn
from niarb.dataset import Dataset, collate_fn

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_parser_arguments(parser):
    parser.add_argument("--out", "-o", type=Path, help="output filename")

    return parser


def run_from_conf_args(conf, args):
    conf = conf.copy()
    for k in ["out", "progress"]:
        if v := getattr(args, k):
            conf[k] = v

    logger.debug(f"config:\n{pprint.pformat(conf)}")
    run(**conf)


def run(
    dataset: Dataset | dict,
    pipeline1: nn.Pipeline | dict,
    pipeline2: nn.Pipeline | dict,
    state_dict: dict[str, Tensor] | str | Path,
    dtype: torch.dtype | str | None = None,
    batch_size: int | None = None,
    out: Path | str | None = None,
    progress: bool = False,
) -> DataFrame | Tensor:
    # handle alternative input types
    if isinstance(state_dict, (str, Path)):
        state_dict = torch.load(state_dict, map_location="cpu", weights_only=True)

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    # initialize dataset
    if isinstance(dataset, dict):
        dataset = Dataset(**dataset)

    if batch_size is None:
        batch_size = len(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # initialize pipelines
    if isinstance(pipeline1, dict):
        pipeline1 = nn.Pipeline(**pipeline1)
    if isinstance(pipeline2, dict):
        pipeline2 = nn.Pipeline(**pipeline2)
    logger.debug(f"pipeline1:\n{pipeline1}")
    logger.debug(f"pipeline2:\n{pipeline2}")

    # load state dict
    nn.load_state_dict(pipeline1, state_dict, strict=False)
    nn.load_state_dict(pipeline2, state_dict, strict=False)
    logger.debug(f"state dict 1:\n{pprint.pformat(pipeline1.state_dict())}")
    logger.debug(f"state dict 2:\n{pprint.pformat(pipeline2.state_dict())}")
    pipeline1.to(device, dtype=dtype)
    pipeline2.to(device, dtype=dtype)

    # run model
    logger.info("Running model...")
    df = []
    for i, (x, kwargs) in tqdm(
        enumerate(dataloader), desc="batch", disable=not progress
    ):
        dV = x.data["dV"].to(device)
        x = x.to(device, dtype=dtype)
        x["dV"] = dV
        logger.debug(f"x:\n{x}")

        row = {"batch": i}
        try:
            with torch.inference_mode():
                y1 = pipeline1(x, **kwargs)
        except exceptions.SimulationError:
            row["is_stable_model1"] = False
        else:
            row["is_stable_model1"] = True

        try:
            with torch.inference_mode():
                y2 = pipeline2(x, **kwargs)
        except exceptions.SimulationError:
            row["is_stable_model2"] = False
        else:
            row["is_stable_model2"] = True

        if row["is_stable_model1"] and row["is_stable_model2"]:
            err = y2["dr"] - y1["dr"]
            if isinstance(err, Tensor):
                row["err_norm"] = err.norm().item()
                row["rel_err_norm"] = (err.norm() / y1["dr"].norm()).item()
            else:
                row["err_norm"] = np.linalg.norm(err)
                row["rel_err_norm"] = np.linalg.norm(err) / np.linalg.norm(y1["dr"])

        logger.info(str(row))
        df.append(row)
    df = pd.DataFrame(df)

    # save output
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(out)

    return df
