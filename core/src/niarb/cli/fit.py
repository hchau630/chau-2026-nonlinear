import copy
import logging
import pprint
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from niarb import exceptions, nn, optimize, random
from niarb.dataset import Dataset, collate_fn

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_parser_arguments(parser):
    parser.add_argument(
        "-N", type=int, help="number of optimization trials (default: 10)"
    )
    parser.add_argument("--out", "-o", type=Path, help="path to output directory")
    parser.add_argument(
        "--ignore-errors", action="store_true", help="ignore errors during optimization"
    )

    return parser


def run_from_conf_args(conf, args):
    conf = conf.copy()
    for k in ["N", "out", "ignore_errors", "progress"]:
        if v := getattr(args, k):
            conf[k] = v

    logger.debug(f"config:\n{pprint.pformat(conf)}")
    run(**conf)


def run(
    *args,
    seed: int | None = None,
    loss_threshold: float = 0.75,
    N: int = 10,
    out: Path | str | None = None,
    progress: bool = False,
    **kwargs,
) -> tuple[list[float], list[dict[str, Tensor]]]:
    # initialize output directory
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=N, desc="fit", disable=not progress)
    N_successes, losses, state_dicts = 0, [], []
    while N_successes < N:
        torch.cuda.empty_cache()

        success, loss, state_dict = fit(
            *args, seed=seed, loss_threshold=loss_threshold, **kwargs
        )

        # save loss and state_dict if successful and loss is below threshold
        if success and loss < loss_threshold:
            if out:
                torch.save(state_dict, out / f"{loss:.10e}.pt")

            state_dicts.append(state_dict)
            losses.append(loss)

            pbar.update()
            N_successes += 1

        # increment random seed whether successful or not
        seed = seed + 1 if seed is not None else None

    return losses, state_dicts


def fit(
    data: Sequence[pd.DataFrame],
    dataset: Dataset | dict,
    pipeline: nn.Pipeline | dict,
    optimizer: dict | None = None,
    validation_dataset: Dataset | dict | None = None,
    validation_pipeline: nn.Pipeline | dict | None = None,
    init_state_dict: dict[str, Tensor] | str | Path | None = None,
    seed: int | None = None,
    loss_threshold: float = 0.75,
    weighted_loss: bool = False,
    equal_loss: bool = False,
    loss_scaling: Sequence[float] | None = None,
    normalized_loss: bool = True,
    dtype: str | torch.dtype | None = None,
    validation_dtype: str | torch.dtype | None = None,
    validation_batch_size: int | None = None,
    max_validation_fails: int = 0,
    ignore_errors: bool = False,
    debug: bool = False,
) -> tuple[bool, float, dict[str, Tensor]]:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if isinstance(validation_dtype, str):
        validation_dtype = getattr(torch, validation_dtype)

    validate = (
        (validation_dataset is not None)
        or (validation_pipeline is not None)
        or (dtype is not validation_dtype)
    )

    # initialize dataset
    if isinstance(dataset, dict):
        dataset = Dataset(**dataset, data=data)
    dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)

    # initialize validation dataset
    if isinstance(validation_dataset, dict):
        validation_dataset = Dataset(**validation_dataset, data=data)
    elif validation_dataset is None:
        validation_dataset = copy.deepcopy(dataset)
    if validation_batch_size is None:
        validation_batch_size = len(validation_dataset)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=validation_batch_size, collate_fn=collate_fn
    )

    # initialize fitting pipeline
    if isinstance(pipeline, dict):
        pipeline = nn.Pipeline(data=data, **pipeline)
    pipeline.to(device, dtype=dtype)
    logger.debug(str(pipeline))

    # initialize validation pipeline
    if isinstance(validation_pipeline, dict):
        validation_pipeline = nn.Pipeline(data=data, **validation_pipeline)
    elif validation_pipeline is None:
        validation_pipeline = copy.deepcopy(pipeline)
    validation_pipeline.to(device, dtype=validation_dtype)

    # initialize fitting criterion
    w = [torch.full((len(df),), 1.0) for df in data]

    if loss_scaling:
        w = [wi * scale for wi, scale in zip(w, loss_scaling, strict=True)]
        logger.debug(f"w:\n{w}")

    if equal_loss:
        # weight loss from each dataset roughly equally by scaling the loss of each
        # dataset inversely proportional to the number of data points in the dataset
        logger.debug(f"len(df): {[len(df) for df in data]}")
        w = [wi / len(df) for wi, df in zip(w, data, strict=True)]
        logger.debug(f"w:\n{w}")

    if weighted_loss:
        # loss is weighted inversely proportional to standard error square if available,
        # and the variance of the dataset as a proxy otherwise
        logger.debug(
            f"dr_se:\n{[df['dr_se'] if 'dr_se' in df else df['dr'].var() for df in data]}"
        )
        w = [
            wi
            / (
                torch.tensor(df["dr_se"].to_numpy()).float() ** 2
                if "dr_se" in df
                else df["dr"].var()
            )
            for wi, df in zip(w, data, strict=True)
        ]
        logger.debug(f"w:\n{w}")

    if loss_scaling or equal_loss or weighted_loss:
        w = torch.cat(w)
        w = w / w.mean()  # normalize to prevent w being too small/too large
        logger.debug(f"w:\n{w}")
    else:
        w = None

    criterion = nn.WeightedNormalizedLoss(w=w, normalized=normalized_loss)
    criterion.to(device, dtype=dtype)

    # initialize optimizer
    if optimizer is None:
        optimizer = {}

    optimizer = optimize.Optimizer(pipeline, criterion, **optimizer)

    # load initial state_dict
    if isinstance(init_state_dict, (str, Path)):
        init_state_dict = torch.load(init_state_dict, map_location="cpu")

    # sample x and y
    dataset.reset_targets()
    x, y, kwargs = next(iter(dataloader))
    x, y = x.to(device, dtype=dtype), y.to(device, dtype=dtype)
    logger.debug(str(x))

    # initialize pipeline parameters randomly
    logger.info(f"Initializing pipeline with {seed=}")
    with random.set_seed(seed):
        pipeline.apply(nn.reset_parameters)

    # if init_state_dict is provided, overwrite the initial pipeline parameters
    if init_state_dict:
        logger.info("Initializing pipeline from state_dict...")
        nn.load_state_dict(pipeline, init_state_dict, strict=False)
    logger.debug(pprint.pformat(nn.param_dict(pipeline)))

    pipeline.zero_grad()

    # optimize pipeline and handle exceptions
    if debug:
        success, loss = True, 0.0
    else:
        try:
            success, loss = optimizer(x, y, **kwargs)
            if normalized_loss:
                with torch.inference_mode():
                    y_pred = pipeline(x, **kwargs)
        except (exceptions.OptimizationError, exceptions.SimulationError) as err:
            logger.info(str(err))
            success, loss = False, np.inf
        except Exception:
            if not ignore_errors:
                raise
            logger.error(traceback.format_exc())
            success, loss = False, np.inf

    validation_pipeline.load_state_dict(pipeline.state_dict())
    if success and loss < loss_threshold:
        if normalized_loss and not debug:
            validation_pipeline.scale_parameters((y.norm() / y_pred.norm()).item())

        if validate:
            # optionally 'validate' fitted parameters on an altered (presumably more
            # accuarate but computationally expensive) pipeline and validation dataset
            logger.info("Validating model...")
            n_fails, y_pred = 0, []
            for i, (x, _, kwargs) in enumerate(validation_dataloader):
                logger.info(f"Running validation batch {i}...")
                x = x.to(device, dtype=validation_dtype)
                try:
                    with torch.inference_mode():
                        y_pred.append(validation_pipeline(x, **kwargs))
                except exceptions.SimulationError as err:
                    logger.info(str(err))
                    n_fails += 1
                except Exception:
                    if not ignore_errors:
                        raise
                    logger.error(traceback.format_exc())
                    n_fails += 1

                if n_fails > max_validation_fails:
                    success, loss = False, np.inf
                    logger.info(
                        "Number of failed validation batches exceeded "
                        f"{max_validation_fails=}. Validation loss: {loss}."
                    )
                    break
            else:
                y_pred = torch.stack(y_pred).mean(dim=0)
                loss = criterion(y_pred, y).item()
                logger.info(
                    f"Number of failed validation batches: {n_fails}. "
                    f"Validation loss: {loss}"
                )

    # Note: deepcopy is necessary since module.state_dict() only returns a shallow copy
    return success, loss, copy.deepcopy(nn.state_dict(validation_pipeline))
