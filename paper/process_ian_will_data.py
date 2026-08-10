import argparse
import pathlib
import sys

import torch
import pandas as pd
import numpy as np

from niarb import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_filename", type=str)
    parser.add_argument("--ori", action="store_true")
    parser.add_argument("--osi", action="store_true")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="ensemble",
        choices=["ensemble", "single_cell", "space_v_ori"],
    )
    parser.add_argument("--xy", action="store_true")
    parser.add_argument("--max-ori-dist", type=float, default=150.0)
    args = parser.parse_args()

    data_path = pathlib.Path(args.input_dir)

    if args.mode == "ensemble":
        data = process_ensemble_data(data_path, xy=args.xy)
    elif args.mode == "single_cell":
        data = process_single_cell_data(data_path, xy=args.xy)
    else:
        data = process_space_v_ori_data(data_path, max_ori_dist=args.max_ori_dist, osi=args.osi, xy=args.xy)

    if not args.ori and args.mode != "single_cell":
        data = data.query("holo_osi == 'low'").drop(columns="holo_osi")

    data.attrs["command"] = " ".join(["python"] + sys.argv)
    io.save_dataframe(data, args.output_filename)


def process_single_cell_data(data_path, xy=False):
    filename = "single_cell_xy.pt" if xy else "space_resp_1_cell.pt"
    data = torch.load(data_path / filename)
    data = pd.DataFrame(
        {
            "cell_type": "PYR",
            "N": 1,
            "distance": data["x"].squeeze().numpy(),
            "dr": data["y"].numpy(),
            "dr_se": data["yerr"].numpy(),
        }
    )

    data["cell_type"] = data["cell_type"].astype("category")
    distance_mids = np.sort(data["distance"].unique())
    freq = distance_mids[1] - distance_mids[0]
    low, high = distance_mids[0], distance_mids[-1]
    distance_intervals = pd.interval_range(low - freq / 2, high + freq / 2, freq=freq, closed="left")
    assert (distance_mids == distance_intervals.mid).all() 
    data["distance"] = pd.Categorical(
        data["distance"], categories=distance_mids
    ).rename_categories(distance_intervals)

    return data


def process_ensemble_data(data_path, xy=False):
    all_data = []
    filename = "cotuned_spreadout_xy.pt" if xy else "space_resp_10_cell_mean_geq200_cotuned.pt"
    data = torch.load(data_path / filename)
    data = pd.DataFrame(
        {
            "cell_type": "PYR",
            "N": 10,
            "holo_osi": "high",
            "density": "spreadout",
            "distance": data["x"].squeeze().numpy(),
            "dr": data["y"].numpy(),
            "dr_se": data["yerr"].numpy(),
        }
    )
    all_data.append(data)

    filename = "cotuned_compact_xy.pt" if xy else "space_resp_10_cell_mean_leq200_cotuned.pt"
    data = torch.load(data_path / filename)
    data = pd.DataFrame(
        {
            "cell_type": "PYR",
            "N": 10,
            "holo_osi": "high",
            "density": "compact",
            "distance": data["x"].squeeze().numpy(),
            "dr": data["y"].numpy(),
            "dr_se": data["yerr"].numpy(),
        }
    )
    all_data.append(data)

    filename = "untuned_spreadout_xy.pt" if xy else "space_resp_10_cell_mean_geq200_untuned.pt"
    data = torch.load(data_path / filename)
    data = pd.DataFrame(
        {
            "cell_type": "PYR",
            "N": 10,
            "holo_osi": "low",
            "density": "spreadout",
            "distance": data["x"].squeeze().numpy(),
            "dr": data["y"].numpy(),
            "dr_se": data["yerr"].numpy(),
        }
    )
    all_data.append(data)

    filename = "untuned_compact_xy.pt" if xy else "space_resp_10_cell_mean_leq200_untuned.pt"
    data = torch.load(data_path / filename)
    data = pd.DataFrame(
        {
            "cell_type": "PYR",
            "N": 10,
            "holo_osi": "low",
            "density": "compact",
            "distance": data["x"].squeeze().numpy(),
            "dr": data["y"].numpy(),
            "dr_se": data["yerr"].numpy(),
        }
    )
    all_data.append(data)

    data = pd.concat(all_data)
    data["density"] = data["density"].astype("category")
    data["cell_type"] = data["cell_type"].astype("category")
    distance_mids = np.sort(data["distance"].unique())
    freq = distance_mids[1] - distance_mids[0]
    low, high = distance_mids[0], distance_mids[-1]
    distance_intervals = pd.interval_range(low - freq / 2, high + freq / 2, freq=freq, closed="left")
    assert (distance_mids == distance_intervals.mid).all()
    data["distance"] = pd.Categorical(
        data["distance"], categories=distance_mids
    ).rename_categories(distance_intervals)

    return data


def process_space_v_ori_data(data_path, max_ori_dist=150, osi=False, xy=False):
    all_data = []
    for rel_ori in [0, 45, 90]:
        filename = f"cotuned_compact_xy_dori_{rel_ori}.pt" if xy else f"space_resp_10_cell_compact_cotuned_dori_{rel_ori}.pt"
        data = torch.load(data_path / filename)
        data = pd.DataFrame(
            {
                "cell_type": "PYR",
                "N": 10,
                "holo_osi": "high",
                "rel_ori": rel_ori,
                "density": "compact",
                "distance": data["x"].squeeze().numpy(),
                "dr": data["y"].numpy(),
                "dr_se": data["yerr"].numpy(),
            }
        )
        all_data.append(data)

        filename = f"cotuned_spreadout_xy_dori_{rel_ori}.pt" if xy else f"space_resp_10_cell_spreadout_cotuned_dori_{rel_ori}.pt"
        data = torch.load(data_path / filename)
        data = pd.DataFrame(
            {
                "cell_type": "PYR",
                "N": 10,
                "holo_osi": "high",
                "rel_ori": rel_ori,
                "density": "spreadout",
                "distance": data["x"].squeeze().numpy(),
                "dr": data["y"].numpy(),
                "dr_se": data["yerr"].numpy(),
            }
        )
        all_data.append(data)

    data = pd.concat(all_data)
    data = data.query(f"distance < {max_ori_dist}").copy()
    data["density"] = data["density"].astype("category")
    data["cell_type"] = data["cell_type"].astype("category")

    distance_mids = np.sort(data["distance"].unique())
    freq = distance_mids[1] - distance_mids[0]
    low, high = distance_mids[0], distance_mids[-1]
    distance_intervals = pd.interval_range(low - freq / 2, high + freq / 2, freq=freq, closed="left")
    assert (distance_mids == distance_intervals.mid).all()
    data["distance"] = pd.Categorical(
        data["distance"], categories=distance_mids
    ).rename_categories(distance_intervals)

    data["rel_ori"] = (
        data["rel_ori"]
        .astype("category")
        .cat.rename_categories(
            {
                0: pd.Interval(0.0, 22.5, closed="left"),
                45: pd.Interval(22.5, 67.5, closed="left"),
                90: pd.Interval(67.5, 90.0, closed="left"),
            }
        )
    )
    if osi:
        data["osi"] = pd.Interval(0.25, 1.0, closed="left")
        data["osi"] = data["osi"].astype("category")

    return data


if __name__ == "__main__":
    main()
