import argparse
from functools import reduce
from pathlib import Path

import pandas as pd


def validate(dfs, expected, primary_keys=None):
    df = reduce(pd.merge, dfs)

    assert set(expected.columns).issubset(df.columns)
    df = df[expected.columns]

    if primary_keys:
        # sort by primary keys for comparison up to row permutation
        assert all(k in df.columns for k in primary_keys)
        df = df.sort_values(by=primary_keys).reset_index(drop=True)
        expected = expected.sort_values(by=primary_keys).reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--normalized", default="normalized")
    parser.add_argument("--compressed", default="compressed.pkl")
    args = parser.parse_args()

    dfs = {}
    for path in (args.data_path / args.normalized).glob("*.pkl"):
        dfs[path.stem] = pd.read_pickle(path)
    expected = pd.read_pickle(args.data_path / args.compressed)

    validate(dfs.values(), expected, primary_keys=["holo_id", "cell_id"])
    print("Validation successful: normalized data matches compressed data.")


if __name__ == "__main__":
    main()
