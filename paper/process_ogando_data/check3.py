from collections.abc import Sequence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def normalize(
    df: pd.DataFrame,
    primary_keys: Sequence[str],
    dependent_keys: Sequence[str] = (),
    suffixes: Sequence[str] = ("new", "old"),
    other_keys: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dependent_keys = [f"{k}_{suffix}" for k in dependent_keys for suffix in suffixes]
    dependent_keys = dependent_keys + other_keys

    # check that primary_keys uniquely determines foreign_keys and other_keys
    g = df.groupby(primary_keys, observed=True)
    for col in dependent_keys:
        # set dropna=False to ensure there it has 1 unique value including NaNs
        if not (g[col].nunique(dropna=False) == 1).all():
            raise ValueError(f"{primary_keys} does not uniquely determine {col}")

    return g[[*primary_keys, *dependent_keys]].nth[0]


def main():
    df_new = pd.read_pickle("ogando_data/preprocessed_windowed.pkl")
    df_old = pd.read_pickle("ogando_data/new_late_mora_mean_2d.pkl")
    assert isinstance(df_new, pd.DataFrame)
    assert isinstance(df_old, pd.DataFrame)

    df_old = df_old.groupby(
        [
            "mouse_id",
            "session_id",
            "cell_id",
            "holo_id",
            "cell_type",
            "_holo_osi",
            "_N",
            "_distance",
        ],
        as_index=False,
        observed=True,
    ).agg(dr=("dr", "median"))  # average across time for each cell and holo

    print(df_new["holo_id"].nunique())
    print(df_old["holo_id"].nunique())

    # Standardize names
    assert (df_new["holo_id"].str[-2:] == ".0").all()
    df_new["holo_id"] = df_new["holo_id"].str[:-2]
    for df in [df_new, df_old]:
        df["date"] = pd.to_datetime(df["session_id"].str[-8:], format="%Y%m%d")
        df["cell_id"] = df["cell_id"].str.replace("_+", "_", regex=True)
        df["holo_id"] = df["holo_id"].str.replace("_+", "_", regex=True)
        df["session_id"] = df["session_id"].str.replace("_+", "_", regex=True)

    # with open("ogando_data/check/df_new_holo_ids.txt", "w") as f:
    #     f.writelines(f"{s}\n" for s in df_new["holo_id"].unique())
    # with open("ogando_data/check/df_old_holo_ids.txt", "w") as f:
    #     f.writelines(f"{s}\n" for s in df_old["holo_id"].unique())

    mapping = pd.read_csv("ogando_data/check/holo_id_mapping.csv")
    mapping = mapping[["old_holo_id", "best_guess_new_holo_id", "confidence"]]
    mapping = mapping.rename(columns={"old_holo_id": "holo_id"})
    df_old = df_old.merge(mapping, how="left", on="holo_id", validate="m:1")
    df_old = df_old.query("confidence > 0.9")
    df_old = df_old.rename(
        columns={"holo_id": "original_holo_id", "best_guess_new_holo_id": "holo_id"}
    )

    not_in_old = df_new["holo_id"][~df_new["holo_id"].isin(df_old["holo_id"])].unique()
    not_in_new = df_old["holo_id"][~df_old["holo_id"].isin(df_new["holo_id"])].unique()
    print(len(not_in_old), len(not_in_new))

    df = df_new.merge(
        df_old,
        on=["holo_id", "cell_id", "mouse_id", "session_id"],
        suffixes=("_new", "_old"),
        validate="1:1",
    )

    # df["delta_distance"] = (df["_distance_new"] - df["_distance_old"]).abs()
    # print(df["delta_distance"].describe())

    # for col in ["dr", "_distance"]:
    #     sns.scatterplot(df, x=f"{col}_old", y=f"{col}_new")
    #     # plt.show()
    #     plt.tight_layout()
    #     plt.savefig(f"ogando_data/check/figures/{col}.png", dpi=300)
    #     plt.close()

    # cols = ["cell_type"]
    # cell_df = normalize(df, primary_keys=["cell_id"], dependent_keys=cols)
    # for col in cols:
    #     sns.scatterplot(cell_df, x=f"{col}_old", y=f"{col}_new")
    #     # plt.show()
    #     plt.tight_layout()
    #     plt.savefig(f"ogando_data/check/figures/{col}.png", dpi=300)
    #     plt.close()

    cols = ["_holo_osi", "_N"]
    other_keys = ["original_holo_id"]
    holo_df = normalize(
        df, primary_keys=["holo_id"], dependent_keys=cols, other_keys=other_keys
    )
    print(
        holo_df.query("_N_new >= 9 and _N_new <= 11 and _N_old > 20")[
            ["original_holo_id", "holo_id", "_N_new", "_N_old"]
        ]
    )
    # with pd.option_context("display.max_rows", None):
    #     print(holo_df.query("_N_new >= 9 and _N_new <= 11")[["_N_new", "_N_old"]])
    #     print(holo_df.query("_N_old >= 9 and _N_old <= 11")[["_N_new", "_N_old"]])

    for col in cols:
        sns.scatterplot(holo_df, x=f"{col}_old", y=f"{col}_new")
        # plt.show()
        plt.tight_layout()
        plt.savefig(f"ogando_data/check/figures/{col}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
