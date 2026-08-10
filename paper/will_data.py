import pandas as pd


def main():
    df = pd.read_feather("will_data/sst_connectivity_mdf.feather")
    df["sst"] = df["sst"].astype("bool")
    df["target"] = df["target"].astype("bool")
    sf = df[
        (df.sst == True)
        & (df.vis_id == 1)
        & (df.num_targets == 1)
        & (df.osi > 0.33)
        & (df.target == False)
    ]
    print(df.columns)
    print(df.dtypes)
    print(df["num_targets"].unique())
    print(df["trial"].unique())
    print(df["out_id"].unique())
    print(df["target"].unique())
    print(df["ori"].unique())
    print(df["cond"].unique())
    print(df["hz"].unique())
    print(df["spikes"].unique())
    print(
        df.query("vis_id == 1 and num_targets == 1 and ~sst")
        .groupby(["out_id", "trial"])["target"]
        .sum()
    )
    # print(df.query("out_id == 1 and trial == 2 and vis_id == 1").head(n=10).T)
    print(sf.head(n=10).T)
    print(sf.groupby(["hz", "spikes"]).size())
    print(sf[["x", "y", "z"]].describe())
    print(df.query("~sst")[["x", "y", "z"]].describe())


if __name__ == "__main__":
    main()
