import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    # df1 = pd.read_pickle("ogando_data/old/preprocessed_normalized.pkl")
    # df2 = pd.read_pickle("ogando_data/preprocessed_alt.pkl")
    # print(df1.shape, df2.shape)
    # columns = ["cell_id", "cell_type", "_N", "_distance", "dr"]
    # holo_id = "MBOT91_857__20230914_ensembleSizes_12.0"
    # # holo_id = "MBOT91_857__20230915_ensembleSizes_17.0"
    # # holo_id = "MBOT91_857__20230928_ensembleSizes_9.0"
    # query = f"holo_id == '{holo_id}' and cell_type == 'SST'"
    # print(df1.query(query)["dr"].var())
    # print(df2.query(query)["dr"].var())
    # # print(df1.query(query)[columns])
    # # print(df2.query(query)[columns])
    # query = f"holo_id == '{holo_id}' and cell_type == 'PYR'"
    # print(df1.query(query)["dr"].median())
    # print(df2.query(query)["dr"].median())
    # df = pd.concat({"old": df1, "new": df2}, names=["dataset"]).reset_index(0)
    # sns.histplot(df.query(query), x="dr", hue="dataset")
    # plt.show()
    # df = df1.merge(
    #     df2,
    #     on=["cell_id", "holo_id", "mouse_id", "session_id", "cell_type", "_N"],
    #     suffixes=("_old", "_new"),
    #     validate="1:1",
    # )
    # print(df)
    # ax = sns.regplot(
    #     df.query(query),
    #     x="dr_old",
    #     y="dr_new",
    #     scatter_kws={
    #         "alpha": 1,  # regplot defaults to 0.8; scatterplot is opaque
    #         "edgecolor": "w",  # scatterplot's white marker edge
    #         "linewidths": 0.48,  # scatterplot's edge width for the default point size
    #         "s": 36,  # default scatter point size (markersize 6 squared)
    #     },
    #     line_kws={"color": "black"},
    # )
    # x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    # ax.plot(x, x, color="black", linestyle="--")
    # ax.grid(which="both")
    # plt.show()
    # plt.show()
    # # print(df1.query(query)[columns])
    # # print(df2.query(query)[columns])
    # df3 = pd.read_csv("ogando_data/old/PC_stim_20260510.csv")
    # df4 = pd.read_csv("ogando_data/PC_stim_20260622.csv")
    # print(
    #     df3.query(
    #         "holoID_s == 'MBOT91_857__20230914_ensembleSizes_12.0' and Cell_uID == 'MBOT91_857_20230914__2191'"
    #     )[["activity", "activity_BL", "activity_STD"]]
    # )
    # print(
    #     df4.query(
    #         "holoID_s == 'MBOT91_857__20230914_ensembleSizes_12.0' and Cell_uID == 'MBOT91_857_20230914__2191'"
    #     )[["activity", "activity_BL", "activity_STD", "activity_std_per_trial"]]
    # )

    df = pd.read_csv("ogando_data/PC_stim_with_trials_specific_holos.csv")
    print(df.shape)
    print(df.groupby(["holoID_s", "Cell_uID"]).size().unique())

    sns.histplot(df, x="activity_STD", log_scale=True)
    plt.show()

    # df["is_pos"] = df["activity"] > 0
    # binned = pd.cut(df["activity_BL"], bins=20)
    # df["activity_BL_binned"] = binned.cat.rename_categories(binned.cat.categories.mid)
    # sns.lineplot(df, x="activity_BL_binned", y="is_pos", hue="cell_type")
    # plt.tight_layout()
    # plt.savefig("figures/variance_mean/check/is_pos_vs_baseline.pdf")
    # plt.show()
    # sns.regplot(df, x="activity_BL", y="activity_STD")
    # plt.tight_layout()
    # plt.savefig("figures/variance_mean/check/std_vs_baseline.pdf")
    # plt.show()
    # sf = df.groupby(["holoID_s", "Cell_uID", "cell_type"], as_index=False)[
    #     "is_pos"
    # ].mean()
    # sf["pos_biased"] = sf["is_pos"] > 0.5
    # print(sf.groupby(["holoID_s", "cell_type"])["pos_biased"].mean())

    # print(
    #     df.sort_values(by="activity_STD").head()[
    #         [
    #             "holoID_s",
    #             "Cell_uID",
    #             "activity",
    #             "activity_STD",
    #             "activity_std_per_trial",
    #         ]
    #     ]
    # )
    # print(
    #     df.query(
    #         "holoID_s == 'MBOT91_857__20230915_ensembleSizes_17.0' and Cell_uID == 'MBOT91_857_20230915__3057'"
    #     )[
    #         [
    #             "holoID_s",
    #             "Cell_uID",
    #             "trial",
    #             "activity",
    #             "activity_BL",
    #             "activity_STD",
    #             "activity_std_per_trial",
    #         ]
    #     ]
    # )

    sf = df.groupby(["cell_type", "holoID_s", "Cell_uID"], as_index=False).agg(
        activity=("activity", "mean"),
        std_mean=("activity_STD", "mean"),
        std_std=("activity_STD", "std"),
        activity_std_per_trial=("activity_std_per_trial", "mean"),
    )
    sf["std_std/mean"] = sf["std_std"] / sf["std_mean"]
    sf["activity_std_per_trial_new"] = sf["activity"] / sf["std_mean"]
    # print(sf)
    # sns.scatterplot(sf, x="std_mean", y="std_std", hue="holoID_s")
    # plt.show()
    # sns.histplot(sf, x="std_std/mean")
    # plt.show()
    # sns.scatterplot(sf, x="activity", y="std_std/mean")
    # plt.show()
    sns.regplot(sf, x="activity_std_per_trial", y="std_std/mean")
    plt.show()
    # sns.scatterplot(
    #     sf, x="activity_std_per_trial_new", y="std_std/mean", hue="cell_type"
    # )
    plt.show()

    sf = df.groupby(["cell_type", "holoID_s", "Cell_uID"], as_index=False)[
        ["activity", "activity_STD", "activity_std_per_trial"]
    ].mean()
    sf["activity_std_per_trial_new"] = sf["activity"] / sf["activity_STD"]
    # print(sf)
    ax = sns.scatterplot(
        sf, x="activity_std_per_trial_new", y="activity_std_per_trial", hue="cell_type"
    )
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    ax.plot(x, x, color="black", linestyle="--")
    plt.show()

    df = df.query("cell_type == 'SST'")
    # df = df.query("holoID_s == 'MBOT91_857__20230914_ensembleSizes_12.0'")
    # print(df.groupby("Cell_uID")["activity_STD"].agg(["mean", "std"]))
    # for cell_id in ["MBOT91_857_20230914__2191", "MBOT91_857_20230914__2200"]:
    #     print(cell_id)
    #     sf = df.query(f"Cell_uID == '{cell_id}'")
    #     # print(sf["activity"] / sf["activity_STD"])
    #     # print(sf["activity_std_per_trial"])
    #     print(
    #         sf[
    #             [
    #                 "trial",
    #                 "activity",
    #                 "activity_BL",
    #                 "activity_STD",
    #                 "activity_std_per_trial",
    #             ]
    #         ]
    #     )
    #     print(sf["activity_std_per_trial"].mean())
    #     print(sf["activity_std_per_trial"].median())


if __name__ == "__main__":
    main()
