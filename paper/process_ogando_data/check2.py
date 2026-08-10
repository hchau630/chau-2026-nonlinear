import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from niarb import viz


def main():
    # df = pd.read_pickle("ogando_data/compressed.pkl")
    # sns.scatterplot(
    #     df.query("exclusion_dist_um > 30 and distance_xy < 75"),
    #     x="f_baseline_std",
    #     y="df",
    # )
    # plt.savefig("figures/variance_mean/check/f_baseline_std_vs_df.png", dpi=300)
    # plt.show()
    # sns.scatterplot(
    #     df.query("exclusion_dist_um > 30 and distance_xy < 75"),
    #     x="f_baseline_std",
    #     y="df_windowed",
    # )
    # plt.savefig(
    #     "figures/variance_mean/check/f_baseline_std_vs_df_windowed.png", dpi=300
    # )
    # plt.show()

    df = pd.read_pickle("ogando_data/preprocessed_windowed.pkl")
    df_old = pd.read_pickle("ogando_data/old/20260622/preprocessed_alt.pkl")
    # df = pd.read_pickle("ogando_data/preprocessed.pkl")
    # df_old = pd.read_pickle("ogando_data/old/20260622/preprocessed_alt_entire.pkl")
    df_merged = df.merge(
        df_old,
        on=["cell_id", "holo_id", "session_id", "mouse_id", "cell_type", "_N"],
        how="left",
        suffixes=("_new", "_old"),
        validate="1:1",
    )
    df_merged["weird"] = (
        (df_merged["dr_old"] > -1)
        & (df_merged["dr_old"] < 0)
        & (df_merged["dr_new"] < -2)
        # & (df_merged["dr_new"] < -1.5)
    )
    df_new = df.copy()
    df_new["weird"] = df_merged["weird"]
    print(f"{len(df)}, {len(df_old)}, {len(df_merged)}")
    # pd.testing.assert_frame_equal(df, df_new.drop(columns="weird"))
    df_new.to_pickle("ogando_data/preprocessed_windowed.pkl")
    # df_new.to_pickle("ogando_data/preprocessed.pkl")
    # print(df_merged["weird"].sum())
    # print(
    #     df_merged.query("weird")[
    #         ["cell_id", "holo_id", "cell_type", "_N", "dr_old", "dr_new"]
    #     ]
    # )
    # sns.scatterplot(df_merged.query("~weird"), x="dr_old", y="dr_new")
    # plt.show()
    # print(
    #     df_weird[
    #         [
    #             "cell_id",
    #             "holo_id",
    #             "cell_type",
    #             "_N",
    #             # "z_dff",
    #             "z_dff_windowed",
    #             "f_baseline_std",
    #             # "df",
    #             "df_windowed",
    #         ]
    #     ]
    # )
    # df_compressed_old = pd.read_pickle("ogando_data/old/20260622/compressed.pkl")
    # df_compressed_old = df_weird[["cell_id", "holo_id", "cell_type", "_N"]].merge(
    #     df_compressed_old,
    #     on=["cell_id", "holo_id"],
    #     how="left",
    #     validate="1:1",
    # )
    # print(
    #     df_compressed_old[
    #         [
    #             "cell_id",
    #             "holo_id",
    #             "cell_type_x",
    #             "_N",
    #             "z_dff",
    #             "z_dff_baseline_std",
    #             "z_dff_normalized",
    #         ]
    #     ]
    # )
    # g = viz.figplot(
    #     df_merged,
    #     "lmplot",
    #     # "relplot",
    #     x="dr_old",
    #     y="dr_new",
    #     ci=None,
    #     grid="xyzero",
    #     scatter_kws={"s": 4, "edgecolor": "none"},
    # )
    # x = np.linspace(*plt.gca().get_xlim())
    # plt.plot(x, x, color="gray", linestyle="--")
    # g.savefig("figures/variance_mean/check/new_v_old_windowed.png", dpi=300)
    # g.savefig("figures/variance_mean/check/new_v_old.png", dpi=300)


if __name__ == "__main__":
    main()
