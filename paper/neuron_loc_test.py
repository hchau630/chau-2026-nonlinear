import seaborn as sns
import matplotlib.pyplot as plt
from scipy import spatial

from niarb import neurons


def main():
    N = 24900
    variables = ["cell_type", "space"]
    cell_types = ["PYR", "PV", "SST"]
    space_extent = (1050.0, 800.0)
    # min_dist = 3.0
    # min_dist = 0.0
    min_dist = 4.0
    min_dist_cell_types = {"PV": 12.0, "SST": 15.0}
    w_dims = []

    x = neurons.sample(
        N=N,
        variables=variables,
        cell_types=cell_types,
        cell_probs_strict=True,
        space_extent=space_extent,
        min_dist=min_dist,
        min_dist_cell_types=min_dist_cell_types,
        w_dims=w_dims,
    )
    print(spatial.distance.pdist(x["space"]).min())
    df = x.to_framelike()
    print(df.groupby("cell_type").size())

    sns.relplot(
        df.query("cell_type != 'PYR'"),
        # df,
        x="space[0]",
        y="space[1]",
        hue="cell_type",
        s=2,
        palette="bright",
    )
    plt.show()


if __name__ == "__main__":
    main()
