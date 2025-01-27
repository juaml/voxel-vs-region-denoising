"""This script generates the figure for the prediction results.

To generate the figure use the following command:
python 1-F2_pred_voxel-vs-region.py

This script uses the following data:
- all_behavior_MINIMAL.csv
- all_behavior_FIX.csv

The script will generate the following files:
- F2_prediciton_voxle-vs-region_data.csv
- F2_prediciton_voxle-vs-region_descriptive_stats.csv
- F2_prediciton_voxle-vs-region.png
- F2_prediciton_voxle-vs-region.svg
- F2_prediciton_voxle-vs-region.pdf

Author: Tobias Muganga
"""

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# Helper functions to get prediction scores
def get_pred_scores(dataset_name, regions):
    """Load dataframe from results dir and prepare it for plotting.

    Input
    -----
    dataset_name: str
        Name of the dataset for which to get the results.
        Can be either "FIX" or "MINIMAL".
    regions: str
        Number of regions for which to get the results.
        Can be either "100", "400" or "1000".

    Return
    ------
    results_df: pandas.DataFrame
        DataFrame containing the loaded data for plotting.
    """

    assert dataset_name in [
        "MINIMAL",
        "FIX",
    ], "Sepcify a valid dataset (FIX or MINIMAL)"
    assert regions in [
        "100",
        "400",
        "1000",
    ], "Sepcify a valid number of regions (100, 400 or 1000)"
    assert isinstance(dataset_name, str), "please provide a string"
    assert isinstance(regions, str), "please provide a string"

    # load results csv file from path
    path = (
        Path("..")
        / ".."
        / "results"
        / "prediction"
        / "scores"
        / dataset_name
        / f"{regions}_regions"
        / f"all_behavior_{dataset_name}.csv"
    )

    cols = ["aggregation", "conflevel", "target", "repeat", "test_corr"]
    results = pd.read_csv(path)[cols]
    results["dataset"] = dataset_name
    results["regions"] = regions

    # some cosmetics
    data = results.replace(
        {
            "VOXEL": "Voxel",
            "ROI": "Region",
            "FIX": "ICA-FIX",
            "MINIMAL": "Minimal",
            "mean": "Mean",
            "eigen": "EV",
            "PMAT24_A_CR": "Fluid Intelligence (PMAT)",
            "ReadEng_Unadj": "Reading (pronounciation)",
            "ListSort_Unadj": "Working Memory (list sorting)",
            "Age_in_Yrs": "Age (in Years)",
        }
    ).rename(columns={"conflevel": "cleanlvl"})
    # add a column that combines cleanlvl and aggregation
    data["clean_aggr"] = list(
        map(
            "-".join,
            zip(
                *([list(data.cleanlvl.values), list(data.aggregation.values)])
            ),
        )
    )

    return data


def get_data(dataset_name):
    """Get data for both datasets.

    Input
    -----
    dataset: str
        Name of the dataset for which to get the results.
        Can be either "FIX" or "MINIMAL".

    Return
    ------
    pipeline_data: pandas.DataFrame
        DataFrame containing the loaded data for plotting.
    """

    assert dataset_name in [
        "MINIMAL",
        "FIX",
    ], "Sepcify a valid dataset (FIX or MINIMAL)"
    assert isinstance(dataset_name, str), "please provide a string"

    data_df = pd.DataFrame()

    for regions in ["100", "400", "1000"]:
        # get scores from result files
        scores = get_pred_scores(dataset_name, regions)

        # accumulate dataframes
        data_df = pd.concat([data_df, scores])

    return data_df


def get_pred_data():
    """Get prediction data for both datasets.

    Return
    ------
    data: pandas.DataFrame
        DataFrame containing the loaded data.
    Pred_df: pandas.DataFrame
        DataFrame containing the loaded and formated data for plotting.
    """

    # Get data
    datasets = ["MINIMAL", "FIX"]
    data = pd.DataFrame()

    for dataset in datasets:
        pred = get_data(dataset)
        data = pd.concat([data, pred])

    data = data.reset_index(drop=True)

    # Prepare data for plotting
    Pred_df = pd.DataFrame()
    Pred_df["Regions"] = data.query("cleanlvl == 'Voxel'").regions
    Pred_df["Aggregation"] = data.query("cleanlvl == 'Voxel'").aggregation
    Pred_df["dataset"] = data.query("cleanlvl == 'Voxel'").dataset
    Pred_df["target"] = data.query("cleanlvl == 'Voxel'").target

    Pred_df["Voxel"] = data.query("cleanlvl == 'Voxel'").test_corr.values
    Pred_df["Region"] = data.query("cleanlvl == 'Region'").test_corr.values

    return data, Pred_df.reset_index(drop=True)


# Plot prediction scores
def plot_pred(data, ax):
    """Plot the prediction scores.

    Input
    -----
    data: pandas.DataFrame
        DataFrame containing the prediction scores to plot.
    ax: matplotlib axis
        Axis to plot the data on.

    Return
    ------
    ax: matplotlib axis
        Axis with the plot.
    """

    # Plot
    sns.scatterplot(
        data=data, x="Voxel", y="Region", hue="Aggregation", s=30, ax=ax
    )
    # Add diagonal line
    sns.lineplot(
        x=[-0.22, 0.55],
        y=[-0.22, 0.55],
        alpha=0.4,
        color="k",
        linestyle="--",
        linewidth=0.3,
    )

    ax.set_title(f"Target: {data.target.values[0]}")
    ax.set_ylim(-0.22, 0.53)
    ax.set_xlim(-0.22, 0.53)

    return ax


#  Generate complete figure
def main():
    """Generate the figure."""

    # Define the datasets and targets and load the data
    datasets = ["Minimal", "FIX"]
    data, data_fig = get_pred_data()
    targets = [
        "Age (in Years)",
        "Fluid Intelligence (PMAT)",
        "Working Memory (list sorting)",
        "Reading (pronounciation)",
    ]

    # Set up the figure
    cm = 1 / 2.54  # size conversion for easy dimensionallity adjustment
    grid_rows, grid_cols = 4, 2
    labels = ["a", "b"]

    # create grid for plots
    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=(14.5 * cm, 26 * cm))
        grid = fig.add_gridspec(grid_rows, grid_cols)

        for (col, dataset), label in zip(enumerate(datasets), labels):
            for target, row in zip(targets, range(grid_rows)):
                ax = fig.add_subplot(grid[row, col])
                ax = plot_pred(data_fig.query(f"target == '{target}'"), ax)

                ax.legend_.remove()
                if row == 0:
                    ax.set_title(
                        f"Data configuration: {dataset}  \n\n Target: {data_fig.target.values[0]}"
                    )
                    ax.text(
                        0,
                        1.05,
                        label,
                        fontsize=11,
                        fontweight="bold",
                        va="bottom",
                        ha="left",
                        transform=ax.transAxes,
                    )

        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()

        fig.legend(
            title="Aggregation",
            handles=handles[:2],
            labels=labels[:2],
            loc="outside upper center",
            ncol=2,
            handlelength=1,
        )

        # Save figure to file
        fig_path = Path("../../figures")
        file_name = "F2_prediciton_voxle-vs-region"

        print(f"Saving prediction score descriptive statistics to {fig_path}")
        data.to_csv(f"{fig_path}/{file_name}_data.csv")
        data.groupby(["target", "dataset", "cleanlvl", "aggregation"])[
            "test_corr"
        ].describe().to_csv(f"{fig_path}/{file_name}_descriptive_stats.csv")

        print(f"Saving prediction score voxel vs region figure to {fig_path}")
        plt.savefig(f"{fig_path}/{file_name}.png")
        plt.savefig(f"{fig_path}/{file_name}.svg")
        plt.savefig(f"{fig_path}/{file_name}.pdf")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
