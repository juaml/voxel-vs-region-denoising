"""This script creates figure 2 from the TFC-motion correlations.

To generate the figure use the following command:
python 2-F3_tfc.py

This script uses the following data:
- 240426_HCP_FIX_TFC_meanFD_corr.csv
- 240426_HCP_MINIMAL_TFC_meanFD_corr.csv

The script will generate the following files:
- F3_tfc.png
- F3_tfc.svg
- F3_tfc.pdf
- F3_tfc_data.csv

Author: Tobias Muganga
"""

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product


# Helper functions
def get_tfc_motion_corr(dataset_name):
    """Get the dataframe containing TFC-motion correlation per pipeline for
    given dataset.

    Input
    -----
    dataset_name: str
        Name of the dataset for which to get the dataframe from results.
        Can be either "MINIMAL" or "FIX".

    Return
    ------
    results: pandas.DataFrame
        DataFrame containing the loaded data for plotting.
    """

    assert dataset_name in [
        "MINIMAL",
        "FIX",
    ], "Sepcify a valid dataset (FIX or MINIMAL)"
    assert isinstance(dataset_name, str), "please provide a string"

    # load csv file from path
    results_path = (
        Path("..")
        / ".."
        / "results"
        / "typicality"
        / f"240426_HCP_{dataset_name}_TFC_meanFD_corr.csv"
    )

    assert results_path.exists(), (
        "please provide a valid path to the results file."
    )

    # load data
    data = pd.read_csv(results_path, index_col=0)

    # cosmetics to improve readability
    data["aggregation"] = [word.split("_")[-1] for word in data.index]
    data["cleanlvl"] = [word.split("_")[2] for word in data.index]
    data["dataset"] = [word.split("_")[1] for word in data.index]
    data["regions"] = [word.split("_")[-2] for word in data.index]

    results = (
        data.replace(
            {
                "V": "Voxel",
                "R": "Region",
                "FIX": "ICA-FIX",
                "MIN": "Minimal",
                "eigen": "EV",
                "mean": "Mean",
            }
        )
        .rename(columns={"Unnamed: 1": "run", "Unnamed: 2": "task"})
        .reset_index(drop=True)
    )

    col_order_list = [
        "task",
        "run",
        "dataset",
        "cleanlvl",
        "regions",
        "aggregation",
        "TFC_FD",
    ]

    results = results[col_order_list]
    results["regions"] = results.regions.astype("str")

    return results


# Plot TFC
def plot_tfc(data, ax):
    """Plot the TFC-motion correlation for the different configurations.

    Input
    -----
    data: pandas.DataFrame
        DataFrame containing the data to plot.
    ax: matplotlib axis
        Axis to plot the data on.

    Return
    ------
    ax: matplotlib axis
        Axis with the plot.
    """

    assert isinstance(data, pd.DataFrame), "please provide a pandas DataFrame"
    assert isinstance(ax, plt.Axes), "please provide a matplotlib axis"

    # Create new columns for plotting
    data["clean_aggr"] = list(
        map(
            "-".join,
            zip(
                *([list(data.cleanlvl.values), list(data.aggregation.values)])
            ),
        )
    )

    # Create bar plot with color hue
    sns.barplot(
        data=data,
        x="clean_aggr",
        y="TFC_FD",
        hue="regions",
        order=["Voxel-Mean", "Region-Mean", "Voxel-EV", "Region-EV"],
        hue_order=["100", "400", "1000"],
        palette="colorblind",
        dodge=True,
        alpha=0.6,
        errorbar=None,
        ax=ax,
    )
    # Create swarmplot to show individual data points
    sns.swarmplot(
        data=data,
        x="clean_aggr",
        y="TFC_FD",
        hue="regions",
        order=["Voxel-Mean", "Region-Mean", "Voxel-EV", "Region-EV"],
        hue_order=["100", "400", "1000"],
        palette="colorblind",
        dodge=True,
        alpha=0.5,
        ax=ax,
    )

    ax.set_ylim(-0.4, 0)
    ax.set(ylabel="Spearman correlation")

    return ax


# Generate complete figure
def main():
    """Generate the complete figure for TFC-motion assoziation."""

    # Define datasets
    datasets = ["MINIMAL", "FIX"]

    # Create empty dataframe to store data
    fig_data = pd.DataFrame()

    # Set up for figure grid
    grid_rows, grid_cols = 1, 2
    cm = 1 / 2.54  # size conversion for easy dimensionallity adjustment

    # Labels for the subplots
    labels = ["a", "b"]

    # Create figure
    with plt.style.context("./style.mplstyle"):
        # Create figure grid
        fig = plt.figure(figsize=(14 * cm, 10 * cm))
        grid = fig.add_gridspec(grid_rows, grid_cols)

        # Loop over datasets and create subplots
        for dataset, (row, col), label in zip(
            datasets, product(range(grid_rows), range(grid_cols)), labels
        ):
            # Create subplot
            ax = fig.add_subplot(grid[row, col])

            # Get data and plot
            data = get_tfc_motion_corr(dataset)
            ax = plot_tfc(data, ax)

            # Add and adjust ax labels and to subplots
            ax.legend_.remove()
            ax.text(
                0,
                1.05,
                label,
                fontsize=10,
                fontweight="bold",
                va="bottom",
                ha="left",
                transform=ax.transAxes,
            )

            if row == 0:
                ax.set_title(f"Data configutation: {dataset}", pad=15)
                ax.set(xlabel=None)
            if col == 1:
                ax.set(ylabel=None)
                ax.set(yticklabels=[])

            ax.tick_params(axis="x", labelrotation=90)

            # Add data to fig_data
            fig_data = pd.concat([fig_data, data])

        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()

        fig.legend(
            title="Regions",
            handles=handles[0:3],
            labels=labels[0:3],
            loc="outside upper center",
            ncol=3,
            handlelength=1,
        )

        # Save figure
        fig_path = Path("../../figures")
        file_name = "F3_tfc"

        print(f"Saving tfc figure to {fig_path}")

        fig_data.to_csv(f"{fig_path}/{file_name}_data.csv")
        plt.savefig(f"{fig_path}/{file_name}.png")
        plt.savefig(f"{fig_path}/{file_name}.svg")
        plt.savefig(f"{fig_path}/{file_name}.pdf")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
