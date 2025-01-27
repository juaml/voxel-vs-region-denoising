"""This script creates supplementary figure 1 from the identification results.

To generate Figure use the following command:
python 04-FS1_id_voxel-vs-region.py

This script uses the following data:
- 240426_HCP_FIX_id_results.tsv
- 240426_HCP_MINIMAL_id_results.tsv

The script will generate the following files:
- FS1_id_voxel-vs-region.png
- FS1_id_voxel-vs-region.svg
- FS1_id_voxel-vs-region.pdf
- FS1_id_voxel-vs-region_data.csv

Author: Tobias Muganga
"""

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from itertools import product


# Helper function to get results
def get_id_results(dataset_name):
    """Load dataframe from results dir and prepare it for plotting.

    Input
    -----
    dataset_name: str
        Name of the dataset for which to get the results from.
        Can be either "FIX" or "MINIMAL".

    Return
    ------
    results_df: pandas.DataFrame
        DataFrame containing the loaded data for plotting.
    """

    assert dataset_name in [
        "MINIMAL",
        "FIX",
    ], "Sepcify a valid dataset (FIX or MINIMAL)"
    assert isinstance(dataset_name, str), "please provide a string"

    # load tsv file from path
    results_path = (
        Path("..")
        / ".."
        / "results"
        / "identification"
        / f"240426_HCP_{dataset_name}_id_results.tsv"
    )

    assert results_path.exists(), (
        "please provide a valid path to the results file."
    )

    data = pd.read_csv(results_path, index_col=0, sep="\t")

    # cosmetics to improve readability
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
        .sort_values(by=["cleanlvl", "aggregation"], ascending=False)
        .sort_values(by=["regions"])
        .rename(
            columns={
                "cleanlvl": "Denoislvl",
                "aggregation": "Aggregation",
                "regions": "Regions",
                "dataset": "Dataset",
            }
        )
        .reset_index(drop=True)
    )
    results["Regions"] = results.Regions.astype("str")
    results["clean_aggr"] = list(
        map(
            "-".join,
            zip(
                *(
                    [
                        list(results.Denoislvl.values),
                        list(results.Aggregation.values),
                    ]
                )
            ),
        )
    )

    return results


def get_id_data(measure):
    """This function generates the data for the figure.

    Input
    -----
    measure: str
        The measure to be plotted. Can be either "id_acc" or "id_diff".

    Return
    ------
    Id_df: pandas.DataFrame
        DataFrame containing the data for the figure.
    """

    assert measure in [
        "id_acc",
        "id_diff",
    ], "Sepcify a valid measure (id_acc or id_diff)"
    assert isinstance(measure, str), "please provide a string"

    datasets = ["MINIMAL", "FIX"]
    data = pd.DataFrame()

    for dataset in datasets:
        id = get_id_results(dataset)
        data = pd.concat([data, id])

    data = data.reset_index(drop=True)

    Id_df = pd.DataFrame()
    Id_df["Regions"] = data.query("Denoislvl == 'Voxel'").Regions
    Id_df["Aggregation"] = data.query("Denoislvl == 'Voxel'").Aggregation
    Id_df["Dataset"] = data.query("Denoislvl == 'Voxel'").Dataset
    Id_df["measure"] = measure

    if measure == "id_acc":
        Id_df["Voxel"] = data.query("Denoislvl == 'Voxel'").id_acc.values
        Id_df["Region"] = data.query("Denoislvl == 'Region'").id_acc.values

    elif measure == "id_diff":
        Id_df["Voxel"] = data.query("Denoislvl == 'Voxel'").id_diff.values
        Id_df["Region"] = data.query("Denoislvl == 'Region'").id_diff.values

    return Id_df.reset_index(drop=True)


# Helper function to plotting
def plot_id(data, measure, ax):
    """Plot the data for the figure.

    Input
    -----
    data: pandas.DataFrame
        DataFrame containing the data to be plotted.
    measure: str
        The measure to be plotted. Can be either "id_acc" or "id_diff".
    ax: matplotlib.axes
        The axes to plot the data on.

    Return
    ------
    ax: matplotlib.axes
        The axes with the plotted data
    """

    assert measure in [
        "id_acc",
        "id_diff",
    ], "Sepcify a valid measure (id_acc or id_diff)"
    assert isinstance(measure, str), "please provide a string"
    assert isinstance(ax, plt.Axes), "please provide a valid axes"

    # plot data
    sns.scatterplot(
        data=data,
        x="Voxel",
        y="Region",
        hue="Dataset",
        style="Aggregation",
        markers=["o", "^"],
        s=50,
        ax=ax,
    )

    # cosmetics for the plot
    if measure == "id_diff":
        # add a line to show the identity line
        sns.lineplot(
            x=[0, 36],
            y=[0, 36],
            alpha=0.4,
            color="k",
            linestyle="--",
            linewidth=0.3,
        )
        ax.set_ylim(18, 36)
        ax.set_xlim(18, 36)
        plt.xticks(np.arange(18, 37, 2))
        plt.yticks([20, 22, 24, 26, 28, 30, 32, 34, 36])

        ax.set_title("Differential Identifiability", pad=15)

    if measure == "id_acc":
        # add a line to show the identity line
        sns.lineplot(
            x=[0, 1.01],
            y=[0, 1.01],
            alpha=0.4,
            color="k",
            linestyle="--",
            linewidth=0.3,
        )
        ax.set_ylim(0.75, 1.01)
        ax.set_xlim(0.75, 1.01)
        plt.xticks([0.75, 0.8, 0.85, 0.9, 0.95, 1])
        plt.yticks([0.8, 0.85, 0.9, 0.95, 1])
        ax.set_title("Identification accuracy", pad=15)

    return ax


# Main function to generate figure
def main():
    """
    This function generates the figure and saves it to the Figures directory.
    """

    # Define the measures to be plotted
    measures = ["id_acc", "id_diff"]

    # Create empty dataframe to store data
    fig_data = pd.DataFrame()

    # Set up figure grid
    cm = 1 / 2.54  # size conversion for easy dimensionallity adjustment
    grid_rows, grid_cols = 1, 2

    # labels for the subplots
    labels = ["a", "b"]

    # Create figure
    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=(17 * cm, 9 * cm))
        grid = fig.add_gridspec(grid_rows, grid_cols, hspace=0.05, wspace=0.05)

        # Loop over measures and create subplots
        for measure, (row, col), label in zip(
            measures, product(range(grid_rows), range(grid_cols)), labels
        ):
            # Create subplot
            ax = fig.add_subplot(grid[row, col])

            # Get data and plot
            data_df = get_id_data(measure)
            ax = plot_id(data_df, measure, ax)

            # Add labels to subplots
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

            # Add data to fig_data
            fig_data = pd.concat([fig_data, data_df])

        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()

        fig.legend(
            handles=handles,
            labels=labels,
            loc="outside upper center",
            ncol=6,
            handlelength=1,
        )

        # Save figure to file
        fig_path = Path(".")  # ("../../figures")
        file_name = "FS1_id_voxel-vs-region"
        print(f"Saving ID voxel vs region figure to {fig_path}")

        fig_data.to_csv(f"{fig_path}/{file_name}_data.csv")
        plt.savefig(f"{fig_path}/{file_name}.png")
        plt.savefig(f"{fig_path}/{file_name}.svg")
        plt.savefig(f"{fig_path}/{file_name}.pdf")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
