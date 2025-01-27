"""This script creates supplementary figure 2 from the identification results.

To generate Figure use the following command:
python 5-FS2_id_overview.py

This script uses the following data:
- 240426_HCP_FIX_id_results.tsv
- 240426_HCP_MINIMAL_id_results.tsv

The script will generate the following files:
- FS2_ID_overview.png
- FS2_ID_overview.svg
- FS2_ID_overview.pdf

Author: Tobias Muganga
"""

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


# Plot Identification accuracy
def plot_iacc(data, ax):
    """
    Plot the identification accuracy for the different configurations.

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

    # Plot
    sns.barplot(
        data=data,
        x="clean_aggr",
        y="id_acc",
        order=["Voxel-Mean", "Region-Mean", "Voxel-EV", "Region-EV"],
        hue="Regions",
        hue_order=["100", "400", "1000"],
        width=0.4,
        palette="flare",
        ax=ax,
    )

    ax.set_ylim(0, 1)
    ax.set(ylabel="Identification accuracy")
    ax.set(xlabel=None)
    plt.xticks(rotation=90)

    return ax


# Plot Differential Identification
def plot_idiff(data, ax):
    """
    Plot the differential identification for the different configurations.

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

    # Plot
    sns.barplot(
        data=data,
        x="clean_aggr",
        y="id_diff",
        order=["Voxel-Mean", "Region-Mean", "Voxel-EV", "Region-EV"],
        hue="Regions",
        hue_order=["100", "400", "1000"],
        width=0.4,
        palette="flare",
        ax=ax,
    )

    ax.set_ylim(0, 36)
    ax.set(ylabel="Differential Identifiability")
    ax.set(xlabel=None)
    plt.xticks(rotation=90)

    return ax


# Get data and generate plot
def generate_plots(dataset, measure, ax):
    """
    Generate the plots for the figure.

    Input
    -----
    dataset: str
        Name of the dataset for which to get the results from.
        Can be either "FIX" or "MINIMAL".
    measure: str
        The measure to be plotted. Can be either "id_acc" or "id_diff".
    ax: matplotlib.axes
        The axes to plot the data on.

    Return
    ------
    ax: matplotlib.axes
        The axes with the plotted data.
    """

    assert measure in [
        "id_acc",
        "id_diff",
    ], "Sepcify a valid measure (id_acc or id_diff)"
    assert isinstance(measure, str), "please provide a string"

    # get data
    data_df = get_id_results(dataset)

    # plot data based on measure
    if measure == "id_acc":
        ax = plot_iacc(data_df, ax)

    elif measure == "id_diff":
        ax = plot_idiff(data_df, ax)

    return ax


# Main function to generate the figure
def main():
    """
    This function generates the figure and saves it to the Figures directory.
    """
    # Define variables to loop over for plotting
    datasets = ["MINIMAL", "FIX"]
    measures = ["id_acc", "id_diff"]

    # Set up figure grid
    grid_rows, grid_cols = 2, 2
    cm = 1 / 2.54  # size conversion for easy dimensionallity adjustment

    # Create figure
    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=(13 * cm, 13 * cm))
        grid = fig.add_gridspec(grid_rows, grid_cols, hspace=0.05, wspace=0.05)
        labels = ["a", "b", "c", "d"]

        # Loop over measures and create subplots
        for (measure, dataset), (row, col), label in zip(
            product(measures, datasets),
            product(range(grid_rows), range(grid_cols)),
            labels,
        ):
            # Create subplot
            ax = fig.add_subplot(grid[row, col])
            ax = generate_plots(dataset, measure, ax)

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

            # Adjust labels to subplots
            if row == 0:
                ax.set_title(f"Data configuration: {dataset}", pad=15)
                ax.set(xlabel=None)
                ax.set(xticklabels=[])
            if col == 1:
                ax.set(ylabel=None)
                ax.set(yticklabels=[])

        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()

        fig.legend(
            title="Regions",
            handles=handles,
            labels=labels,
            loc="outside upper center",
            ncol=4,
            handlelength=1,
        )

        # Save figure
        fig_path = Path("../../figures")
        file_name = "FS2_id_overview"
        print(f"Saving ID overview figure to {fig_path}")

        plt.savefig(f"{fig_path}/{file_name}.png")
        plt.savefig(f"{fig_path}/{file_name}.svg")
        plt.savefig(f"{fig_path}/{file_name}.pdf")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
