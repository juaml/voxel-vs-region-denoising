"""This script generates supplementary figure 3 that shows the prediction 
accuracy for both data configurations (Minimal and ICA-FIX).

To generate Figure use the following command:
python 6-FS03_pred_overview.py

This script uses the following data:
- all_behavior_MINIMAL.csv
- all_behavior_FIX.csv

The script will generate the following files:
- FS3_prediciton_overview_data.csv
- FS3_prediciton_overview.png
- FS3_prediciton_overview.svg
- FS3_prediciton_overview.pdf

Author: Tobias Muganga
"""

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# Helper functions to get results
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

    # check if file exists
    assert path.exists(), "please provide a valid path to the results file."

    # load relevant data
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

    assert isinstance(data, pd.DataFrame), "please provide a pandas DataFrame"
    assert isinstance(ax, plt.Axes), "please provide a matplotlib axis"

    # Create plot
    sns.boxplot(
        data=data,
        x="clean_aggr",
        y="test_corr",
        hue="regions",
        hue_order=["100", "400", "1000"],
        width=0.8,
        boxprops={"alpha": 0.8},
        ax=ax,
    )

    ax.set_title(f"Target: {data.target.values[0]}")
    ax.set_ylim(-0.3, 0.6)
    ax.set(ylabel="Pearson correlation")
    ax.set(xlabel=None)

    return ax


#  Generate complete figure
def main():
    """Generate the complete figure for the prediction scores (accuracy)."""

    # Define datasets
    datasets = ["MINIMAL", "FIX"]

    # Create empty dataframe to store data
    fig_data = pd.DataFrame()

    # Create grid for plots
    cm = 1 / 2.54  # size conversion for easy dimensionallity adjustment
    grid_rows, grid_cols = 4, 2

    # Define labels for subplots
    labels = ["a", "b"]

    # Create the figure
    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=(20 * cm, 30 * cm))
        grid = fig.add_gridspec(grid_rows, grid_cols)

        # Loop over datasets and figure columns
        for (col, dataset), label in zip(enumerate(datasets), labels):
            # Load data
            data = get_data(dataset)
            targets = data.target.unique()

            # Add data to fig_data
            fig_data = pd.concat([fig_data, data])

            # Loop over targets and figure rows
            for target, row in zip(targets, range(grid_rows)):
                # Create subplot
                ax = fig.add_subplot(grid[row, col])
                ax = plot_pred(data.query(f"target == '{target}'"), ax)

                # Add and adjust ax labels
                ax.legend_.remove()
                if col > 0:
                    ax.set_ylabel(None)
                if row < (grid_rows - 1):
                    ax.set(xticklabels=[])
                if row == 0:
                    ax.set_title(
                        f"Data configuration: {dataset}  \n\n Target: {data.target.values[0]}"
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

        # Add legend to figure
        handles, labels = plt.gca().get_legend_handles_labels()

        fig.legend(
            title="Regions",
            handles=handles[:3],
            labels=labels[:3],
            loc="outside upper center",
            ncol=4,
            handlelength=1,
        )

        # Save figure to file
        fig_path = Path("../../figures")
        file_name = "FS03_pred_overview"
        print(f"Saving prediction score overview figure to {fig_path}")

        fig_data.to_csv(f"{fig_path}/{file_name}_data.csv")
        plt.savefig(f"{fig_path}/{file_name}.png")
        plt.savefig(f"{fig_path}/{file_name}.svg")
        plt.savefig(f"{fig_path}/{file_name}.pdf")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
