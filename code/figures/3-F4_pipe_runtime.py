"""This script creates figure 4 from the preprocessing runtime files.

To generate the figure use the following command:
python 3-F4_pipe_runtime.py

This script uses the following data:
- FIX_time.csv
- MINIMAL_time.csv

The script will generate the following files:
- F4_runtime_overview.png
- F4_runtime_overview.svg
- F4_runtime_overview.pdf
- F4_runtime_overview_data.csv

Author: Tobias Muganga
"""

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import repeat


# Helper functions
def get_runtime(dataset_name):
    """Get the dataframe containing runtime per pipeline for given dataset.

    Input
    -----
    dataset_name: str
        Name of the dataset for which to get the results from.
        Can be either "FIX" or "MINIMAL".

    Return
    ------
    results: pandas.DataFrame
        DataFrame containing the loaded data for plotting.
    """

    assert dataset_name in [
        "FIX",
        "MINIMAL",
    ], "Sepcify a valid dataset (FIX or MINIMAL)"
    assert isinstance(dataset_name, str), "please provide a string"

    # load time file from path
    time_path = (
        Path("..") / ".." / "results" / "runtime" / f"{dataset_name}_time.csv"
    )

    # check if file exists
    assert time_path.exists(), (
        "please provide a valid path to the results file."
    )

    print(f"Getting pipeline runtime results from {time_path}")

    data = pd.read_csv(time_path, index_col=0)

    # cosmetics for readability
    results = []
    for index in data.index:
        new_data = pd.DataFrame()
        new_data["subject"] = list(
            repeat(index.split()[0].strip("(',)"), len(data.values[0]))
        )
        new_data["task"] = list(
            repeat(index.split()[1].strip("(',)"), len(data.values[0]))
        )
        new_data["run"] = list(
            repeat(index.split()[2].strip("(',)"), len(data.values[0]))
        )
        new_data["Dataset"] = [word.split("_")[0] for word in data.columns]
        new_data["Denoislvl"] = [word.split("_")[1] for word in data.columns]
        new_data["Regions"] = [word.split("_")[2] for word in data.columns]
        new_data["Aggregation"] = [word.split("_")[3] for word in data.columns]
        new_data["Runtime"] = data.loc[index].values
        results.append(new_data)

    results = pd.concat(results)
    results = results.replace(
        {
            "V": "Voxel",
            "R": "Region",
            "FIX": "ICA-FIX",
            "MIN": "Minimal",
            "eigen": "EV",
            "mean": "Mean",
        }
    )
    results["Regions"] = results.Regions.astype("str")

    return results.reset_index(drop=True)


def get_combined_data():
    """
    Get the runtime data averaged across both datasets.

    Return
    ------
    comb_data: pandas.DataFrame
        DataFrame containing the runtime data averaged across both datasets.
    """

    # Get data for both datasets
    datasets = ["FIX", "MINIMAL"]
    all_data = pd.DataFrame()

    # Loop through datasets and combine them into one dataframe
    for dataset_name in datasets:
        data = get_runtime(dataset_name)
        data = data.sort_values(
            by=[
                "task",
                "run",
                "subject",
                "Dataset",
                "Regions",
                "Aggregation",
                "Denoislvl",
            ],
            ascending=True,
        )

        if not all_data.empty:
            # filter for complete time data
            index_list = all_data.subject.unique()
            subj_list = [
                i
                for i in index_list
                if data[data["subject"] == i].shape[0] == 48
            ]
            data = data[data["subject"].isin(subj_list)]

        all_data = pd.concat([all_data, data])

        # Average datasets
        mean_data = (
            all_data.drop(columns=["Dataset"])
            .groupby(
                [
                    "task",
                    "run",
                    "subject",
                    "Regions",
                    "Aggregation",
                    "Denoislvl",
                ]
            )
            .mean()
        )
        comb_data = mean_data.index.to_frame(index=False)
        comb_data["Runtime"] = mean_data["Runtime"].values

    return comb_data.reset_index(drop=True)


# Plot pipeline runtime data
def plot_runtime(data, ax):
    """Plot the runtime of the different pipelines.

    Input
    -----
    data: pandas.DataFrame
        DataFrame containing the runtime data averaged across datasets
        (MINIMAL and FIX) to plot.
    ax: matplotlib axis
        Axis to plot the data on.

    Return
    ------
    ax: matplotlib axis
        Axis with the plot.
    """

    assert isinstance(data, pd.DataFrame), "please provide a pandas DataFrame"

    # Average sessions and runs
    mean_data = (
        data.drop(columns=["task", "run"])
        .groupby(["subject", "Regions", "Aggregation", "Denoislvl"])
        .mean()
    )
    data = mean_data.index.to_frame(index=False)
    data["clean_aggr"] = list(
        map(
            "-".join,
            zip(
                *([list(data.Denoislvl.values), list(data.Aggregation.values)])
            ),
        )
    )
    data["Runtime"] = mean_data["Runtime"].values

    # Plot runtime
    sns.boxplot(
        data=data,
        x="clean_aggr",
        y="Runtime",
        order=["Voxel-Mean", "Region-Mean", "Voxel-EV", "Region-EV"],
        hue="Regions",
        hue_order=["100", "400", "1000"],
        width=0.4,
        palette="colorblind",
        ax=ax,
    )

    ax.set_ylim(0, 250)
    ax.set(ylabel="Seconds")
    ax.set(xlabel=None)
    ax.tick_params(axis="x", labelrotation=90)

    return ax


# Generate complete figure
def main():
    """Generate the complete figure for the runtime of the different pipelines."""

    # Get data
    fig_data = get_combined_data()

    with plt.style.context("./style.mplstyle"):
        # Set figure size
        cm = 1 / 2.54  # size conversion for easy dimensionallity adjustment
        fig = plt.figure(figsize=(8 * cm, 8 * cm))

        # Plot runtime
        ax = fig.add_subplot()
        ax = plot_runtime(fig_data, ax)

        # Save figure
        fig_path = Path("../../figures")
        file_name = "F4_runtime_overview"

        print(f"Saving runtime figure to {fig_path}")

        fig_data.to_csv(f"{fig_path}/{file_name}_data.csv")
        plt.savefig(f"{fig_path}/{file_name}.png")
        plt.savefig(f"{fig_path}/{file_name}.svg")
        plt.savefig(f"{fig_path}/{file_name}.pdf")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
