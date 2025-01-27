"""This script is perfroming TFC analysis and correlates TFC values with 
mean FD per existing resting-state phase-encoding run.
"""

# Authors: Tobias Muganga <t.muganga@fz-juelich.de>

# Imports
import yaml
import argparse
import pandas as pd

from pathlib import Path
from junifer.storage import HDF5FeatureStorage


# Function for typicality calcualtion
def typicality(fc_subj, fc_group, correlation_method="pearson"):
    """
    Calculate the typicality of a subject's functional connectivity (FC).

    Typicality is computed by correlating the subject's FC with the group-level
    average FC. The function returns a typicality score between 0 and 1, where
    higher scores indicate that the subject's FC is more similar to the group.

    Parameters:
    -----------
    fc_subj : pd.Series or pd.DataFrame
        The subject's functional connectivity data.

    fc_group : pd.Series or pd.DataFrame
        The group-level average functional connectivity data.

    correlation_method : str, optional
        The method of correlation to use, with "pearson" as the default.
        Other options include "spearman" or "kendall".

    Returns:
    --------
    float
        A typicality score between 0 and 1, where a higher score indicates
        greater similarity between the subject's FC and the group average FC.

    References:
    -----------
    Kopal J, Pidnebesna A, Tomeček D, Tintěra J, Hlinka J. Typicality of
    functional connectivity robustly captures motion artifacts in rs-fMRI
    across datasets, atlases, and preprocessing pipelines.
    Hum Brain Mapp. 2020; 41: 5325–5340. https://doi.org/10.1002/hbm.25195

    """

    # Compute correlation between the subject's FC and group-level FC
    correlation = fc_subj.corr(pd.Series(fc_group), method=correlation_method)

    # Scale correlation to typicality score between 0 and 1
    return (1 + correlation) / 2


# Some helpers to grab FD and marker names
def get_fd(run, subj, sess, pipe=None):
    """Load framewise displacement (FD) from confounds directory.

    Parameters
    ----------
    run: str
        phase-encoding run (LR or RL).
    subj: str
        subject ID.
    sess: str
        resting-state session (REST1 or REST2).
    pipe: str (Optional)
        confounds for which cleaning pipeline.
        Minimal preprocessing (no_FIX) or ICA-FIX preprocessing (FIX).
        Defaults to confounds for ICA-FIX cleaned data.

    Return
    confounds: pandas.Series
        values of FD column from confound file.
    """

    conf_dir = (
        Path("..") 
        / ".."
        / "data" 
        / "confounds" 
        / "hcp_ya_confounds"
    )
    if pipe == "noFIX":
        # print("Getting confounds for minimally cleaned data.")
        # Case for minimally cleaned data
        confound_file = (
            Path(f"{subj}")
            / "MNINonLinear"
            / "Results"
            / f"rfMRI_{sess}_{run}"
            / f"Confounds_{subj}_{pipe}.tsv"
        )
    elif pipe == "ICA-FIX":
        # Case for ICA-FIX cleaned data
        # print("Getting confounds for ICA-FIX cleaned data.")
        confound_file = (
            Path(f"{subj}")
            / "MNINonLinear"
            / "Results"
            / f"rfMRI_{sess}_{run}"
            / f"Confounds_{subj}.tsv"
        )

    conf_path = Path(conf_dir) / confound_file

    conf_df = pd.read_csv(conf_path, sep="\t")

    return conf_df["FD"]


def get_mean_fd(tfc, dataset):
    """
    Get mean framewise displacement (FD) values for each subject.

    This function calculates the mean framewise displacement (FD) for each scan
    associated with a subject.

    Parameters:
    -----------
    tfc : pd.DataFrame
        Typicality of functional connectivity dataframe. The index is expected
        to contain the scans for which FD values are to be calculated.

    dataset : str
        The name of the dataset. Based on the string provided, this function
        will infer the preprocessing pipeline.

    Returns:
    --------
    pd.Series
        A pandas Series containing the mean FD values for each scan, with the
        index corresponding to the subjects in the input dataframe `tfc`.

    Raises:
    -------
    AssertionError
        If the dataset is not provided as a string.

    """

    # Ensure the dataset name is correct
    assert isinstance(
        dataset, str
    ), "Please provide the dataset as string type."

    # Adjust the pipeline according to the dataset
    if "FIX" in dataset.split("_"):
        print("Getting mean FD for ICA-FIX cleaned dataset.")
        dataset = "ICA-FIX"
    elif "MINIMAL" in dataset.split("_"):
        print("Getting mean FD for minimally cleaned dataset.")
        dataset = "noFIX"
    else:
        print("Please privde a valid dataset name (HCP_FIX or HCP_MINIMAL)")

    # Retrieve and calculate mean FD for each run
    fd = [get_fd(*scan, pipe=dataset).mean() for scan in tfc.index]

    # Return mean FD values as pandas Series, indexed by the same index as tfc
    return pd.Series(fd, index=tfc.index)


# IO utils
def get_marker_names(yaml_file):
    """
    Retrieve all defined marker names from a dataset-specific YAML file.

    Parameters
    ----------
    yaml_file : str
        The name of the YAML file (without extension) containing marker
        definitions. This file is expected to be located in the 
        'preprocessing/yamls/' directory.

    Returns
    -------
    list of str
        A list of marker names prefixed with 'BOLD_' as defined in the YAML 
        file. Each marker name corresponds to an entry in the 'markers' section
        of the YAML file.

    Raises
    ------
    yaml.YAMLError
        If there is an error in parsing the YAML file, the exception is caught
        and printed.

    Notes
    -----
    - The YAML file should contain a 'markers' section with each marker
      defined as a dictionary containing at least a 'name' key.
    - Marker names are prefixed with 'BOLD_' to standardize the format.
    """

    # Construct path to YAML file by navigating to
    # 'preprocessing/yamls/' directory
    yaml_path = (
        Path("..")
        / "preprocessing"
        / "yamls"
        / f"{yaml_file}.yaml"
    )

    # Open and read the YAML file
    with open(yaml_path, "r") as stream:
        try:
            # Parse the YAML file into a dictionary
            dict_yaml = yaml.safe_load(stream)

            # Extract marker names, prefixing each with 'BOLD_' and return 
            # as a list
            return ["BOLD_" + x["name"] for x in dict_yaml["markers"]]

        except yaml.YAMLError as exc:
            # Catch and print any YAML parsing errors
            print(exc)


def parse_args():
    """Parse and return the command-line arguments.

    The function expects two required arguments:

    - dataset: Name of dataset to convert. Valid options: 'FIX' or 'MINIMAL'.
    - yaml: The path to the YAML configuration file to be parsed.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.

    """

    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Convert the HDF5 feature storage into .tsv files."
    )

    # Add required positional argument for the dataset to convert
    parser.add_argument(
        "dataset",
        type=str,
        help=("Which dataset to convert. {'FIX', 'MINIMAL'}"),
    )

    # Add required positional argument for the YAML file to parse
    parser.add_argument("yaml", type=str, help=("Which yaml file to parse."))

    # Parse and return command-line arguments
    return parser.parse_args()


def main():
    """Loading the FC from HDF5 storage obejct given a dataset and yaml name
    to run TFC analysis and perform correlation with mean FD values
    (according to Kopal et al., 2020)

    Input
    -----
    dataset: str
        name of the storage file without suffix as specified in yaml file
    yaml: str
        name of the yaml file used for preprocessing without suffix

    """

    args = parse_args()
    dataset = args.dataset
    yaml = args.yaml

    # path to stored connectomes generated by junifer (stored as hdf5)
    path_storage = (
        Path("..")
        / ".."
        / "data"
        / "connectomes"
        / f"{dataset}.hdf5"
    )

    # path to results directory to save output
    path_out = Path("..") / ".." / "results" / "typicality"

    # load hdf5 stroage object
    storage = HDF5FeatureStorage(path_storage, single_output=True)
    # grab list of markers from yaml to iterate over in the storage object
    markers = get_marker_names(yaml)

    # dicts for accumulation of results
    tfc_fd_corr = {}
    tfc_fd_raw = {}

    # Generate typicality and TFC x mean FD for each marker
    for marker in markers:

        # Read FC for a single marker
        conn_df = storage.read_df(feature_name=marker)

        # Per phase encoding and session
        for run, task in conn_df.index.droplevel(1).unique():
            print(f"Generating TFC for {marker}, {task}, {run}")
            # Select relevant rows (connectomes)
            conn = conn_df.query(
                f"phase_encoding == '{run}' & task == '{task}'"
            )

            # Compute mean connectome aka average all indecies in a column
            avg_conn = conn.mean(axis=0)

            # Compute Typicality of FC (TFC)
            typicality_series = conn.apply(
                typicality,
                axis=1,
                fc_group=avg_conn,
                correlation_method="pearson",
            )

            # Get mean framewise displacement (FD)
            mean_fd = get_mean_fd(typicality_series, dataset)

            # Save tfc and mean FD values
            tfc_mean_fd_df = pd.concat(
                [
                    pd.Series(typicality_series.sort_index(), name="tfc"),
                    pd.Series(mean_fd.sort_index(), name="mean_fd"),
                ],
                axis=1,
            )

            # Accumulate the raw TFC and mean FD values per run
            tfc_fd_raw.update({(marker, run, task): tfc_mean_fd_df})

            # Compute correlation of TFC and mean FD and accumulate
            tfc_fd_corr.update(
                {
                    (marker, run, task): tfc_mean_fd_df.corr(
                        method="spearman"
                    )["mean_fd"].iloc[0]
                }
            )

    # Save raw TFC and mean FD values
    tfc_fd_raw_df = pd.DataFrame(tfc_fd_raw, index=[])
    tfc_fd_raw_df.to_csv(
        (Path(path_out) / f"{dataset}_TFC_meanFD_values.csv")
    )

    # Save TFC X mean FD correlation
    tfc_fd_corr_df = pd.DataFrame(tfc_fd_corr, index=["TFC_FD"]).T
    tfc_fd_corr_df.to_csv((Path(path_out) / f"{dataset}_TFC_meanFD_corr.csv"))


if __name__ == "__main__":
    main()
