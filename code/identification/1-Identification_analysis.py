"""This script performs identification analysis."""

# Authors: Tobias Muganga <t.muganga@fz-juelich.de>

# Imports 
import argparse

from pathlib import Path
import pandas as pd
import numpy as np

import identification_utils as id
from junifer.storage import HDF5FeatureStorage


def parse_args():
    """Parse and return the command-line arguments. 

    The function expects two required arguments:
    
    - dataset: The name of the dataset to convert. Valid options are 'FIX' or 'MINIMAL'.
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
    parser.add_argument(
        "yaml",
        type=str,
        help=("Which yaml file to parse.")
    )

    # Parse and return command-line arguments
    return parser.parse_args()


def main():
    """
    Load functional connectivity (FC) data from an HDF5 storage object, perform identification 
    analysis, and compute identification accuracy and differential identifiability.

    Inputs:
    -------
    dataset : str
        The name of the dataset (without file extension) as specified in the YAML file.
    yaml : str
        The name of the YAML file used for preprocessing (without file extension).
    
    Workflow:
    ---------
    1. Load the dataset and YAML file names from command-line arguments.
    2. Load the HDF5 storage object for the given dataset.
    3. Retrieve marker names from the YAML file, which are used to access specific feature data.
    4. For each marker:
        - Load the corresponding connectomes data.
        - Replace NaN values with 0.
        - Average connectomes across runs for each subject.
        - Split connectomes into target (REST1) and database (REST2) for identification.
        - Ensure that subjects align between target and database.
        - Calculate identification accuracy and differential identifiability.
        - Save identification matrices and accumulate the results for each marker.
    5. Save the final results in a `.tsv` file.
    
    Outputs:
    --------
    - Saves an identification matrix for each marker as a `.tsv` file.
    - Saves a results file containing identification accuracy and differential identifiability 
      metrics for the dataset.
    
    """

    args = parse_args()
    dataset = args.dataset
    yaml = args.yaml

    # Base directory path
    base_path = Path(__file__).parent / ".." / ".."

    # Storage file path
    storage_path = base_path / "data" / "connectomes" / f"{dataset}.hdf5"

    # Out directory path
    out_path = base_path / "results" / "identification"

    # Load storage object
    storage = HDF5FeatureStorage(storage_path, single_output=True)

    # Get marker names as specificed in the yaml file used for preprocessing
    markers = id.get_marker_names(yaml)

    # Start of accumulation pattern for identification results
    results = []

    for marker in markers:

        # Read stroage object to pandas dataframe
        print(f"Loading dataframe for {marker}...")
        connectomes = storage.read_df(feature_name=marker)

        # Replacing nan values with 0
        if np.isnan(connectomes.values).sum() > 0:
            print(
                f"{np.isnan(connectomes.values).sum()} NaN values detected."
                "Replacing with 0!"
            )
            connectomes = connectomes.fillna(0)

        # Average runs of the same task and subject (RL and LR)
        conn_run_avg = (
            id.value_check(connectomes.copy())
                .groupby(["subject", "task"])
                .mean()
        )

        # Split connectomes into target and database for identification 
        # (Rest1 = target / Rest2 = database) 
        target = conn_run_avg.query("task == 'REST1'")
        target.index = target.index.droplevel(1)

        database = conn_run_avg.query("task == 'REST2'")
        database.index = database.index.droplevel(1)

        target_subj = target.index.get_level_values("subject")
        database_subj = database.index.get_level_values("subject")

        # Make sure subjects align in both dataframes
        assert (target_subj == database_subj).sum() == 370, ("subjects don't" 
            " match between target and database!"
        )

        # Calcualte identification accuracy and differential identifiabiliy
        print("Calculating identification accuracy")
        acc1 = id.identify(target, database, metric='spearman')
        acc2 = id.identify(database, target, metric='spearman')
        iacc = (acc1 + acc2) / 2

        # Generate identification matrix
        print("Generating identification matrix")
        idiff_matrix = id.get_idiff_matrix(target, database)

        # Calculate differential identifiability
        print("Calculating differential identifiability")
        idiff = id.get_idiff(idiff_matrix)

        # Parse marker name into condtions
        dataqual, cleanlvl, num_parcel, aggr = marker.split("_")[1:5]
        
        # Accumulate results
        results.append({
            'dataset': dataqual,
            'cleanlvl': cleanlvl, 
            'regions': num_parcel,
            'aggregation': aggr,
            'id_diff': idiff,
            'id_acc': iacc
        })

        # Save identification matrix
        print(
            f"Saving ID matrix for {marker} to {out_path / marker}_"
            "idiff_mat.tsv"
        )
        idiff_matrix.to_csv(
            f"{out_path / marker}_idiff_mat.tsv", index=True, sep="\t"
        )

    # Save results dataframe
    print(
        f"Saving results for {dataset} to {out_path / dataset}_id_results.tsv"
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"{out_path / dataset}_id_results.tsv", index=True, sep="\t"
    )


if __name__ == "__main__":
    main()
