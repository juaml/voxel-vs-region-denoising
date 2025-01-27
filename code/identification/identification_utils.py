#!/usr/bin/bash env python3
# -*- coding: utf-8 -*-

# Authors: Tobias Muganga <t.muganga@fz-juelich.de>

# Imports
import os
import yaml

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.neighbors import NearestNeighbors
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# IO utils
def get_marker_names(yaml_file):
    """
    Retrieve all defined marker names from a dataset-specific YAML file.

    Parameters
    ----------
    yaml_file : str
        The name of the YAML file (without extension) containing marker 
        definitions. This file is expected to be located in the 'preprocessing/yamls/' directory.

    Returns
    -------
    list of str
        A list of marker names prefixed with 'BOLD_' as defined in the YAML file.
        Each marker name corresponds to an entry in the 'markers' section of 
        the YAML file.

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
    
    # Construct the path to the YAML file by navigating to the 'preprocessing/yamls/' directory
    yaml_path = (
        Path(__file__).parent
        / ".."
        / "preprocessing"
        / "yamls"
        / f"{yaml_file}.yaml"
    )

    # Open and read the YAML file
    with open(yaml_path, "r") as stream:
        try:
            # Parse the YAML file into a dictionary
            dict_yaml = yaml.safe_load(stream)
            
            # Extract marker names, prefixing each with 'BOLD_' and return as a list
            return ["BOLD_" + x["name"] for x in dict_yaml["markers"]]
        
        except yaml.YAMLError as exc:
            # Catch and print any YAML parsing errors
            print(exc)


def value_check(connectomes):
    """
    Perform basic checks on the input connectome data.

    Parameters
    ----------
    connectomes : pandas.DataFrame
        A DataFrame where each row corresponds to a subject and each column 
        corresponds to a feature.

    Returns
    -------
    connectomes : pandas.DataFrame
        The input DataFrame, returned unchanged if all checks pass.

    Raises
    ------
    AssertionError
        If any of the following conditions are not met:
        - The number of subjects (rows) is 1480 or 1472.
        - The number of features (columns) is one of 4950, 79800, or 499500.
        - There are no missing (NaN) values in the DataFrame.
        - All values are within the range [-1, 1].
    
    Notes
    -----
    These checks ensure that the connectome data has the expected structure 
    and that all values fall within a valid range for connectivity matrices.
    """
    
    # Check if the number of rows (subjects) is either 1480 or 1472
    assert connectomes.shape[0] in [1480, 1472], "subjects missing!"

    # Check if the number of columns (features) is one of the expected sizes
    assert connectomes.shape[1] in [4950, 79800, 499500], (
        "some values are missing!"
    )
    
    # Check if there are any NaN (missing) values in the DataFrame
    assert np.isnan(connectomes.values).sum() == 0, (
        "some values are missing!"
    )
    
    # Ensure that no values in the DataFrame are smaller than -1
    assert (connectomes.values < -1).any() == False, (
        "some values are too small!"
    )
    
    # Ensure that no values in the DataFrame are larger than 1
    assert (connectomes.values > 1).any() == False, (
        "some values are too high!"
    )

    return connectomes


# Identification utils
def get_idiff_matrix(df1, df2):
    """
    Compute the Differential Identifiability Matrix between two datasets 
    (e.g., from two different sessions).

    Parameters
    ----------
    df1 : pandas.DataFrame
        A DataFrame where each row represents a subject's data from session 1. 
        Each column represents a feature.
    df2 : pandas.DataFrame
        A DataFrame where each row represents a subject's data from session 2. 
        The structure of `df2` should be identical to `df1` (same subjects and features).

    Returns
    -------
    idiff_matrix : pandas.DataFrame
        A Differential Identifiability Matrix. The diagonal elements represent 
        correlations between the same subject across two sessions (within-subject correlations), 
        while off-diagonal elements represent correlations between different subjects 
        (between-subject correlations).

    References
    ----------
    Amico, E., & Goñi, J. (2018). The quest for identifiability in human
    functional connectomes. Scientific Reports, 8(1), 8254.
    https://doi.org/10.1038/s41598-018-25089-1

    For implementation see:
    https://stackoverflow.com/questions/41823728/how-to-perform-correlation-between-two-dataframes-with-different-column-names

    """

    # Transpose the dataframes so that subjects are represented in columns
    # and features are represented in rows
    df1 = df1.T
    df2 = df2.T

    # Concatenate df1 and df2 along the columns, creating a multi-level index
    # The 'keys' parameter allows for labeling the columns of df1 as 'df1' and 
    # the columns of df2 as 'df2'
    idiff_matrix = pd.concat(
        [df1, df2], axis=1, keys=['df1', 'df2']
    ).corr().loc['df2', 'df1']  # Compute correlations and extract df2-df1 correlations

    # Return the resulting differential identifiability matrix
    return idiff_matrix


def get_idiff(idiff_matrix):
    """
    Calculate the differential identifiability (iDiff) between two datasets 
    based on a differential identifiability matrix.

    Parameters
    ----------
    idiff_matrix : pandas.DataFrame
        A DataFrame representing the Differential Identifiability Matrix. 
        This matrix captures how well subjects can be identified across two 
        datasets by comparing within-subject similarity (diagonal elements) 
        and between-subject similarity (off-diagonal elements).

    Returns
    -------
    idiff : float
        The differential identifiability score between two datasets, expressed 
        as a percentage. A higher iDiff value indicates better identifiability 
        (i.e., within-subject similarity is higher than between-subject similarity).

    References
    ----------
    Amico, E., & Goñi, J. (2018). The quest for identifiability in human
    functional connectomes. Scientific Reports, 8(1), 8254.
    https://doi.org/10.1038/s41598-018-25089-1

    For implementation see:
    https://stackoverflow.com/questions/41823728/how-to-perform-correlation-between-two-dataframes-with-different-column-names

    """

    # Convert the idiff_matrix to numpy array for easier manipulation
    idiff_matrix = np.array(idiff_matrix)
    
    # Extract the diagonal elements, which represent within-subject similarity
    diag = np.diagonal(idiff_matrix)
    
    # Create a mask to exclude the diagonal elements (within-subject) and 
    # extract the off-diagonal elements (between-subject similarity)
    mask = np.ones(idiff_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    off_diag = idiff_matrix[mask]
    
    # Calculate differential identifiability (iDiff) as the difference between 
    # the mean of diagonal and off-diagonal elements
    idiff = 100 * (np.mean(diag) - np.mean(off_diag))

    # Return the differential identifiability score
    return idiff


def identify(target, database, metric="spearman"):
    """ 
    Identify subjects in a target dataset based on their similarity to subjects
    in a database dataset using a specified metric.
    
    Parameters
    ----------
    target : pandas.DataFrame
        A DataFrame where each row corresponds to a subject, and each column 
        corresponds to a feature (e.g., functional connectivity data).
    database : pandas.DataFrame
        A DataFrame with the same structure as `target`, where each row is a 
        subject, and each column is a feature.
    metric : str, optional
        The metric used to calculate similarity between subjects in the 
        `target` and `database`. The default is "spearman", which ranks the 
        features and computes the correlation. 
        Other supported metrics include those from the scikit-learn NearestNeighbors API:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    
    Returns
    -------
    float
        Identification accuracy, i.e., the proportion of subjects from 
        `database` that are correctly matched to subjects in `target` based 
        on the specified metric.
    
    References
    ----------
    Finn, E. S., Shen, X., Scheinost, D., Rosenberg, M. D., Huang, J., Chun,
    M. M., Papademetris, X., & Constable, R. T. (2015). Functional connectome
    fingerprinting: Identifying individuals using patterns of brain
    connectivity. Nature Neuroscience, 18(11), 1664–1671.
    https://doi.org/10.1038/nn.4135
    """

    # Get number of subjects (rows) in the target dataset
    Dim = target.shape[0]

    # If metric is 'spearman', rank features for target and database
    if metric == "spearman":
        target = target.rank()
        database = database.rank()
        metric = "correlation"     # Set metric to 'correlation' for NearestNeighbors
    else:
        metric = metric            # Keep the specified metric

    # Initialize the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=1).fit(target)

    # Compute the first nearest neighbor in target for each subject in database
    distances, indices = nbrs.kneighbors(database)

    # Return the proportion of correct matches (identification accuracy)
    return np.mean(indices.T == np.array(list(range(Dim))))
