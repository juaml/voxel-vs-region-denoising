import pandas as pd
from pathlib import Path


def _load_hcp_csv(columns=None, subjects=None, restriction="unrestricted"):
    """Get data from a specific HCP csv file.

    Parameters
    ----------
    columns : str | list of str | None
        Column(s) for which to obtain data from the csv.
        If None, get all columns.
    subjects : str | list of str | None
        Subject ID's of subjects to get.
        If None, get all subjects.
    restriction : str
        Which of the files to use. Can be 'unrestricted' or 'restricted'.

    Returns
    -------
    pandas.DataFrame
        DataFrame of data specified.
    """

    assert restriction in [
        "unrestricted",
        "restricted",
    ], "'restriction' can be 'restricted' or 'unrestricted'."

    hcp_data_path = (
        Path(__file__).parent.resolve() / ".." / "data" / f"{restriction}.csv"
    )
    hcp_data = pd.read_csv(hcp_data_path, dtype={"Subject": str})
    hcp_data.set_index("Subject", inplace=True)

    if columns is not None:
        hcp_data = hcp_data[columns]

    if subjects is not None:
        if isinstance(subjects, list):
            subjects = list(map(str, subjects))
        else:
            assert isinstance(
                subjects, str
            ), "Subjects should be None, str, or list of str!"
        hcp_data = hcp_data.loc[subjects]

    return hcp_data


def _load_fd(subjects=None):
    fd_path = (
        Path(__file__).parent.resolve() / ".." / "data" / "FD_HCP_370subs.csv"
    )
    fd_data = pd.read_csv(fd_path, dtype={"subjects": str}, index_col=0)
    fd_data["FD_REST1_REST2"] = fd_data[["REST1", "REST2"]].mean(axis=1)
    if subjects is None:
        return fd_data

    if isinstance(subjects, str):
        subjects = [subjects]

    return fd_data.loc[subjects]


def _load_tiv(subjects=None):
    tiv_path = (
        Path(__file__).parent.resolve()
        / ".."
        / "data"
        / "vol_TIV_HCP_370subs.csv"
    )
    tiv_data = pd.read_csv(tiv_path, dtype={"subjects": str}, index_col=0)

    if subjects is None:
        return tiv_data

    if isinstance(subjects, str):
        subjects = [subjects]

    return tiv_data.loc[subjects]


def load_hcp_phenotypes(columns=None, subjects=None):
    """Get HCP phenotype from both restricted and unrestricted csv files.

    Parameters
    ----------
    columns : str | list of str | None
        Column(s) for which to obtain data from the whole data.
        If None, get all columns.
    subjects : str | list of str | None
        Subject ID's of subjects to get.
        If None, get all subjects.

    Returns
    -------
    pandas.DataFrame
        DataFrame of data specified.
    """

    unrestricted_data = _load_hcp_csv(
        subjects=subjects, restriction="unrestricted"
    )
    restricted_data = _load_hcp_csv(
        subjects=subjects, restriction="restricted"
    )

    combined_data = pd.concat([unrestricted_data, restricted_data], axis=1)

    fd = _load_fd(subjects)
    combined_data["FD_REST1_REST2"] = fd["FD_REST1_REST2"]
    combined_data["CAT_TIV"] = _load_tiv(subjects)["vol_TIV"]
    
    if isinstance(columns, str):
        columns = [columns]

    if columns is not None:
        assert isinstance(
            columns, list
        ), "Provide columns as str or list or str!"
        for col in columns:
            assert isinstance(col, str), (
                "If columns are provided as list, "
                "each element should still be a str."
            )

        combined_data = combined_data[columns]

    return combined_data


if __name__ == "__main__":
    load_hcp_phenotypes()
