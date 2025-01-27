"""
The script reads the log files of the junifer jobs and extracts the run time of
each marker computation. The run time is then saved in a csv file for plotting.

The script is called with the name of the dataset as argument (FIX or MINIMAL).
e.g. python 19-time_extraction.py FIX

The script assumes that the log files are located in the directory:
    preprocessing/junifer_jobs/240426_VvRHCP{dataset}/logs

The script assumes that the log files are named according to the pattern:
    run_{subject}_{scan}_{marker}.out

The csv file is saved in the directory:
    results/runtime/{dataset}_time.csv

Author: Tobias Muganga

"""

# Imports
import sys
import os
import pandas as pd

from datetime import datetime
from pathlib import Path


# custom function for time conversion from string to seconds
def str_to_sec(time_str):
    """Convert string of format H:M:S.MS to seconds."""

    time = datetime.strptime(time_str, "%H:%M:%S.%f")

    # check that default year is matching (don't know if it's the same for all systems..)
    assert (1900, 1, 1) == time.utctimetuple()[0:3], (
        "default year does not match (1900, 1, 1)"
    )
    date = datetime(1900, 1, 1)

    return (time - date).total_seconds()


def main():
    # define data set for which to grab time data from logs
    dataset = sys.argv[1]

    assert dataset in [
        "FIX",
        "MINIMAL",
    ], "please provide the name of a known data set (FIX or MINIMAL)"

    # define logs directory
    path_log_dir = (
        Path("..")
        / "preprocessing"
        / "junifer_jobs"
        / f"240426_VvRHCP{dataset}"
        / "logs"
    )

    time_dict = {}

    # ensure existance of log files for all subjects
    assert (len(os.listdir(path_log_dir)) / 3) == 371, (
        "Some log files are missing"
    )

    # loop over all files in log directory
    for path in os.listdir(path_log_dir):
        # exclude files that are not related to subjects (like collect)
        if "run" in path.split("_") and path.endswith(".out"):
            # open the file
            with open(path_log_dir / path) as file:
                # read lines
                lines = file.readlines()
                # for each line in the document
                for line in lines:
                    # strip leading and trailing white space
                    line = line.strip()
                    # if we are loading a scan
                    if "element" in line.split():
                        # grab relevant scan info
                        data = line.split()[-3:]

                        scan_info = (
                            data[0].strip("(',"),
                            data[1].strip("',"),
                            data[2].strip(")',"),
                        )
                        # start a dict with the scan info as index
                        time_dict.update({f"{scan_info}": {}})

                    # if a marker is being computed
                    if "Fitting" in line.split() and "marker" in line.split():
                        # grab marker name
                        marker = line.split()[-1]

                    # if marker computation is done
                    if "extraction" in line.split():
                        # grab the run time of the marker
                        time = str_to_sec(line.split()[-1])
                        # save marker and runtime in associated scan dict
                        time_dict[f"{scan_info}"].update({f"{marker}": time})

    # save the accummulated dictionary as dataframe
    time_df = pd.DataFrame(time_dict).T

    out_path = Path("..") / ".." / "results" / "runtime"
    file_name = f"{dataset}_time.csv"
    print(f"Saving {file_name} to {out_path}")
    time_df.to_csv(out_path / file_name)


if __name__ == "__main__":
    main()
