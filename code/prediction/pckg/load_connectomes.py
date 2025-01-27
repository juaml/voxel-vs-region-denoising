import yaml
import numpy as np
from pathlib import Path

from junifer.storage import HDF5FeatureStorage


def get_marker_names(yaml_file):
    """Get all defined marker names from dataset specific yaml files."""

    yaml_path = (
        Path(__file__).parent
        / ".." / ".."
        / "preprocessing"
        / "yamls"
        / f"{yaml_file}.yaml"
    )

    with open(yaml_path, "r") as stream:
        try:
            dict_yaml = yaml.safe_load(stream)
            return ["BOLD_" + x["name"] for x in dict_yaml["markers"]]
        except yaml.YAMLError as exc:
            print(exc)


def load_connectomes(preprocess, parcellation, aggregation, confound_level):
    """Load specified connectomes.

    preprocess : str
        Which HCP preprocessing was used. {"FIX" or "MINIMAL"}
    parcellation : str or int
        Which parcellation to use. Can be 100, 400, 1000
    aggregation : str
        Which voxel aggregation method was used. {"mean" or "eigen"}
    confound_level : str
        At which level was confound regression carried out.
        {"VOXEL" or "ROI"}
    """

    # select marker to load connectomes
    markers = get_marker_names(f"HCP_{preprocess}")
    marker = [marker
              for marker
              in markers
              if str(parcellation) ==  marker.split("_")[3]
              if aggregation in marker
              if confound_level[0] in marker]
        
    # load connectomes from junifer storage
    storage_path = (
        Path(__file__).parent.resolve()  
        / ".." / ".." / ".." 
        / "data" 
        / "connectomes" 
        / f"240426_HCP_{preprocess}.hdf5"
    )
    storage = HDF5FeatureStorage(storage_path, single_output=True)
    print(f"Loading connectomes for {marker[0]}") 
    connectomes = storage.read_df(feature_name=marker[0])
    
    # replacing nan values with 0
    if np.isnan(connectomes.values).sum() > 0:
        print(
            f"{np.isnan(connectomes.values).sum()} NaN values detected."
            "Replacing with 0!"
        )
        connectomes = connectomes.fillna(0)

    # average each subjects runs (RL and LR) and tasks (REST1 and REST2) 
    conn_avg = connectomes.copy().groupby(["subject"]).mean()

    # rename columns from Yeo network labels to numbers 
    conn_avg = conn_avg.set_axis(
        [str(i) for i in np.arange(len(conn_avg.columns))], axis=1
    )

    return conn_avg


if __name__ == "__main__":
    load_connectomes("FIX", 400, "mean", "VOXEL")
