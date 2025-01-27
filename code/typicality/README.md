# Typicality of FC analysis
This analysis is taken from Kopal et al., 2020 (DOI: 10.1002/hbm.25195)

Typicality (TFC) is calculated by correlating an individuals FC to the group FC. The higher the similarity the "higher" the quality aka. the lower the impact of noise on the individual FC.

To emphasise this point, TFC is correlated with mean FD as a second analysis step.

Individual TFC values are saved as well as TFC's association to mean_FD.

## Environment
Make sure to have the conda env VvR_analysis setup. If not use `env_typicality/environment.yaml` to install.

## Usage
Before executing the analysis please be sure to have the needed confound files downloaded to the `data/confounds/hcp_ya_confounds/` directory.

The specific confound files used for this project can be downloaded here: 
https://gin.g-node.org/juaml/hcp_ya_confounds

To perform TFC analysis in a local session call  
`python 1-run_typicality.py (dataset name)(yaml name without suffix)`
e.g. `python 1-run_typicality.py 240426_HCP_FIX HCP_FIX`

Assuming same directory structure as this project results will be safed to `results/typicality` directory.

To alter the path change the out_path in the `1-TFC_analysis.py` script accordingly.

To execute on a HPC using HTCondor: 
`condor_submit run_typicality.submit`

This will run TFC anlysis for both HCP data configurations (FIX and MINIMAL).