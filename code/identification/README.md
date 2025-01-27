# Identification analysis
This analysis is inspired by Finn et al. 2015 and Amico and Goñi 2018.

Finn's identification accuracy and Amico & Goñi's differential identifiability are generated and saved to the `results/identification` directory.

## Environment
Before running analysis, make sure to have the conda env VvR_analysis installed and activated. 
To do so use the provided `environment.yaml` in `env_identification/` 
## Usage
To perform ID analysis in a local session, use: 
`python 1-run_identification.py (dataset name)(yaml name)` 
e.g. 
`python 1-run_identification.py 240626_HCP_FIX HCP_FIX` (don't include file type suffix)

To run identification anlysis for both datasets (FIX and MINIMAL) on the HPC running HTCondor, use the provided `*.submit` file and call:
`condor_submit run_identification.submit`

Assuming same directory structure results will be safed to 
`../results/identification/`