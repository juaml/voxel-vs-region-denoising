# Preprocessing

## Navigating the Repo
`env/`: contains an env file to replicate the exact environment for preprocessing

`extensions/`: contains the scripts needed to execute preprocessing in the 
Junifer environment
-> go here, if you want to adjust the preprocessing pipeline itself 

`junifer_jobs/`: contains the junifer specific job directories per Dataset 
-> go here to see what exactly was executed and to rerun the preprocessing

`yamls/`: contains the specifications needed to generate the junifer jobs 
-> go here, if you want to change specifications or scope of a junifer_job

## Requirements 
To replicate the exact environment, use the `env_preproc.yml` provided in the `env/` directory.

This project was designed and executed on protected subject data from the Human Connectome Procejt (HCP). A data access agreement with the HCP is needed before running the preprocessing code.

This can be attained here: https://db.humanconnectome.org/

The data for this project is accessed using Datalad. For help setting up datalad, see: 

https://www.datalad.org
Documentation: https://handbook.datalad.org/en/latest/

Preprocessing is carried out with the Junifer preprocessing package. Instructions and help for setting up Junifer without the provided env file can be found here:

https://github.com/juaml/junifer
Documentation: https://juaml.github.io/junifer/main/index.html

## Running on a HPC cluster managed by HTC Condor 

To execute the preprocessing on all subjects on a HTC Condor managed system:
Submit the dag file found in `junifer_jobs` directory per Dataset (FIX & MINIMAL).

To submit a dag file: 
`condor_submit_dag --maxjobs 50 240426_VvRHCPFIX.dag` (or `240426_VvRHCPMINIMAL.dag`)
- maxjobs caps the jobs that run at the same time to 50

To generate the submit .dag file to submit without submitting directly:
`junifer queue yamls/*filename*.yaml --element yamls/subjectIDs370.txt`

To submit directly add `--submit` flag to the above call.

To recreate a fresh junifer job with an existing name add `--overwrite` flag to the above call.

## Running on a local machine 

To run preprocessing for all subjects on a local machine call:
`junifer run yamls/filename.yaml --element yamls/subjectIDs370.txt`

Or for a single subject call:
`junifer run yamls/filename.yaml --element (HCP subject ID)`
