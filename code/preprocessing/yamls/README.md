# VvR_FC 

## Junifer specific yaml configuration files for FC generation

Yaml files for junifer denoising and FC generation (inkl. lists of subjects).

- HCP_FIX defines all markers to be applied to the ICA-FIX preprocessed 
resting-state data to generate FC.

- HCP_MINIMAL defines all markers to be applied to the minimally preprocessed 
resting-state dataset to generate FC. 

- subjectIDs370 includes all subjects for the project.
- subjectsIDs_recompute includes 6 subjects which FCs were generated rerun 
in single jobs outside the bulk dagman ccommit to the cluster.
    - There were some problems due to cluster architecture for 4 
    subjects (Intel MKL error).
    - For 2 subjects we got NaN values for one region in the 1000 
    region parcellation. 
        - these were replaced by zeros
