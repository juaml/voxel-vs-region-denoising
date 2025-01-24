# Voxel-wise or Region-wise Nuisance Regression for Functional Connectivity Analyses
Individual specificity and predicatbility using region-level denoised rs-FC.

The main idea is to perform denoising on the regional time series and compare it to the FCs resulting from denoising performed on the voxel time series.

Comparisons include the measures of identification accuracy, differential identification, typicality of FC and prediction accuracy for age and fluid intelligence, reading and working memory in the HCP S1200 dataset.

Preprint: https://www.biorxiv.org/content/10.1101/2024.12.10.627766v1

## Data
This project uses data from the Human connectome project (HCP) accessed through DataLad. A data use agreement with the HCP is needed to obtain AWS credentials used by DataLad to access the dataset.

Further instructions regarding AWS credentials for DataLad can be found in the DataLad handbook: 
https://handbook.datalad.org/en/latest/usecases/HCP_dataset.html#usecase-hcp-dataset

This project uses confound files with precomputed confound files. These can be shared upon request with proof of agreement to the HCP's data use agreement (https://db.humanconnectome.org/).

## Contact
For questions please do not hesitate to contact:
k.patil@fz-juelich.de and t.muganga@fz-juelich.de
