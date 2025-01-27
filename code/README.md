## Data
This project uses data from the Human connectome project (HCP) accessed through DataLad.
A data use agreement with the HCP is needed to obtain AWS credentials used by DataLad to access the dataset.

Further instructions regarding AWS credentials for DataLad can be found in the DataLad handbook: 
https://handbook.datalad.org/en/latest/usecases/HCP_dataset.html#usecase-hcp-dataset

This project uses confound files with precomputed confounds for `preprocessing` as well as subject related information (e.g. cognitive scores and age) for prediction analysis.
These can be shared upon request with proof of agreement to the HCP's data use agreement (https://db.humanconnectome.org/).

## Code 
The code utilized in this study included `preprocessing` using the `junifer` toolbox performing `identification`, `typicality` as well as `prediction` analyses. For connectome-based prediction analysis the `julearn` toolbox was used.

To reproduce the rs-FC used in this project please start by creating the FC with the instructions in the `preprocessing` directory.

Next, analyses can be performed on the created FCs using the respective scripts in `identification`, `typicallity` and `prediction` directory.

Finally, figures can be recreated using the scripts in the `figures` directory.

## Contact
For questions please do not hesitate to contact:
k.patil@fz-juelich.de and t.muganga@fz-juelich.de
