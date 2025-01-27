## Setup
Install julearn. I installed julearn from github with the following commit ID
```3ee3879ef821dac18ff92447be0006d3e522f2af```.
So:
```
git clone https://github.com/juaml/julearn.git
cd julearn
git checkout -b install_julearn 3ee3879ef821dac18ff92447be0006d3e522f2af
pip install .
cd ..
```
The classification part of the project also depends on a KernelRidgeClassifier.
This can be installed as follows:
```
git clone https://github.com/LeSasse/kernelridgeclass.git
cd kernelridgeclass
pip install .
cd ..
```
Install julearn. I installed julearn from github with the following commit ID
```3ee3879ef821dac18ff92447be0006d3e522f2af```.
So:
```
git clone https://github.com/juaml/julearn.git
cd julearn
git checkout -b install_julearn 3ee3879ef821dac18ff92447be0006d3e522f2af
pip install .
cd ..
```
The classification part of the project also depends on a KernelRidgeClassifier.
This can be installed as follows:
```
git clone https://github.com/LeSasse/kernelridgeclass.git
cd kernelridgeclass
pip install .
cd ..
```

## Running Predictions

To run predictions using the provided scripts, follow these steps:

1. **Activate the Environment**

    Before running any scripts, make sure to install and activate a virtual environment that matches the specification in the provided `pip_freeze.txt`.

2. **Run the Prediction Script**

    To perform prediction analysis in a local session, use: 
    ```
    python 1-run_identification.py (aggregation method) (level for denoising) (data set) (parcellation) (target) (model)
    ``` 
    Example: 
    ```
    predict.py 01_prediction.py mean VOXEL MINIMAL 400 Age_in_Yrs kernelridge
    ```

    To run prediction anlysis for both datasets (FIX and MINIMAL) on a HTC cluster running HTCondor, use the provided `create_submit_prediction.py` file in `submit/` to generate submit files for each granularity (100, 400 and 1000). 
    Example:
    ```
    pyhon create_submit_prediction.py
    condor_submit submit_prediction_100.submit
    ```
