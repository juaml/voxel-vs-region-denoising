#!/usr/bin/bash
eval "$(conda shell.bash hook)"
conda activate VvR_predict
which python
python $@
conda deactivate