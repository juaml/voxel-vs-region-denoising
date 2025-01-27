
from itertools import product

PREAMBLE = """# The environment
executable = /usr/bin/bash
transfer_executable = False
initial_dir=../exe
universe = vanilla
getenv = True

# Resources
request_cpus    = 1
request_memory  = 2500

"""

job = """arguments=./run_in_venv_conda.sh 01_prediction.py {aggregation} {level} {data_type} {regions} {target} {model}

log = ../submit/logs/{data_type}/{regions}/{target}_{aggregation}_{level}_{data_type}_{model}_$(Cluster).$(Process).log
output = ../submit/logs/{data_type}/{regions}/{target}_{aggregation}_{level}_{data_type}_{model}_$(Cluster).$(Process).out
error = ../submit/logs/{data_type}/{regions}/{target}_{aggregation}_{level}_{data_type}_{model}_$(Cluster).$(Process).err

queue

"""

targets = [
    "Age_in_Yrs",
    "PMAT24_A_CR",
    "ReadEng_Unadj",
    "ListSort_Unadj",
]
aggregations = ["mean", "eigen"]
confound_levels = ["VOXEL", "ROI"]
data_qualities = ["MINIMAL", "FIX"]
granularities = ["100", "400", "1000"]
models = ["kernelridge"]

for granularity in granularities:
    with open(f"submit_prediction_{granularity}.submit", "w") as f:
        f.write(PREAMBLE)

        i = 0
        for target, agg, level, data_quality, model in product(
            targets,
            aggregations,
            confound_levels,
            data_qualities,
            models
        ):

            f.write(
                job.format(
                    aggregation=agg,
                    level=level,
                    data_type=data_quality,
                    regions=granularities,
                    target=target,
                    model=model
                )
            )
            i += 1

    print(f"{i} jobs queued for {granularity} regions!")
