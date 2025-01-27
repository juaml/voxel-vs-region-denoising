"""
This script runs a prediction analysis based on connectome data and HCP 
phenotypes.

Functions:
    corr(y_true, y_pred): Computes the Pearson correlation coefficient between 
        true and predicted values.
    parse_args(): Parses command-line arguments.
    validate_args(args): Validates the parsed arguments.
    main(): Main function to run the prediction analysis.

Command-line Arguments:
    aggregation (str): Aggregation method, either 'mean' or 'eigen'.
    confound_level (str): Confound level, either 'ROI' or 'VOXEL'.
    preprocess (str): Preprocessing method, either 'MINIMAL' or 'FIX'.
    regions (str): Number of regions, either '100', '400', or '1000'.
    target (str): HCP variable to predict.
    model (str): Sklearn estimator defined in modelgrids.py.

Example Usage:
    python 01_prediction.py mean ROI MINIMAL 100 Age_in_Yrs kernelridge
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from julearn import run_cross_validation
from julearn.scoring import register_scorer
from julearn.utils import configure_logging
from scipy.stats import pearsonr, zscore
from sklearn.metrics import make_scorer

sys.path.append("../pckg")
from load_connectomes import load_connectomes
from load_phenotypes import load_hcp_phenotypes
from model_grids import get_model_grid

configure_logging(level="INFO")


# Make pearsonr scorer
def corr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


sklearn_scorer = make_scorer(corr)
register_scorer("corr", sklearn_scorer)


def parse_args():
    """Parse arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the prediction analysis for a given set of parameters."
        )
    )
    parser.add_argument("aggregation", choices=["mean", "eigen"],type=str, help="'mean' or 'eigen'")
    parser.add_argument("confound_level", choices=["ROI", "VOXEL"], type=str, help="'VOXEL' or 'ROI'")
    parser.add_argument("preprocess", choices=["MINIMAL", "FIX"], type=str, help= "'MINIMAL' or 'FIX'")
    parser.add_argument("regions", choices=["100", "400", "1000"], type=str, help= "'100', '400' or '1000'")
    parser.add_argument(
        "target", type=str, help="Which HCP variable to predict."
    )
    parser.add_argument(
        "model", type=str, help="Sklearn estimator defined in modelgrids.py"
    )
    return parser.parse_args()


def validate_args(args):
    """Validate arguments."""
    all_available_targets = [
        "Age_in_Yrs",
        "PMAT24_A_CR",
        "ReadEng_Unadj",
        "ListSort_Unadj"
    ]
    assert args.aggregation in ["mean", "eigen"], "Invalid 'aggregation'!"
    assert args.confound_level in ["VOXEL", "ROI"], "Invalid 'confound_level'!"
    assert args.target in all_available_targets, "Invalid 'target'!"
    assert args.model in ["kernelridge"], "Invalid model!"
    assert args.preprocess in ["MINIMAL", "FIX"], "Invalid data quality!"
    assert args.regions in ["100", "400", "1000"], "Invalid granularity!"
    return args


def main():
    """Run the main analysis."""
    args = validate_args(parse_args())

    outfile = (
        f"aggr-{args.aggregation}_conflevel-{args.confound_level}"
        f"_target-{args.target}_model-{args.model}"
    )
    outdir_model = Path("..") / ".." / ".." / "results" / "prediction" / "models" / f"{args.preprocess}" / f"{args.regions}_regions"
    outdir_scores = Path("..") / ".." / ".." / "results" / "prediction" / "scores" / f"{args.preprocess}" / f"{args.regions}_regions"
    assert outdir_model.exists(), f"{outdir_model} does not exist!"
    assert outdir_scores.exists(), f"{outdir_scores} does not exist!"

    outfile_model = outdir_model / f"{outfile}.sav"
    outfile_scores = outdir_scores / f"{outfile}.csv"

    print(f"{outfile_model=}")
    print(f"{outfile_scores=}")
    connectomes = load_connectomes(
        args.preprocess, args.regions, args.aggregation, args.confound_level
    )
    assert connectomes.shape in [(370, 79800), (370, 4950), (370, 499500)], (
        "Shape of connectome dataframe not as expected!"
    )
    features = list(connectomes.columns)

    target_data = load_hcp_phenotypes(
        subjects=list(connectomes.index), columns=args.target
    )[args.target]

    # load target and configure pipeline based on target
    if args.target == "Age_in_Yrs":
        confounds = ["FD_REST1_REST2"]
    else:
        confounds = ["FD_REST1_REST2", "Age_in_Yrs"]
    problem_type = "regression"
    preprocess_X = None
    preprocess_y = "remove_confound"

    scoring = [
        "corr",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
        "r2",
    ]

    confound_data = load_hcp_phenotypes(
        subjects=list(connectomes.index), columns=confounds
    )

    assert confound_data.shape == (
        370,
        len(confounds),
    ), "Confound shape not correct!"

    # for kernel ridge regression we zscore rows, to turn
    # linear kernel into a pearson kernel
    if args.model == "kernelridge":
        connectomes = connectomes.apply(lambda V: zscore(V), axis=1)

    model, m_params = get_model_grid(
        model_name=args.model, problem_type=problem_type
    )

    m_params["remove_confound"] = "passthrough"

    prediction_data = pd.concat([connectomes, confound_data], axis=1)
    prediction_data[args.target] = target_data

    assert prediction_data.shape in [
        (370, 79800 + len(confounds) + 1),
        (370, 4950 + len(confounds) + 1),
        (370, 499500 + len(confounds) + 1)
    ], "Shape of prediction data not as expected!"

    assert prediction_data.isna().sum().sum() == 0, (
        "There are some NaN values in 'prediction_data'!" "Take care of it."
    )

    print(f"{confounds=}")
    print(f"{args.target=}")
    print(f"{args.model=}")
    print(f"{problem_type=}")
    print(f"{preprocess_X=}")
    print(f"{preprocess_y=}")
    print(f"{scoring=}")

    print("Starting to run julearn...")
    scores, final_estimator = run_cross_validation(
        X=features,
        y=args.target,
        data=prediction_data,
        confounds=confounds,
        model=model,
        problem_type=problem_type,
        preprocess_X=preprocess_X,
        preprocess_y=preprocess_y,
        return_estimator="final",
        scoring=scoring,
        model_params=m_params,
        seed=500,
    )
    scores.to_csv(outfile_scores, index=False)
    joblib.dump(final_estimator, outfile_model)


if __name__ == "__main__":
    main()
