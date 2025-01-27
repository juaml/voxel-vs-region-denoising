import pandas as pd
from pathlib import Path
from itertools import product


def main():

    targets = [
        "Age_in_Yrs",
        "PMAT24_A_CR",
        "ReadEng_Unadj",
        "ListSort_Unadj"
    ]
    aggregations = ["mean", "eigen"]
    confound_levels = ["VOXEL", "ROI"]
    models = "kernelridge"
    data_types = ["MINIMAL", "FIX"]
    regions = ["100", "400", "1000"]

    path_scores = (
        Path("..") / ".." / ".." / "results" / "prediction" / "scores"
    )

    for data_type, region in product(data_types, regions):

        all_scores = []

        outpath = (
            path_scores 
            / data_type 
            / f"{region}_regions" 
            / f"all_behavior_{data_type}"
        )
        print(outpath)
        # start the main loop to collect scores across parameters
        for target, aggregation, conflevel in product(
            targets, aggregations, confound_levels
        ):
            # prepare file path
            file_name = (
                f"aggr-{aggregation}_conflevel-{conflevel}_"
                f"target-{target}_model-{models}.csv"
            )
            path_file = (
                path_scores 
                / data_type 
                / f"{region}_regions" 
                / file_name
            )

            # load data
            scores = pd.read_csv(path_file)

            # add info on parameters to df
            n_rows, _ = scores.shape

            names = ["target", "aggregation", "conflevel", "model"]
            params = [target, aggregation, conflevel, models]
            for name, param in zip(names, params):
                scores[name] = [param for _ in range(n_rows)]

            all_scores.append(scores)

        all_scores_df = (
            pd.concat(all_scores, axis=0)
            .reset_index(drop=True)
            .to_csv(f"{outpath}.csv", index=False)
        )

        print(all_scores_df)


if __name__ == "__main__":
    main()
