"""
Creates an exclusion list based on criteria
"""
import argparse
import json

import pandas as pd

def main(data_filepath, output_filepath, exclusion_criteria, add_column=False):
    """
    Outputs exclusion list based on exclusion criteria. Adds an exclusion column
    if ``add_column=True``.

    Parameters
    ----------
    data_filepath : str

    output_filepath : str

    exclusion_criteria : dict

    add_column : bool

    Notes
    -----
    Only one exclusion criterion can be handled.

        exclusion_criteria = {
            "field1": [lower_bound, upper bound]
        }


    """

    df = pd.read_csv(data_filepath)
    # Print statistics
    sample_size = df.index.size
    print("Input data size:", sample_size)

    cancer_total = (df["Cancer"] == 1).sum()
    normal_total = (df["Cancer"] == 0).sum()

    print("Cancer total:", cancer_total)
    print("Normal total:", normal_total)

    for key, value in exclusion_criteria.items():

        print("Exclusion criteria:",key)
        print("Lower bound:",value[0])
        print("Upper bound:",value[1])

        exclusion_series = ((df[key] < value[0]) | (df[key] > value[1])).astype(int)
        df["Exclude"] = exclusion_series

        if add_column:
            df.to_csv(output_filepath, index=False)
        else:
            df[exclusion_series.astype(bool)]["Filename"].to_csv(
                    output_filepath, index=False)

        # Print statistics
        exclusion_total = exclusion_series.sum()
        print("Exclude:", exclusion_total)
        print("Exclusion ratio:", exclusion_total/sample_size)

        cancer_excluded = ((df["Cancer"] == 1) & (df["Exclude"] == 1)).sum()
        normal_excluded = ((df["Cancer"] == 0) & (df["Exclude"] == 1)).sum()

        print("Cancer excluded:", cancer_excluded)
        print("Normal excluded:", normal_excluded)

        cancer_excluded_ratio = cancer_excluded/cancer_total
        normal_excluded_ratio = normal_excluded/normal_total
        print("Cancer exclusion ratio:", cancer_excluded_ratio)
        print("Normal exclusion ratio:", normal_excluded_ratio)


if __name__ == '__main__':
    """
    Generate an exclusion list from 2D Gaussian fit results
    based on exclusion criteria
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--data_filepath", default=None, required=True,
            help="The file path containing 2D Gaussian fit results.")
    parser.add_argument(
            "--output_filepath", default=None, required=True,
            help="The output file path to store the exclusion list.")
    parser.add_argument(
            "--criteria_file", default=None, required=True,
            help="A file containing JSON-formatted exclusion criteria.")
    parser.add_argument(
            "--add_column", required=False, action="store_true",
            help="Adds a exclusion column to the data set.")

    args = parser.parse_args()

    data_filepath = args.data_filepath
    output_filepath = args.output_filepath

    add_column = args.add_column

    # Get exclusion criteria from file
    criteria_file = args.criteria_file
    if criteria_file:
        with open(criteria_file,"r") as criteria_fp:
            criteria = json.loads(criteria_fp.read())
    else:
        raise ValueError("Exclusion criteria file required.")

    main(data_filepath, output_filepath, criteria, add_column)
