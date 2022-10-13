"""
Creates an exclusion list based on criteria
"""
import argparse
import json

import pandas as pd

def main(data_filepath, output_filepath, exclusion_criteria):
    """
    Outputs exclusion list based on exclusion criteria

    exclusion_criteria = {
        "field1": [lower_bound, upper bound],
        "field2": [lower_bound, upper bound],
        ...
    }


    """

    df = pd.read_csv(data_filepath)
    df_exclude = pd.DataFrame()

    for key, value in exclusion_criteria.items():
        exclusion_series = (df[key] < value[0]) | (df[key] > value[1])
        df[exclusion_series]["Filename"].to_csv(output_filepath, index=False)

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

    args = parser.parse_args()

    data_filepath = args.data_filepath
    output_filepath = args.output_filepath

    # Get exclusion criteria from file
    criteria_file = args.criteria_file
    if criteria_file:
        with open(criteria_file,"r") as criteria_fp:
            criteria = json.loads(criteria_fp.read())
    else:
        raise ValueError("Exclusion criteria file required.")

    main(data_filepath, output_filepath, criteria)
