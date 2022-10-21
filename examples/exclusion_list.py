"""
Creates an exclusion list based on criteria
"""
import argparse
import json

import numpy as np
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

    columns = df.columns

    # Create the exclusion column set to zeros (no exclusions yet)
    df["Exclude"] = np.zeros(df.shape[0]).reshape(-1,1).astype(int)

    if "Cancer" in columns:
        # Total cancer and normal statistics
        cancer_total = (df["Cancer"] == 1).sum()
        normal_total = (df["Cancer"] == 0).sum()

    for key, value in exclusion_criteria.items():
        exclusion_parameter = key
        lower_bound = value[0]
        upper_bound = value[1]

        print("Exclusion criterion parameter: {}".format(exclusion_parameter))
        print("Lower bound: {}".format(lower_bound))
        print("Upper bound: {}".format(upper_bound))

        exclusion_series = ((df[exclusion_parameter] < lower_bound) | \
                (df[exclusion_parameter] > upper_bound)).astype(int)
        df["Exclude"] = df["Exclude"] | exclusion_series

        if add_column:
            df.to_csv(output_filepath, index=False)
        else:
            df[exclusion_series.astype(bool)]["Filename"].to_csv(
                    output_filepath, index=False)

        # Calculate parameter exclusion statistics
        parameter_exclusion_total = exclusion_series.sum()
        parameter_exclusion_ratio = parameter_exclusion_total/sample_size
        print("Exclude: {}".format(parameter_exclusion_total))
        print("Exclusion percentage: {:.1f}%".format(
            100*parameter_exclusion_ratio))

        # Calculate normal vs. cancer parameter exclusion statistics
        if "Cancer" in columns:
            # Cancer (=1) and is excluded (=1)
            cancer_excluded = ((df["Cancer"] == 1) & (exclusion_series == 1)).sum()
            # Normal (=0) and is excluded (=1)
            normal_excluded = ((df["Cancer"] == 0) & (exclusion_series == 1)).sum()

            print("Cancer excluded: {}".format(cancer_excluded))
            print("Normal excluded: {}".format(normal_excluded))

            cancer_excluded_ratio = cancer_excluded/cancer_total
            normal_excluded_ratio = normal_excluded/normal_total
            print("Cancer exclusion percentage: {:.1f}%".format(
                100*cancer_excluded_ratio))
            print("Normal exclusion percentage: {:.1f}%".format(
                100*normal_excluded_ratio))

    print("##########################")
    print("Total exclusion statistics")
    print("##########################")

    # Total data set size
    print("Input data size:", sample_size)

    # Calculate total data set statistics
    exclusion_total = df["Exclude"].sum()
    print("Total excluded: {}".format(exclusion_total))
    exclusion_total_ratio = exclusion_total / sample_size
    print("Total exclusion percentage: {:.1f}%".format(
        100*exclusion_total_ratio))

    remaining_total = sample_size - exclusion_total
    print("Total remaining: {}".format(remaining_total))
    remaining_total_ratio = remaining_total / sample_size
    print("Total remaining percentage: {:.1f}%".format(
        100*remaining_total_ratio))

    # Calculate training (A) vs. blind (B) statistics
    measurement_series = df["Filename"].str.extract(r"CR_(.)", expand=False)

    training_total = (measurement_series == "A").sum()
    blind_total = (measurement_series == "B").sum()

    # Excluded (exclude=1)
    training_excluded = ((measurement_series == "A") & (df["Exclude"] == 1)).sum()
    blind_excluded = ((measurement_series == "B") & (df["Exclude"] == 1)).sum()

    training_excluded_ratio = training_excluded/training_total
    blind_excluded_ratio = blind_excluded/blind_total

    # Reamining (exclude=0)
    training_remaining = ((measurement_series == "A") & (df["Exclude"] == 0)).sum()
    blind_remaining = ((measurement_series == "B") & (df["Exclude"] == 0)).sum()

    print("Training excluded: {}".format(training_excluded))
    print("Blind excluded: {}".format(blind_excluded))

    print("Training exclusion percentage: {:.1f}%".format(
        100*training_excluded_ratio))
    print("Blind exclusion percentage: {:.1f}%".format(
        100*blind_excluded_ratio))

    if "Cancer" in columns:
        # Total cancer and normal statistics
        print("Cancer total: {}".format(cancer_total))
        print("Normal total: {}".format(normal_total))

        # Normal vs. cancer
        # Cancer (=1) and is excluded (=1)
        cancer_total_excluded = ((df["Cancer"] == 1) & (df["Exclude"] == 1)).sum()
        # Normal (=0) and is excluded (=1)
        normal_total_excluded = ((df["Cancer"] == 0) & (df["Exclude"] == 1)).sum()

        print("Total cancer excluded: {}".format(cancer_total_excluded))
        print("Total normal excluded: {}".format(normal_total_excluded))

        cancer_total_excluded_ratio = cancer_total_excluded/cancer_total
        normal_total_excluded_ratio = normal_total_excluded/normal_total
        print("Total cancer exclusion percentage: {:.1f}%".format(
            100*cancer_total_excluded_ratio))
        print("Total normal exclusion percentage: {:.1f}%".format(
            100*normal_total_excluded_ratio))


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
