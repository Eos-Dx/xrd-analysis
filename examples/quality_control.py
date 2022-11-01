"""
Apply quality control based on criteria
"""
import argparse
import json

import numpy as np
import pandas as pd

def main(data_filepath, output_filepath, control_criteria, add_column=False):
    """
    Performs quality control based on criteria. Adds a ``qc_pass`` column
    if ``add_column=True``.

    Parameters
    ----------
    data_filepath : str
        Full path to the input csv file containing data for quality control.

    output_filepath : str
        Full path to the output csv file.

    control_criteria : dict
        Dictionary for specifying quality control parameters and ranges.

    add_column : bool
        Adds a ``qc_pass`` column if ``True``.

    Notes
    -----
    Specify quality control ranges for individual parameters as follows:

        control_criteria = {
            "parameter1": [lower_bound1, upper bound1],
            "parameter2": [lower_bound2, upper bound2]
        }

    """

    # Load the 
    df = pd.read_csv(data_filepath)

    # Print statistics
    dataset_size = df.index.size

    columns = df.columns

    # Create the exclusion column set to zeros (no passes yet)
    df["qc_pass"] = np.ones(df.shape[0]).reshape(-1,1).astype(int)

    # Extract the measurement series info (barcode letter)
    measurement_series = df["Filename"].str.extract(r"CR_(.)", expand=False)

    # Calculate the training and blind totals
    training_total = (measurement_series == "A").sum()
    blind_total = (measurement_series == "B").sum()

    if "Cancer" in columns:
        # Total cancer and normal statistics
        cancer_total = (df["Cancer"] == 1).sum()
        normal_total = (df["Cancer"] == 0).sum()

    for key, value in control_criteria.items():
        # Set meaningful variables
        control_parameter = key
        lower_bound = value[0]
        upper_bound = value[1]

        # Report current criterion and bounds
        print("Control criterion parameter: {}".format(control_parameter))
        print("Lower bound: {}".format(lower_bound))
        print("Upper bound: {}".format(upper_bound))

        # Perform quality control using the current criterion
        control_series = ((df[control_parameter] >= lower_bound) & \
                (df[control_parameter] <= upper_bound)).astype(int)
        # Merge current criterion results with prior results
        df["qc_pass"] = df["qc_pass"] & control_series
        criterion_column = "criterion_{}".format(control_parameter)
        df[criterion_column] = control_series

        if add_column:
            df.to_csv(output_filepath, index=False)
        else:
            df[control_series.astype(bool)]["Filename"].to_csv(
                    output_filepath, index=False)

        # Calculate parameter control statistics
        parameter_pass_count = control_series.sum()
        parameter_pass_ratio = parameter_pass_count/dataset_size
        parameter_fail_count = dataset_size - parameter_pass_count
        parameter_fail_ratio = parameter_fail_count/dataset_size

        # Report parameter control statistics
        print("Pass: {}".format(parameter_pass_count))
        print("Fail: {}".format(parameter_fail_count))
        print("Pass %: {:.1f}%".format(
            100*parameter_pass_ratio))
        print("Fail %: {:.1f}%".format(
            100*parameter_fail_ratio))

        # Calculate normal vs. cancer parameter control statistics
        if "Cancer" in columns:
            # Calculate number of control passes
            normal_pass_count = (
                    (df["Cancer"] == 0) & (control_series == 1)).sum()
            cancer_pass_count = (
                    (df["Cancer"] == 1) & (control_series == 1)).sum()

            # Report control passes
            print("Normal pass: {}".format(normal_pass_count))
            print("Cancer pass: {}".format(cancer_pass_count))

            # Calculate number of control failures
            normal_fail_count = (
                    (df["Cancer"] == 0) & (control_series == 0)).sum()
            cancer_fail_count = (
                    (df["Cancer"] == 1) & (control_series == 0)).sum()

            # Report control failures
            print("Normal fail: {}".format(normal_fail_count))
            print("Cancer fail: {}".format(cancer_fail_count))

            # Calculate ratios
            normal_pass_ratio = normal_pass_count/normal_total
            cancer_pass_ratio = cancer_pass_count/cancer_total
            normal_fail_ratio = normal_fail_count/normal_total
            cancer_fail_ratio = cancer_fail_count/cancer_total

            # Report ratios
            print("Normal pass (%): {:.1f}%".format(
                100*normal_pass_ratio))
            print("Cancer pass (%): {:.1f}%".format(
                100*cancer_pass_ratio))
            print("Normal fail (%): {:.1f}%".format(
                100*normal_fail_ratio))
            print("Cancer fail (%): {:.1f}%".format(
                100*cancer_fail_ratio))

            # Calculate training pass/fail statistics
            training_pass_count = normal_pass_count + cancer_pass_count
            training_pass_ratio = training_pass_count / training_total
            training_fail_count = normal_fail_count + cancer_fail_count
            training_fail_ratio = training_fail_count / training_total
            print("Training pass (%): {:.1f}%".format(
                100*training_pass_ratio))
            print("Training fail (%): {:.1f}%".format(
                100*training_fail_ratio))

        # Calculate blind pass/fail statistics
        # Note: blind data has no "Cancer" value 
        blind_pass_count = (
                (df["Cancer"].isnull()) & (control_series == 1)).sum()
        blind_fail_count = (
                (df["Cancer"].isnull()) & (control_series == 0)).sum()
        blind_pass_ratio = blind_pass_count / blind_total
        blind_fail_ratio = blind_fail_count / blind_total

        print("Blind pass: {}".format(blind_pass_count))
        print("Blind fail: {}".format(blind_fail_count))
        print("Blind pass (%): {:.1f}%".format(100*blind_pass_ratio))
        print("Blind fail (%): {:.1f}%".format(100*blind_fail_ratio))

    print("##########################")
    print("Total data set statistics")
    print("##########################")

    # Total data set size
    print("Input data size:", dataset_size)

    # Calculate total data set statistics
    pass_total = df["qc_pass"].sum()
    print("Total pass: {}".format(pass_total))
    pass_total_ratio = pass_total / dataset_size
    print("Total pass (%): {:.1f}%".format(
        100*pass_total_ratio))

    fail_total = dataset_size - pass_total
    print("Total fail: {}".format(fail_total))
    fail_total_ratio = fail_total / dataset_size
    print("Total fail (%): {:.1f}%".format(
        100*fail_total_ratio))


    # Calculate training (A) vs. blind (B) statistics

    print("Training total: {}".format(training_total))
    print("Blind total: {}".format(blind_total))

    # Passed
    training_pass_count = (
        (measurement_series == "A") & (df["qc_pass"] == 1)).sum()
    blind_pass_count = (
        (measurement_series == "B") & (df["qc_pass"] == 1)).sum()

    training_pass_ratio = training_pass_count / training_total
    blind_pass_ratio = blind_pass_count / blind_total

    print("Training pass: {}".format(training_pass_count))
    print("Blind pass: {}".format(blind_pass_count))

    print("Training pass (%): {:.1f}%".format(
        100*training_pass_ratio))
    print("Blind pass (%): {:.1f}%".format(
        100*blind_pass_ratio))

    # Failed
    training_fail_count = (
        (measurement_series == "A") & (df["qc_pass"] == 0)).sum()
    blind_fail_count = (
        (measurement_series == "B") & (df["qc_pass"] == 0)).sum()

    training_fail_ratio = training_fail_count / training_total
    blind_fail_ratio = blind_fail_count / blind_total

    print("Training fail: {}".format(training_fail_count))
    print("Blind fail: {}".format(blind_fail_count))

    print("Training fail (%): {:.1f}%".format(
        100*training_fail_ratio))
    print("Blind fail (%): {:.1f}%".format(
        100*blind_fail_ratio))

    if "Cancer" in columns:
        # Total normal and cancer statistics
        print("Normal total: {}".format(normal_total))
        print("Cancer total: {}".format(cancer_total))

        # Passed
        normal_total_pass_count = (
                (df["Cancer"] == 0) & (df["qc_pass"] == 1)).sum()
        cancer_total_pass_count = (
                (df["Cancer"] == 1) & (df["qc_pass"] == 1)).sum()
        normal_total_pass_ratio = normal_total_pass_count/normal_total
        cancer_total_pass_ratio = cancer_total_pass_count/cancer_total
        # Failed
        normal_total_fail_count = (
                (df["Cancer"] == 0) & (df["qc_pass"] == 0)).sum()
        cancer_total_fail_count = (
                (df["Cancer"] == 1) & (df["qc_pass"] == 0)).sum()
        normal_total_fail_ratio = normal_total_fail_count/normal_total
        cancer_total_fail_ratio = cancer_total_fail_count/cancer_total

        # Passed
        print("Total normal pass: {}".format(normal_total_pass_count))
        print("Total cancer pass: {}".format(cancer_total_pass_count))
        print("Total normal pass (%): {:.1f}".format(
            100*normal_total_pass_ratio))
        print("Total cancer pass (%): {:.1f}".format(
            100*cancer_total_pass_ratio))
        # Failed
        print("Total normal fail: {}".format(normal_total_fail_count))
        print("Total cancer fail: {}".format(cancer_total_fail_count))
        print("Total normal fail (%): {:.1f}%".format(
            100*normal_total_fail_ratio))
        print("Total cancer fail (%): {:.1f}%".format(
            100*cancer_total_fail_ratio))


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
            help="A file containing JSON-formatted quality control criteria.")
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
        raise ValueError("Control criteria file required.")

    main(data_filepath, output_filepath, criteria, add_column)
