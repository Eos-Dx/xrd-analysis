"""
Apply quality control based on criteria
"""
import os
import shutil

import argparse
import json

import numpy as np
import pandas as pd

def main(
        csv_input_filepath, csv_output_filepath, control_criteria,
        no_add_columns=False, no_qc_pass_fail_output_folders=False,
        preprocessed_data_path=None):
    """
    Performs quality control based on criteria. Adds a ``qc_pass`` column
    unless ``no_add_columns=True``. Copies data to pass/fail subfolders unless
    ``no_qc_pass_fail_output_folders=True``.

    Parameters
    ----------
    csv_input_filepath : str
        Full path to the input csv file containing data for quality control.

    csv_output_filepath : str
        Full path to the output csv file containing quality control data.

    control_criteria : dict
        Dictionary for specifying quality control parameters and ranges.

    no_add_columns : bool
        Does not add a ``qc_pass`` column if ``True``. Default is ``False``.

    no_qc_pass_fail_output_folders : bool
        Does not copy data to pass/fail subfolders if ``True``. Default is
        ``False``.

    Notes
    -----
    Specify quality control ranges for individual parameters as follows:

        control_criteria = {
            "parameter1": [lower_bound1, upper bound1],
            "parameter2": [lower_bound2, upper bound2]
        }

    """

    # Load the data
    df = pd.read_csv(csv_input_filepath)

    # Print statistics
    dataset_size = df.index.size

    if dataset_size == 0:
        raise ValueError("Dataset is empty!")

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

        # Save quality control data to csv file
        if no_add_columns:
            # Save the quality control data only
#            df[control_series.astype(bool)]["Filename"].to_csv(
#                    csv_output_filepath, index=False)
            df[control_series.astype(bool)].to_csv(
                    csv_output_filepath, index=False)
        else:
            # Save original data with quality control data
            df.to_csv(csv_output_filepath, index=False)

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

            # Calculate normal and cancer statistics
            if normal_total:
                normal_pass_ratio = normal_pass_count/normal_total
                normal_fail_ratio = normal_fail_count/normal_total

                print("Normal pass (%): {:.1f}%".format(
                    100*normal_pass_ratio))
                print("Normal fail (%): {:.1f}%".format(
                    100*normal_fail_ratio))

            if cancer_total:
                cancer_pass_ratio = cancer_pass_count/cancer_total
                cancer_fail_ratio = cancer_fail_count/cancer_total

                print("Cancer pass (%): {:.1f}%".format(
                    100*cancer_pass_ratio))
                print("Cancer fail (%): {:.1f}%".format(
                    100*cancer_fail_ratio))

        # Calculate training and blind statistics
        if training_total:
            training_pass_count = (
                    (measurement_series == "A") & (control_series == 1)).sum()
            training_fail_count = (
                    (measurement_series == "A") & (control_series == 0)).sum()
            training_pass_ratio = training_pass_count / training_total
            training_fail_ratio = training_fail_count / training_total
            print("Training pass (%): {:.1f}%".format(
                100*training_pass_ratio))
            print("Training fail (%): {:.1f}%".format(
                100*training_fail_ratio))

        if blind_total:
            blind_pass_count = (
                    (measurement_series == "B") & (control_series == 1)).sum()
            blind_fail_count = (
                    (measurement_series == "B") & (control_series == 0)).sum()
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

    if training_total or blind_total:

        print("Training total: {}".format(training_total))
        print("Blind total: {}".format(blind_total))

        # Passed
        training_pass_count = (
            (measurement_series == "A") & (df["qc_pass"] == 1)).sum()
        blind_pass_count = (
            (measurement_series == "B") & (df["qc_pass"] == 1)).sum()

        training_pass_ratio = training_pass_count / training_total if \
                training_total else 0
        blind_pass_ratio = blind_pass_count / blind_total if blind_total else 0

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

        training_fail_ratio = training_fail_count / training_total if \
                training_total else 0
        blind_fail_ratio = blind_fail_count / blind_total if blind_total else 0

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

        if normal_total:
            # Normal statistics
            normal_total_pass_count = (
                    (df["Cancer"] == 0) & (df["qc_pass"] == 1)).sum()
            normal_total_pass_ratio = normal_total_pass_count/normal_total

            normal_total_fail_count = (
                    (df["Cancer"] == 0) & (df["qc_pass"] == 0)).sum()
            normal_total_fail_ratio = normal_total_fail_count/normal_total

            print("Total normal pass: {}".format(normal_total_pass_count))
            print("Total normal pass (%): {:.1f}".format(
                100*normal_total_pass_ratio))

            print("Total normal fail: {}".format(normal_total_fail_count))
            print("Total normal fail (%): {:.1f}%".format(
                100*normal_total_fail_ratio))

        if cancer_total:
            # Cancer statistics
            cancer_total_pass_count = (
                    (df["Cancer"] == 1) & (df["qc_pass"] == 1)).sum()
            cancer_total_pass_ratio = cancer_total_pass_count/cancer_total

            cancer_total_fail_count = (
                    (df["Cancer"] == 1) & (df["qc_pass"] == 0)).sum()
            cancer_total_fail_ratio = cancer_total_fail_count/cancer_total

            print("Total cancer pass: {}".format(cancer_total_pass_count))
            print("Total cancer pass (%): {:.1f}".format(
                100*cancer_total_pass_ratio))
            # Failed
            print("Total cancer fail: {}".format(cancer_total_fail_count))
            print("Total cancer fail (%): {:.1f}%".format(
                100*cancer_total_fail_ratio))

    # Copy data and image previews to pass/fail folders
    if not no_qc_pass_fail_output_folders:
        # Check if preprocessed path was specified
        if not preprocessed_data_path:
            raise ValueError("Must specify preprocessed path to copy data.")

        # Set preprocessed images path
        preprocessed_images_path = preprocessed_data_path + "_images"

        # Get the parent folder of the output csv file
        output_path = os.path.dirname(csv_output_filepath)

        # Get the directory name of the input data
        preprocessed_data_dir = os.path.basename(preprocessed_data_path)
        preprocessed_images_dir = preprocessed_data_dir + "_images"

        # Construct output directory names
        data_pass_output_dir = "{}_qc_pass".format(preprocessed_data_dir)
        data_fail_output_dir = "{}_qc_fail".format(preprocessed_data_dir)
        image_pass_output_dir = "{}_qc_pass".format(
                preprocessed_images_dir)
        image_fail_output_dir = "{}_qc_fail".format(
                preprocessed_images_dir)

        # Construct output paths
        data_pass_output_path = os.path.join(
                output_path, data_pass_output_dir)
        data_fail_output_path = os.path.join(
                output_path, data_fail_output_dir)
        image_pass_output_path = os.path.join(
            output_path, image_pass_output_dir)
        image_fail_output_path = os.path.join(
            output_path, image_fail_output_dir)

        # Create the directories
        os.makedirs(data_pass_output_path, exist_ok=True)
        os.makedirs(data_fail_output_path, exist_ok=True)
        os.makedirs(image_pass_output_path, exist_ok=True)
        os.makedirs(image_fail_output_path, exist_ok=True)

        # Copy data into appropriate folders
        for idx in df.index:
            # Get data and preprocessed image filenames
            preprocessed_data_filename = df["Preprocessed_Filename"][idx]
            preprocessed_image_filename = df["Preprocessed_Filename"][idx] + \
                    ".png"

            # Get quality control pass/fail data
            qc_pass = df["qc_pass"][idx]

            # Get data and image filepaths
            data_filepath = os.path.join(preprocessed_data_path,
                    preprocessed_data_filename)
            image_filepath = os.path.join(preprocessed_images_path,
                    preprocessed_image_filename)

            # Copy to appropriate folder
            if qc_pass:
                # Copy data to qc pass data folder
                shutil.copy(data_filepath, data_pass_output_path)
                # Copy image to qc pass image folder
                shutil.copy(image_filepath, image_pass_output_path)
            else:
                # Copy data to qc fail data folder
                shutil.copy(data_filepath, data_fail_output_path)
                # Copy image to qc fail image folder
                shutil.copy(image_filepath, image_fail_output_path)


if __name__ == '__main__':
    """
    Generate an exclusion list from 2D Gaussian fit results
    based on exclusion criteria
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--csv_input_filepath", default=None, required=True,
            help="The csv input file containing data to perform quality"
                " control on.")
    parser.add_argument(
            "--csv_output_filepath", default=None, required=True,
            help="The csv output file path to store the quality control data.")
    parser.add_argument(
            "--criteria_file", default=None, required=True,
            help="A file containing JSON-formatted quality control criteria.")
    parser.add_argument(
            "--no_add_columns", required=False, action="store_true",
            help="Does not add quality control columns to the"
                " data set.")
    parser.add_argument(
            "--no_qc_pass_fail_output_folders", required=False,
            action="store_true",
            help="Does not copy data and image previews to pass/fail folders.")
    parser.add_argument(
            "--preprocessed_data_path", default=None, required=False,
            help="The csv input file containing data to perform quality"
                " control on.")

    args = parser.parse_args()

    csv_input_filepath = args.csv_input_filepath
    csv_output_filepath = args.csv_output_filepath

    no_add_columns = args.no_add_columns
    no_qc_pass_fail_output_folders = args.no_qc_pass_fail_output_folders
    preprocessed_data_path = args.preprocessed_data_path


    # Get exclusion criteria from file
    criteria_file = args.criteria_file
    if criteria_file:
        with open(criteria_file,"r") as criteria_fp:
            criteria = json.loads(criteria_fp.read())
    else:
        raise ValueError("Control criteria file required.")

    main(
            csv_input_filepath, csv_output_filepath, criteria, no_add_columns,
            no_qc_pass_fail_output_folders, preprocessed_data_path)
