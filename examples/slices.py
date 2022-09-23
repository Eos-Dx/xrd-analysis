"""
Code to analyze horizontal or vertical slices of measurements
"""
import os
import glob
import argparse
from datetime import datetime
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter

def plot_slices(input_path=None, db_filepath=None, output_path=None, width=10):
    """
    Preprocessing is 2 x uniform 3x3 filter across CRQF image,
    then width=10 averaged.

    Parameters
    ----------
    input_path : str
        Path to CRQF files to analyze.

    db_filepath : str
        Full path to csv file with ``Filename`` and ``Patient`` columns.

    output_path : str
        Path to create output directory in.

    width : int
        With of slice to average over, must be even integer.

    """
    # Set timestamp
    timestr = "%y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Set output path with a timestamp
    output_dir = "meridional_slice_plots_{}".format(timestamp)
    output_subpath = os.path.join(output_path, output_dir)

    os.makedirs(output_subpath)

    # Load measurements database
    df = pd.read_csv(db_filepath)

    # Plot patient slices
    _plot_patient_slices(
            df, input_path, output_subpath, patient_key="Patient")
    # Plot patient code slices
    _plot_patient_slices(
            df, input_path, output_subpath, patient_key="Patient_Code")

def _plot_patient_slices(df, input_path, output_subpath, patient_key="Patient"):
    """
    Plot patient slices
    """

    # Extract list of patients
    patient_list = df[patient_key].dropna().unique()

    # Set file suffix
    file_suffix = ".txt"

    # Get list of files
    file_path_list = glob.glob(os.path.join(input_path, "*" + file_suffix))

    # Get the file names
    basename_list = [os.path.basename(file_path) for file_path in file_path_list]

    # Get barcode list
    active_barcode_list = [
            re.sub(r"CRQF_","", os.path.splitext(bname)[0]) for bname in \
                    basename_list]

    # Get the list of patients we have files for
    patient_list = df[df["Barcode"].isin(
        active_barcode_list)][patient_key].dropna().unique().astype(str)

    # Create a dataframe subset of active barcodes only
    df_active = df[df["Barcode"].isin(active_barcode_list)].astype(str)

    # Plot all slices per patient onto a single graph
    for patient in patient_list:
        fig = plt.figure(patient)
        fig.suptitle(patient)

        # Get barcodes for this patient
        patient_active_barcode_list = \
                df_active[df_active[patient_key] == patient]["Barcode"].astype(
                        str)

        # For each barcode, load the corresponding measurement data
        for barcode in patient_active_barcode_list:
            # Construct the file path based on the barcode
            file_prefix_pattern = "CRQF_"
            file_name = file_prefix_pattern + barcode + file_suffix
            file_path = os.path.join(input_path, file_name)

            # Load the file
            data = np.loadtxt(file_path, dtype=np.uint32)
            # Calculate the center based on shape
            center = (data.shape[0]/2-0.5, data.shape[1]/2-0.5)

            # Perform filtering twice in a row
            filtered_data = uniform_filter(
                    uniform_filter(data, size=3), size=3)

            # Take a vertical slice
            row_start = 0
            row_end = int(center[0]+0.5)
            col_start = int(center[1]+0.5-width/2)
            col_end = int(center[1]+0.5+width/2)
            data_slice = filtered_data[row_start:row_end, col_start:col_end]

            # Average the slice to get 1D curve
            # slice_1d = np.mean(data_slice, axis=1)[::-1]
            slice_1d = np.mean(data_slice, axis=1)

            plt.plot(slice_1d, label=barcode)

        # Set output image path
        plot_suffix = ".png"
        output_filepath = os.path.join(
                output_subpath, str(patient) + plot_suffix)

        plt.legend()
        plt.xlabel("Distance from top [pixels]")
        # plt.show()
        fig.savefig(output_filepath)
        plt.close(fig)
        fig.clear()


if __name__ == '__main__':
    """
    Extract slices from measurements
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=True,
            help="The path containing preprocessed CRQF files.")
    parser.add_argument(
            "--db_filepath", default=None, required=True,
            help="The file path of the patient barcode csv database.")
    parser.add_argument(
            "--output_path", default=None, required=True,
            help="The output path for the patient curves plots.")
    parser.add_argument(
            "--width", default=10, required=False,
            help="The width of the slice to take.")

    # Collect arguments
    args = parser.parse_args()
    input_path = args.input_path
    db_filepath = args.db_filepath
    output_path = args.output_path
    width = args.width

    plot_slices(
            input_path=input_path, db_filepath=db_filepath,
            output_path=output_path, width=width)
