"""
Code to analyze horizontal or vertical slices of measurements
"""
import os
import glob
import argparse
from datetime import datetime

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
            df, output_subpath, patient_key="Patient")
    # Plot patient code slices
    _plot_patient_slices(
            df, output_subpath, patient_key="Patient_Code")

def _plot_patient_slices(df, output_subpath, patient_key="Patient"):
    """
    Plot patient slices
    """

    # Extract list of patients
    patient_list = np.sort(np.unique(df[patient_key].values))

    # Set file suffix
    file_suffix = ".txt"

    # Plot all slices per patient onto a single graph
    for patient in patient_list:
        fig = plt.figure(patient)
        fig.suptitle(patient)

        # Get list of barcodes
        barcode_list = df[df["Patient"] == patient]["Barcode"].values

        # For each barcode, load the corresponding measurement data
        for barcode in barcode_list:
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
        output_filepath = os.path.join(output_subpath, str(patient))

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
