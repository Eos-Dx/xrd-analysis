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
    df = pd.read_csv(db_filepath, dtype=str)

    # Plot patient slices
    _plot_patient_slices(
            df, input_path, output_subpath, patient_key="Patient")
    # Plot patient code slices
    _plot_patient_slices(
            df, input_path, output_subpath, patient_key="Patient_Code")

def _plot_patient_slices(df, input_path, output_subpath, patient_key="Patient"):
    """
    Plot patient slices

    Parameters
    ----------

    df : DataFrame

    input_path : str

    output_subpath : str

    patient_key : str

    """

    # Extract list of patients
    patient_list = df[patient_key].dropna().unique()

    # Set file suffix
    file_suffix = ".txt"

    # Get list of files
    file_path_list = glob.glob(os.path.join(input_path, "*" + file_suffix))

    # Get the file names
    basename_list = [os.path.basename(file_path) for file_path in file_path_list]

    # Get barcode list, remove any repeating As or Bs
    active_subbarcode_list = [
            re.sub(
                r"CRQF_([A-Z])[A-Z]*([0-9]+.*)", r"\1\2",
                os.path.splitext(bname)[0]) for bname in basename_list]

    active_barcode_list = [
            re.sub(
                r"([A-Z])[A-Z]*([0-9]+).*", r"\1\2",
                barcode) for barcode in active_subbarcode_list]

    file_lookup_dict = dict(zip(file_path_list, active_barcode_list))

    # Get the list of patients we have files for
    active_patient_list = df[df["Barcode"].isin(
        active_barcode_list)][patient_key].dropna().unique()

    # Create a dataframe subset of active barcodes only
    df_active = df[df["Barcode"].isin(active_barcode_list)]

    # Plot all slices per patient onto a single graph
    for patient in active_patient_list:
        fig = plt.figure(patient)
        fig_suptitle = "Meridional slices for {} {}".format(
                patient_key, patient)
        fig.suptitle(fig_suptitle)

        # Get barcodes for this patient
        patient_active_barcode_list = \
                df_active[df_active[patient_key] == patient]["Barcode"]

        # Get Files for this patient using ``file_lookup_dict``
        patient_file_path_list = []
        for barcode in patient_active_barcode_list:
            patient_file_path_matches = \
                    [file_path for file_path, active_barcode in \
                    file_lookup_dict.items() if active_barcode == barcode]
            patient_file_path_list += patient_file_path_matches

        # For each barcode, load the corresponding measurement data
        for file_path in patient_file_path_list:
            # Load the file
            data = np.loadtxt(file_path, dtype=np.uint32)
            # Calculate the center based on shape
            center = (data.shape[0]/2-0.5, data.shape[1]/2-0.5)

            # Rescale the data
            data_rescaled = data/data.sum()*data.size

            # Perform filtering twice in a row
            filtered_data = uniform_filter(
                    uniform_filter(data_rescaled, size=3), size=3)

            # Take a vertical slice
            row_start = 0
            row_end = int(center[0]+0.5)
            col_start = int(center[1]+0.5-width/2)
            col_end = int(center[1]+0.5+width/2)
            data_slice = filtered_data[row_start:row_end, col_start:col_end]

            # Average the slice to get 1D curve
            # slice_1d = np.mean(data_slice, axis=1)[::-1]
            slice_1d = np.mean(data_slice, axis=1)

            # Generate plot label
            basename = os.path.basename(file_path)
            file_prefix = os.path.splitext(basename)[0]
            plot_label = re.sub(r"CRQF_", r"", file_prefix)

            plt.plot(slice_1d, label=plot_label)

        # Set output image path
        plot_suffix = ".png"
        output_filepath = os.path.join(
                output_subpath, patient + plot_suffix)

        plt.legend()
        plt.xlabel("Distance from top [pixels]")
        plt.ylabel("Intensity [arbitrary units]")
        plt.ylim([0, 5])
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
