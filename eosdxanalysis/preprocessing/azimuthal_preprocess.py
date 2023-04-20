"""
Example with polar grid
"""
import os
import glob
import re

import numpy as np
import pandas as pd

from datetime import datetime

import argparse
import json
import glob

import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from eosdxanalysis.preprocessing.utils import azimuthal_integration
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import find_center

from eosdxanalysis.preprocessing.image_processing import pad_image

from eosdxanalysis.calibration.utils import radial_profile_unit_conversion


def run_azimuthal_preprocessing(
        input_path, output_path, input_path_list=None,
        sample_distance_filepath=None,
        input_dataframe_filepath=None,
        find_sample_distance_filepath=False,
        beam_rmax=15, visualize=False):
    """
    """
    sample_distance_m = None

    if input_path:
        # Given single input path
        # Get filepath list
        filepath_list = glob.glob(os.path.join(input_path, "*.txt"))
        # Sort files list
        filepath_list.sort()
    elif input_dataframe_filepath:
        input_df = pd.read_csv(input_dataframe_filepath, index_col="Filepath")
        filepath_list = input_df.index.tolist()

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Store output directory info
    # Create output directory if it does not exist
    if not output_path:
        # Create preprocessing results directory
        parent_path = os.path.dirname(input_path)
        results_dir = "preprocessed_results"
        results_path = os.path.join(parent_path, results_dir)

        # Create timestamped output directory
        input_dir = os.path.basename(input_path)
        output_dir = "preprocessed_{}_{}".format(
                input_dir,
                timestamp)
        output_path = os.path.join(results_path, output_dir)

    if find_sample_distance_filepath:
        # Given a single sample distance file path
        # Try to locate the sample distance filepath:
        parent_path = os.path.dirname(input_path)
        calibration_results_dir = "calibration_results"
        sample_distance_path = os.path.join(
                parent_path, calibration_results_dir)

        # Find files
        calibration_results_filepath_list = glob.glob(os.path.join(sample_distance_path, "*.json"))
        if len(calibration_results_filepath_list) != 1:
            raise ValueError("Only a single calibration file is allowed.")
        else:
            sample_distance_filepath = calibration_results_filepath_list[0]

    # Data output path
    data_output_dir = "data"
    data_output_path = os.path.join(output_path, data_output_dir)
    os.makedirs(data_output_path, exist_ok=True)

    # Image output path
    image_output_dir = "images"
    image_output_path = os.path.join(output_path, image_output_dir)
    os.makedirs(image_output_path, exist_ok=True)

    # Loop over files list
    for filepath in filepath_list:
        filename = os.path.basename(filepath)
        image = np.loadtxt(filepath, dtype=np.float64)

        # Find the center
        center = find_center(image)

        # Block out the beam
        beam_mask = create_circular_mask(
                image.shape[0], image.shape[1], center=center, rmax=beam_rmax)

        masked_image = image.copy()
        masked_image[beam_mask] = np.nan

        # Enlarge the image
        padding_amount = (np.sqrt(2)*np.max(image.shape)).astype(int)
        padding_top = padding_amount
        padding_bottom = padding_amount
        padding_left = padding_amount
        padding_right = padding_amount
        padding = (padding_top, padding_bottom, padding_left, padding_right)
        enlarged_masked_image = pad_image(
                masked_image, padding=padding, nan=True)

        new_center = (padding_top + center[0], padding_left + center[1])

        radial_profile = azimuthal_integration(
                enlarged_masked_image, center=new_center, radius=padding_amount)

        # Save data to file
        # Get approximate sample distance from folder name
        sample_distance_approx_list = re.findall(r"dist_[0-9]{2,3}mm", filepath)
        if len(sample_distance_approx_list) != 1:
            raise ValueError("Unable to find the approximate sample distance from folder name.")
        else:
            sample_distance_approx = sample_distance_approx_list[0]
        data_output_filename = "radial_{}_{}".format(sample_distance_approx, filename)
        data_output_filepath = os.path.join(data_output_path,
                data_output_filename)

        if sample_distance_filepath:
            with open(sample_distance_filepath, "r") as distance_fp:
                sample_distance_results = json.loads(distance_fp.read())
                sample_distance_m = sample_distance_results["sample_distance_m"]

            # Combine radial profile intensity and q-units
            q_range = radial_profile_unit_conversion(
                    radial_profile.size,
                    sample_distance_m,
                    radial_units="q_per_nm")
            results = np.hstack([q_range.reshape(-1,1), radial_profile.reshape(-1,1)])
            np.savetxt(data_output_filepath, results)
        elif input_dataframe_filepath:
            sample_distance_m = input_df.loc[filepath, "Sample_Distance_m"]

            # Combine radial profile intensity and q-units
            q_range = radial_profile_unit_conversion(
                    radial_profile.size,
                    sample_distance_m,
                    radial_units="q_per_nm")
            results = np.hstack([q_range.reshape(-1,1), radial_profile.reshape(-1,1)])
            np.savetxt(data_output_filepath, results)
        else:
            np.savetxt(data_output_filepath, radial_profile)

        # Save image preview to file
        plot_title = "Azimuthal Integration {}".format(filename)
        fig = plt.figure(plot_title)

        if sample_distance_m:
            plt.scatter(q_range, radial_profile)
            plt.xlabel(r"q $\mathrm{{nm}^{-1}}$")
        else:
            plt.scatter(range(radial_profile.size), radial_profile)
            plt.xlabel("Radius [pixel units]")
        plt.ylabel("Mean Intensity [photon count]")

        plt.title(plot_title)

        # Set image output file
        image_output_filename = "{}.png".format(data_output_filename)
        image_output_filepath = os.path.join(image_output_path,
                image_output_filename)

        # Save image preview to file
        plt.savefig(image_output_filepath)

        if visualize:
            plt.show()

        plt.close(fig)


if __name__ == '__main__':
    """
    Run azimuthal integration on an image
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", type=str, required=False,
            help="The path to data to extract features from")
    parser.add_argument(
            "--input_path_list", type=str, required=False,
            help="The path to data to extract features from")
    parser.add_argument(
            "--input_dataframe_filepath", type=str, required=False,
            help="The dataframe containing file paths to extract features from")
    parser.add_argument(
            "--output_path", type=str, default=None, required=False,
            help="The output path to save radial profiles and peak features")
    parser.add_argument(
            "--sample_distance_filepath", type=str, default=None, required=False,
            help="The path to calibrated sample distance.")
    parser.add_argument(
            "--find_sample_distance_filepath", action="store_true",
            help="Automatically try to find sample distance file.")
    parser.add_argument(
            "--beam_rmax", type=int, default=15, required=True,
            help="The maximum beam radius in pixel lengths.")
    parser.add_argument(
            "--visualize", action="store_true",
            help="Visualize plots.")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path) if args.input_path \
            else None
    if not input_path:
        input_path_list = args.input_path_list
    output_path = os.path.abspath(args.output_path) if args.output_path \
            else None
    input_dataframe_filepath = os.path.abspath(args.input_dataframe_filepath) \
            if args.input_dataframe_filepath else None
    find_sample_distance_filepath = args.find_sample_distance_filepath
    beam_rmax = args.beam_rmax
    sample_distance_filepath = args.sample_distance_filepath
    visualize = args.visualize

    run_azimuthal_preprocessing(
        input_path=input_path,
        output_path=output_path,
        input_path_list=input_path_list,
        input_dataframe_filepath=input_dataframe_filepath,
        beam_rmax=beam_rmax,
        find_sample_distance_filepath=find_sample_distance_filepath,
        visualize=visualize,
        )
