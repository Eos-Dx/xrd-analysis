"""
Example with polar grid
"""
import os

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


def run_feature_extraction(
        input_path, output_path, beam_rmax=15, sample_distance=None,
        visualize=False):
    """
    """

    # Get filepath list
    filepath_list = glob.glob(os.path.join(input_path, "*.txt"))
    # Sort files list
    filepath_list.sort()

    # Create dataframe to collect extracted features
    columns = ["Filename"] + feature_list
    df = pd.DataFrame(data={}, columns=columns, dtype=object)

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
        data_output_filename = "radial_{}".format(filename)
        data_output_filepath = os.path.join(data_output_path,
                data_output_filename)

        if sample_distance:
            # Combine radial profile intensity and q-units
            q_range = radial_profile_unit_conversion(
                    radial_profile.size,
                    sample_distance,
                    radial_units="q_per_nm")
            results = np.hstack([q_range.reshape(-1,1), radial_profile.reshape(-1,1)])
            np.savetxt(data_output_filepath, results)
        else:
            np.savetxt(data_output_filepath, radial_profile)

        # Save image preview to file
        plot_title = "Azimuthal Integration {}".format(filename)
        fig = plt.figure(plot_title)

        if sample_distance:
            plt.scatter(q_range, radial_profile)
            plt.xlabel(r"q $\mathrm{{nm}^{-1}}$")
        else:
            plt.scatter(range(radial_profile.size), radial_profile)
            plt.xlabel("Radius [pixel units]")
        plt.ylabel("Mean Intensity [photon count]")

        plt.title(plot_title)

        # Set image output file
        image_output_filename = "radial_{}".format(filename) + ".png"
        image_output_filepath = os.path.join(image_output_path,
                image_output_filename)

        # Save image preview to file
        plt.savefig(image_output_filepath)

        # Add extracted features to dataframe
        # df.loc[len(df.index)+1] = [filename] + [radial_profile]

        if visualize:
            plt.show()

        plt.close(fig)

    # Save dataframe to csv
    # df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    """
    Run azimuthal integration on an image
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", type=str, required=True,
            help="The path to data to extract features from")
    parser.add_argument(
            "--output_path", type=str, default=None, required=False,
            help="The output path to save radial profiles and peak features")
    parser.add_argument(
            "--beam_rmax", type=int, default=15, required=True,
            help="The maximum beam radius in pixel lengths.")
    parser.add_argument(
            "--sample_distance", type=float, default=None, required=False,
            help="The maximum beam radius in pixel lengths.")
    parser.add_argument(
            "--visualize", action="store_true",
            help="Visualize plots.")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path) if args.input_path \
            else None
    output_path = os.path.abspath(args.output_path) if args.output_path \
            else None
    beam_rmax = args.beam_rmax
    sample_distance = args.sample_distance
    visualize = args.visualize

    run_feature_extraction(
        input_path=input_path,
        output_path=output_path,
        beam_rmax=beam_rmax,
        sample_distance=sample_distance,
        visualize=visualize,
        )
