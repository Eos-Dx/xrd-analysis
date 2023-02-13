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

from eosdxanalysis.preprocessing.beam_utils import azimuthal_integration


feature_list = [
        "peak_location_beam",
        "peak_location_9A",
        "peak_location_5A",
        "peak_location_4A",
        "peak_width_beam",
        "peak_width_9A",
        "peak_width_5A",
        "peak_width_4A",
        ]

# Create dictionary of peaks and their theoretical locations in angstroms
peak_location_angstrom_dict = {
        "peak_location_beam": 1,
        "peak_location_9A": 9.8e-10,
        "peak_location_5A": 5.1e-10,
        "peak_location_4A": 4.5e-10,
        }

## Convert to pixel locations
#peak_location_pixel_dict = {}
#for peak_name, peak_location in peak_location_angstrom_dict.items():
#    peak_location_pixel_dict[peak_name] = feature_pixel_location(peak_location)


def sum_of_gaussians():
    gau = None
    return gau

def run_feature_extraction(input_path, output_path):
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
    data_output_dir = "preprocessed_{}".format(timestamp)
    data_output_path = os.path.join(output_path, data_output_dir)
    os.makedirs(data_output_path, exist_ok=True)

    # Loop over files list
    for filepath in filepath_list:
        filename = os.path.basename(filepath)
        image = np.loadtxt(filepath, dtype=np.uint32)

        radial_profile = azimuthal_integration(image)

        peak_indices, properties = find_peaks(
                radial_profile)

        # Save data to file
        output_filename = "radial_{}".format(filename) + ".png"
        data_output_file_path = os.path.join(data_output_path, output_filename)

        np.savetxt(data_output_file_path,
                        np.round(radial_profile).astype(np.uint32), fmt='%i')

        # Save image preview to file
        plot_title = "Azimuthal Integration {}".format(filename)
        fig = plt.figure(plot_title)
        plt.scatter(range(radial_profile.size), radial_profile)
        plt.scatter(peak_indices, radial_profile[peak_indices])

        plt.title(plot_title)

        # Save image preview to file
        plt.savefig(data_output_file_path)

        # Add extracted features to dataframe
        # df.loc[len(df.index)+1] = [filename] + [radial_profile]

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
            "--input_path", default=None, required=True,
            help="The path to data to extract features from")
    parser.add_argument(
            "--output_path", default=None, required=True,
            help="The output path to save radial profiles and peak features")

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    run_feature_extraction(
        input_path=input_path,
        output_path=output_path,
        )
