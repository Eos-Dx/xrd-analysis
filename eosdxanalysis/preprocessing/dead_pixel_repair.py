"""
Code to turn dead pixels that are stuck into nan values
Note: Can only repair images with high-value dead pixels.
"""

import os
import argparse
import glob

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from eosdxanalysis.preprocessing.utils import azimuthal_integration
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import find_center


DEFAULT_DEAD_PIXEL_THRESHOLD = 1e4
DEFAULT_BEAM_RMAX = 15



def dead_pixel_repair_dir(
            input_path=None,
            output_path=None,
            beam_rmax=DEFAULT_BEAM_RMAX,
            dead_pixel_threshold=DEFAULT_DEAD_PIXEL_THRESHOLD):
    """
    Repairs dead pixels in images contained in a directory
    """

    parent_path = os.path.dirname(input_path)

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    if input_path:
        # Given single input path
        # Get filepath list
        filepath_list = glob.glob(os.path.join(input_path, "*.txt"))
        # Sort files list
        filepath_list.sort()

    # Store output directory info
    # Create output directory if it does not exist
    if not output_path:
        # Create preprocessing results directory
        results_dir = "preprocessed_results"
        results_path = os.path.join(parent_path, results_dir)

        # Create timestamped output directory
        input_dir = os.path.basename(input_path)
        output_dir = "preprocessed_results_{}".format(
                timestamp)
        output_path = os.path.join(results_path, output_dir)

    os.makedirs(output_path, exist_ok=True)

    print("Saving data to\n{}".format(output_path))

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

        # Find dead pixels
        dead_pixel_locations = masked_image > dead_pixel_threshold

        output_image = image.copy()
        output_image[dead_pixel_locations] = np.nan

        # Save results
        data_output_filepath = os.path.join(output_path, filename)
        np.savetxt(data_output_filepath, output_image)


if __name__ == '__main__':
    """
    Repairs dead pixels in images
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", type=str, required=False,
            help="The path to measurement files with dead pixels.")
    parser.add_argument(
            "--input_image_filepath", type=str, required=False,
            help="The path to measurement file with dead pixels.")
    parser.add_argument(
            "--output_path", type=str, default=None, required=False,
            help="The output path to save radial profiles and peak features")
    parser.add_argument(
            "--hot_spot_coords_approx", type=tuple,
            help="Approximate hot spot coordinates.")
    parser.add_argument(
            "--find_dead_pixels", action="store_true",
            help="Try to find the dead pixels automatically.")
    parser.add_argument(
            "--beam_rmax", type=int, default=DEFAULT_BEAM_RMAX,
            help="Radius of beam to ignore.")
    parser.add_argument(
            "--dead_pixel_threshold", type=float,
            default=DEFAULT_DEAD_PIXEL_THRESHOLD,
            help="Set pixel value of dead pixels.")
    parser.add_argument(
            "--visualize", action="store_true",
            help="Visualize plots.")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path) if args.input_path \
            else None
    input_image_filepath = args.input_image_filepath
    output_path = os.path.abspath(args.output_path) if args.output_path \
            else None
    hot_spot_coords_approx = args.hot_spot_coords_approx
    find_dead_pixels = args.find_dead_pixels
    beam_rmax = args.beam_rmax
    dead_pixel_threshold = args.dead_pixel_threshold 
    visualize = args.visualize

    if input_path and find_dead_pixels:
        dead_pixel_repair_dir(
            input_path=input_path,
            output_path=output_path,
            beam_rmax=beam_rmax,
            dead_pixel_threshold=dead_pixel_threshold,
            )
    else:
        raise NotImplementedError()
