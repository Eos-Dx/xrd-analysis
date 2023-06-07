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

from skimage import io

from scipy.signal import find_peaks

from eosdxanalysis.preprocessing.utils import azimuthal_integration
from eosdxanalysis.preprocessing.utils import radial_intensity
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import find_center

from eosdxanalysis.preprocessing.image_processing import pad_image

from eosdxanalysis.calibration.utils import radial_profile_unit_conversion

DEFAULT_DET_XSIZE = 256


def run_azimuthal_preprocessing(
        input_path, wavelength_nm,
        output_path=None,
        input_dataframe_filepath=None,
        sample_distance_filepath=None,
        find_sample_distance_filepath=None,
        beam_rmax=15, visualize=False,
        azimuthal_mean=True,
        azimuthal_sum=False,
        file_format=None,
        det1_center=None,
        det_xspacing=None,
        det_xsize=None):
    """
    Parameters
    ----------

    det1_center : tuple
        Center of diffraction pattern on detector 1 in pixel coordinates

    det_xspacing : float
        Horizontal distance between detectors in pixel units

    det_xsize : float
        Horizontal length of detector in pixel units
    """
    if not (azimuthal_mean ^ azimuthal_sum):
        raise ValueError("Choose azimuthal_mean or azimuthal_sum.")
    sample_distance_m = None

    if input_path:
        # Given single input path
        # Get filepath list
        if file_format:
            if file_format == "txt":
                file_pattern = "*.txt"
            elif file_format == "npy":
                file_pattern = "*.npy"
            elif "tif" in file_format:
                file_pattern = "*.tif*" # Finds tiff or tif
            else:
                raise ValueError(
                        "Unrecognized file format: {}\n".format(
                            file_format) + "Must be ``txt``,``tiff``, or ``npy``.")
            filepath_list = glob.glob(os.path.join(input_path, file_pattern))
        else:
            filepath_list= glob.glob(os.path.join(input_path, "*.*"))
        # Sort files list
        filepath_list.sort()
    elif input_dataframe_filepath:
        input_path = os.path.basename(input_dataframe_filepath)
        input_df = pd.read_csv(input_dataframe_filepath, index_col="Filename")
        filepath_list = input_df["Filepath"].tolist()

    parent_path = os.path.dirname(input_path)

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

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

    # Data output path
    data_output_dir = "preprocessed_data"
    data_output_path = os.path.join(output_path, data_output_dir)
    os.makedirs(data_output_path, exist_ok=True)

    # Image output path
    image_output_dir = "preprocessed_images"
    image_output_path = os.path.join(output_path, image_output_dir)
    os.makedirs(image_output_path, exist_ok=True)

    print("Saving data to\n{}".format(data_output_path))
    print("Saving images to\n{}".format(image_output_path))

    # Loop over files list
    for filepath in filepath_list:
        filename = os.path.basename(filepath)
        if file_format is None:
            file_root, file_format = os.path.splitext(filename)
        if file_format == "txt":
            image = np.loadtxt(filepath, dtype=np.float64)
        elif file_format == "npy":
            image = np.load(filepath)
        elif "tif" in file_format:
            image = io.imread(filepath).astype(np.float64)
        else:
            raise ValueError(
                    "Unrecognized file format: {}\n".format(
                        file_format.strip(".")) + "Must be ``txt`` or ``tiff``.")

        if type(det1_center) != type(None):
            if all([det_xspacing, det_xsize]):
                center = (det1_center[0], - (det_xsize - det1_center[1]) - det_xspacing)
        else:
            # Find the center
            center = find_center(image)

        if beam_rmax > 0:
            # Block out the beam
            beam_mask = create_circular_mask(
                    image.shape[0], image.shape[1], center=center, rmax=beam_rmax)
            masked_image = image.copy()
            masked_image[beam_mask] = np.nan
        else:
            masked_image = image

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

        if azimuthal_mean:
            radial_profile = azimuthal_integration(
                    enlarged_masked_image, center=new_center, end_radius=padding_amount)
        elif azimuthal_sum:
            radial_profile = radial_intensity(
                    enlarged_masked_image, center=new_center, end_radius=padding_amount)

        # Save data to file
        # Get approximate sample distance from folder name
        sample_distance_approx_list = np.unique(
                re.findall(r"dist_[0-9]{2,3}mm", filepath, re.IGNORECASE))
        if len(sample_distance_approx_list) != 1:
            data_output_filename = "radial_{}.txt".format(filename)
        else:
            sample_distance_approx = sample_distance_approx_list[0].lower()
            data_output_filename = "radial_{}_{}".format(sample_distance_approx, filename)
        data_output_filepath = os.path.join(data_output_path,
                data_output_filename)

        if find_sample_distance_filepath:
            calibration_results_dir = "calibration_results"
            sample_distance_path = os.path.join(
                    parent_path, calibration_results_dir)
            try:
                sample_distance_filepath = glob.glob(os.path.join(
                    sample_distance_path, "*.json"))[0]
            except IndexError as err:
                print("No sample distance file found.")
                raise err

        if sample_distance_filepath:
            with open(sample_distance_filepath, "r") as distance_fp:
                sample_distance_results = json.loads(distance_fp.read())
                sample_distance_m = sample_distance_results["sample_distance_m"]

            # Combine radial profile intensity and q-units
            q_range = radial_profile_unit_conversion(
                    radial_count=radial_profile.size,
                    sample_distance=sample_distance_m,
                    wavelength_nm=wavelength_nm,
                    radial_units="q_per_nm")
            results = np.hstack([q_range.reshape(-1,1), radial_profile.reshape(-1,1)])
            np.savetxt(data_output_filepath, results)
        elif input_dataframe_filepath:
            sample_distance_m = input_df.loc[filename, "Sample_Distance_m"]

            # Combine radial profile intensity and q-units
            q_range = radial_profile_unit_conversion(
                    radial_profile.size,
                    sample_distance_m,
                    wavelength_nm,
                    radial_units="q_per_nm")
            results = np.hstack([q_range.reshape(-1,1), radial_profile.reshape(-1,1)])
            np.savetxt(data_output_filepath, results)

            # Save q-min and q-max to dataframe
            input_df.loc[filename, "q_min"] = q_range.min()
            input_df.loc[filename, "q_max"] = q_range.max()

        else:
            np.savetxt(data_output_filepath, radial_profile)

        # Save image preview to file
        plot_title = "Radial Intensity Profile\n{}".format(filename)
        fig = plt.figure(plot_title)

        if sample_distance_m:
            plt.plot(q_range, radial_profile)
            plt.xlabel(r"q $\mathrm{{nm}^{-1}}$")
        else:
            plt.plot(range(radial_profile.size), radial_profile)
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

    # Save dataframe ext
    if input_dataframe_filepath:
        input_dataframe_filename = os.path.basename(input_dataframe_filepath)
        output_dataframe_filename = input_dataframe_filename.replace(".", "_ext.")
        output_dataframe_filepath = os.path.join(
                output_path, output_dataframe_filename)
        input_df.to_csv(output_dataframe_filepath, index=True)

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
            "--wavelength_nm", type=float, required=True,
            help="Wavelength in nanometers.")
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
    parser.add_argument(
            "--azimuthal_mean", action="store_true",
            help="Use azimuthal mean.")
    parser.add_argument(
            "--azimuthal_sum", action="store_true",
            help="Use azimuthal sum.")
    parser.add_argument(
            "--file_format", type=str, default=None, required=False,
            help="File format: ``txt`` or ``tiff``.")
    parser.add_argument(
            "--det1_center", type=str, default=None, required=False,
            help="Center of diffraction pattern on detector 1 in pixel coordinates.")
    parser.add_argument(
            "--det_xspacing", type=float, default=None, required=False,
            help="Horizontal distance between detectors.")
    parser.add_argument(
            "--det_xsize", type=float, default=DEFAULT_DET_XSIZE, required=False,
            help="Horizontal length of detector 1.")


    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path) if args.input_path \
            else None
    wavelength_nm = args.wavelength_nm
    output_path = os.path.abspath(args.output_path) if args.output_path \
            else None
    input_dataframe_filepath = os.path.abspath(args.input_dataframe_filepath) \
            if args.input_dataframe_filepath else None
    beam_rmax = args.beam_rmax
    find_sample_distance_filepath = args.find_sample_distance_filepath 
    sample_distance_filepath = args.sample_distance_filepath
    visualize = args.visualize
    azimuthal_mean = args.azimuthal_mean
    azimuthal_sum = args.azimuthal_sum
    file_format = args.file_format
    det1_center_arg = args.det1_center
    if det1_center_arg != None:
        det1_center = np.array(det1_center_arg.split(",")).astype(float)
        if len(det1_center) != 2:
            raise ValueError("Detector 1 center must be a tuple.")
    else:
        det1_center = None
    det_xspacing = args.det_xspacing
    det_xsize = args.det_xsize

    if not (azimuthal_sum or azimuthal_mean):
        azimuthal_mean = True


    run_azimuthal_preprocessing(
        input_path=input_path,
        wavelength_nm=wavelength_nm,
        output_path=output_path,
        input_dataframe_filepath=input_dataframe_filepath,
        find_sample_distance_filepath=find_sample_distance_filepath,
        sample_distance_filepath=sample_distance_filepath,
        beam_rmax=beam_rmax,
        visualize=visualize,
        azimuthal_mean=azimuthal_mean,
        azimuthal_sum=azimuthal_sum,
        file_format=file_format,
        det1_center=det1_center,
        det_xspacing=det_xspacing,
        det_xsize=det_xsize,
        )
