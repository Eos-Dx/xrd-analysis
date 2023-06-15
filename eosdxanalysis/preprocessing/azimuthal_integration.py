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

from sklearn.base import OneToOneFeatureMixin
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from eosdxanalysis.preprocessing.image_processing import enlarge_image

from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import find_center
from eosdxanalysis.preprocessing.utils import warp_polar_preprocessor
from eosdxanalysis.preprocessing.utils import AZIMUTHAL_POINT_COUNT_DEFAULT

from eosdxanalysis.calibration.utils import radial_profile_unit_conversion

DEFAULT_DET_XSIZE = 256


def azimuthal_integration(
        image, center=None, start_radius=None, end_radius=None,
        azimuthal_point_count=AZIMUTHAL_POINT_COUNT_DEFAULT,
        start_angle=None, end_angle=None, res=1):
    """
    Performs 2D -> 1D azimuthal integration yielding mean intensity as a
    function of radius

    Parameters
    ----------

    image : ndarray
        Diffraction image.

    center : (num, num)
        Center of diffraction pattern.

    radius : int

    azimuthal_point_count : int
        Number of points in azimuthal dimension.

    start_angle : float
        Radians

    end_angle : float
        Radians

    res : int
        Resolution

    Returns
    -------

    profile_1d : (n,1)-array float
        n = azimuthal_point_count
    """
    polar_image_subset = warp_polar_preprocessor(
        image,
        center=center,
        start_radius=start_radius,
        end_radius=end_radius,
        azimuthal_point_count=azimuthal_point_count,
        start_angle=start_angle,
        end_angle=end_angle,
        res=1)

    # Calculate the mean
    profile_1d = np.nanmean(polar_image_subset, axis=0)

    return profile_1d

def radial_intensity_sum(
        image, center=None, start_radius=None, end_radius=None,
        azimuthal_point_count=AZIMUTHAL_POINT_COUNT_DEFAULT,
        start_angle=None, end_angle=None, res=1):
    """
    Performs 2D -> 1D radial intensity summation yielding total intensity
    as a function of radius.

    Parameters
    ----------

    image : ndarray
        Diffraction image.

    center : (num, num)
        Center of diffraction pattern.

    radius : int

    azimuthal_point_count : int
        Number of points in azimuthal dimension.

    start_angle : float
        Radians

    end_angle : float
        Radians

    res : int
        Resolution

    Returns
    -------

    profile_1d : (n,1)-array float
        n = azimuthal_point_count
    """
    polar_image_subset = warp_polar_preprocessor(
        image,
        center=center,
        start_radius=start_radius,
        end_radius=end_radius,
        azimuthal_point_count=azimuthal_point_count,
        start_angle=start_angle,
        end_angle=end_angle,
        res=1)

    # Calculate the sum
    profile_1d = np.nansum(polar_image_subset, axis=0)

    return profile_1d


class AzimuthalIntegration(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Adapted from scikit-learn transforms
    Repair dead pixel areas of detector
    Replace dead pixels by np.nan
    """

    def __init__(self, *, copy=True,
            center=None, start_radius=None, end_radius=None,
            azimuthal_point_count=AZIMUTHAL_POINT_COUNT_DEFAULT,
            start_angle=None, end_angle=None, res=1):
        """
        Parameters
        ----------
        copy : bool
            Creates copy of array if True (default = False).

        center : (num, num)
            Center of diffraction pattern.

        start_radius : int

        end_radius : int

        azimuthal_point_count : int
            Number of points in azimuthal dimension.

        start_angle : float
            Radians

        end_angle : float
            Radians

        res : int
            Resolution

        center : {(float, float)}
            Center of diffraction pattern (row, column).
        """
        self.copy = copy
        self.center = center
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.azimuthal_point_count = azimuthal_point_count
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.res = res

    def fit(self, X, y=None, sample_weight=None):
        """Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        return self

    def transform( self, X, copy=True):
        """Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        if copy is True:
            # Create a copy of the data, otherwise overwrite
            result = np.zeros_like(X)

        # Loop over all samples using batches
        for idx in range(X.shape[0]):
            image = X[idx, ...].reshape(X.shape[1:])
            center = find_center(image)

            radial_profile = azimuthal_integration()

            # Set radial profile data based on ``copy`` value
            if copy is True:
                output_data = radial_profile.copy()
            else:
                output_data = radial_profile

            # Store radial profile data in appropriate array
            if copy is True:
                result[idx, ...] = output_data
            else:
                # Overwrite data
                X[idx, ...] = output_data

        if copy is True:
            return result
        else:
            return X

def azimuthal_integration_dir(
        input_path,
        output_path=None,
        wavelength_nm=None,
        sample_distance_m = None,
        input_dataframe_filepath=None,
        sample_distance_filepath=None,
        find_sample_distance_filepath=None,
        autofind_center=False,
        beam_rmax=15,
        visualize=False,
        file_format=None,
        center=None,
        det_xspacing=None,
        det_xsize=None,
        fill=np.nan):
    """
    Parameters
    ----------

    center : tuple
        Center of diffraction pattern on detector 1 in pixel coordinates

    det_xspacing : float
        Horizontal distance between detectors in pixel units

    det_xsize : float
        Horizontal length of detector in pixel units
    """
    # Ensure center is given or autofind is set
    if center is None and not autofind_center:
        raise ValueError("Must specify center or set ``autofind_center=True``.")

    # Ensure wavelength is given if required
    if any([
        sample_distance_filepath, find_sample_distance_filepath,
        sample_distance_m]):
        if not wavelength_nm:
            raise ValueError(
                    "Must specify wavelength for conversion to momentum transfer units q.")

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
        parent_path = os.path.dirname(input_path)
    elif input_dataframe_filepath:
        input_path = os.path.dirname(input_dataframe_filepath)
        input_df = pd.read_csv(
                input_dataframe_filepath, index_col="Filename").fillna("")
        filepath_list = input_df["Filepath"].tolist()
        parent_path = input_path

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
            file_root, file_ext = os.path.splitext(filename)
            file_format = file_ext[1:]
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

        # Center of diffraction pattern
        if autofind_center:
            # Second detector case
            if all([det_xspacing, det_xsize]):
                center = (center[0], - (det_xsize - center[1]) - det_xspacing)
            # First detector case
            else:
                center = find_center(image)

        # Beam masking
        if beam_rmax > 0:
            # Block out the beam
            beam_mask = create_circular_mask(
                    image.shape[0], image.shape[1], center=center, rmax=beam_rmax)
            masked_image = image.copy()
            masked_image[beam_mask] = fill
        else:
            masked_image = image

        enlarged_masked_image, new_center, padding_amount = enlarge_image(
                masked_image, center=center)

        radial_profile = azimuthal_integration(
                enlarged_masked_image, center=new_center, end_radius=padding_amount)

        # Save data to file
        if input_dataframe_filepath:
            orig_filename = input_df.loc[filename, "orig_file_name"]
            if orig_filename:
                output_filename = "{}.txt".format(orig_filename)
            else:
                # Blind data
                output_filename = "{}.txt".format(filename)
        else:
            output_filename = "{}.txt".format(filename)

        # Get approximate sample distance from folder name
        sample_distance_approx_list = np.unique(
                re.findall(r"dist_[0-9]{2,3}mm", filepath, re.IGNORECASE))
        if len(sample_distance_approx_list) != 1:
            data_output_filename = "radial_{}.txt".format(output_filename)
        else:
            sample_distance_approx = sample_distance_approx_list[0].lower()
            data_output_filename = "radial_{}_{}".format(
                    sample_distance_approx, output_filename)
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
            sample_distance_m = input_df.loc[filename, "sample_distance_m"]

            # Combine radial profile intensity and q-units
            q_range = radial_profile_unit_conversion(
                    radial_count=radial_profile.size,
                    sample_distance=sample_distance_m,
                    wavelength_nm=wavelength_nm,
                    radial_units="q_per_nm")
            results = np.hstack([q_range.reshape(-1,1), radial_profile.reshape(-1,1)])
            np.savetxt(data_output_filepath, results)

            # Save q-min and q-max to dataframe
            input_df.loc[filename, "q_min"] = q_range.min()
            input_df.loc[filename, "q_max"] = q_range.max()

        else:
            np.savetxt(data_output_filepath, radial_profile)

        # Save image preview to file
        plot_title = "Radial Intensity Profile\n{}".format(output_filename)
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
            "--input_path", type=str, required=False,
            help="The path to data to extract features from")
    parser.add_argument(
            "--output_path", type=str, default=None, required=False,
            help="The output path to save radial profiles and peak features")
    parser.add_argument(
            "--wavelength_nm", type=float, required=True,
            help="Wavelength in nanometers.")
    parser.add_argument(
            "--sample_distance_m", type=float, required=True,
            help="Sample distance in meters.")
    parser.add_argument(
            "--input_dataframe_filepath", type=str, required=False,
            help="The dataframe containing file paths to extract features from")
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
            "--file_format", type=str, default=None, required=False,
            help="File format: ``txt`` or ``tiff``.")
    parser.add_argument(
            "--center", type=str, default=None, required=False,
            help="Center of diffraction pattern on detector 1 in pixel coordinates.")
    parser.add_argument(
            "--autofind_center", action="store_true",
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
    output_path = os.path.abspath(args.output_path) if args.output_path \
            else None
    wavelength_nm = args.wavelength_nm
    sample_distance_m = args.sample_distance_m
    input_dataframe_filepath = os.path.abspath(args.input_dataframe_filepath) \
            if args.input_dataframe_filepath else None
    beam_rmax = args.beam_rmax
    find_sample_distance_filepath = args.find_sample_distance_filepath 
    sample_distance_filepath = args.sample_distance_filepath
    visualize = args.visualize
    file_format = args.file_format
    center_kwarg = args.center
    if center_kwarg != None:
        center = np.array(center_kwarg.split(",")).astype(float)
        if len(center) != 2:
            raise ValueError("Detector 1 center must be a tuple.")
    else:
        center = None
    autofind_center = args.autofind_center
    det_xspacing = args.det_xspacing
    det_xsize = args.det_xsize

    if not (input_path ^ input_dataframe_filepath):
        raise ValueError("Input path or dataframe is required.")

    azimuthal_integration_dir(
        input_path=input_path,
        output_path=output_path,
        wavelength_nm=wavelength_nm,
        sample_distance_m=sample_distance_m,
        input_dataframe_filepath=input_dataframe_filepath,
        find_sample_distance_filepath=find_sample_distance_filepath,
        sample_distance_filepath=sample_distance_filepath,
        beam_rmax=beam_rmax,
        visualize=visualize,
        file_format=file_format,
        center=center,
        autofind_center=autofind_center,
        det_xspacing=det_xspacing,
        det_xsize=det_xsize,
        )
