"""
Methods for denoising the diffraction pattern
"""


import os
import argparse
import glob

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import generic_filter

from sklearn.base import OneToOneFeatureMixin
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from sklearn.utils import gen_batches

from eosdxanalysis.preprocessing.azimuthal_integration import azimuthal_integration

from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import find_center

DEFAULT_BEAM_RMAX = 15
DEFAULT_THRESHOLD = 0.75
DEFAULT_RELATIVE_THRESHOLD = True
DEFAULT_THRESHOLD_TYPE = "max"
DEFAULT_MAX_PIXELS = 10
DEFAULT_FILTER_SIZE = 5


def find_outlier_pixel_values(
        masked_image,
        threshold=DEFAULT_THRESHOLD,
        relative_threshold=DEFAULT_RELATIVE_THRESHOLD,
        threshold_type=DEFAULT_THRESHOLD_TYPE,
        max_pixels=DEFAULT_MAX_PIXELS):
    """
    Faulty pixels can be too high or too low.

    Parameters
    ----------

    masked_image : ndarray
        Image with beam removed (zeroed)

    threshold : number
        Any pixels with photon count greater than ``intensity_threshold`` are
        considered hot spots. If using ``relative`` detection method,
        ``intensity_threshold = threshold * max(masked_image)``. If using
        ``absolute`` detection method, ``intensity_threshold = threshold``.

    threshold_type : str
        Choice of ``max`` or ``min``. If ``max``, pixels above the threshold
        value will be filtered.

    detection_method : str
        Choice of ``relative`` or ``absolute`` threshold.

    max_faulty_pixels : int
        If more than ``max_faulty_pixels`` are found, only the first
        ``max_faulty_pixels`` are returned.

    Returns
    -------

    coords_array

    """
    if relative_threshold:
        intensity_threshold = threshold * np.max(masked_image)
    else:
        intensity_threshold = threshold

    if threshold_type == "max":
        coords_array = np.array(
                np.where(masked_image > intensity_threshold)).T
    elif threshold_type == "min":
        coords_array = np.array(
                np.where(masked_image < intensity_threshold)).T
    elif threshold_type == "minmax":
        max_coords_array = np.array(
                np.where(masked_image > intensity_threshold)).T
        min_coords_array = np.array(
                np.where(masked_image < intensity_threshold)).T
        coords_array = np.vstack(
                max_coords_array, min_coords_array)
    else:
        raise ValueError(
                "``threshold_type`` must be ``min`` or ``max``, or ``minmax``.")

    # Check if more than the maximum number of hot spots were found
    if len(coords_array) > max_pixels:
        # Sort by intensity, take the first ``max_pixels``
        intensities_all = np.sort(
                masked_image[coords_array])[::-1]
        intensities = intensities_all[:max_pixels]

        # Get the coordinates of the first ``max_pixels``
        coords_array = np.array(
                np.where(
                    masked_image == intensities)).T

    return coords_array

def filter_outlier_pixel_values(
        masked_image,
        threshold=DEFAULT_THRESHOLD,
        relative_threshold=DEFAULT_RELATIVE_THRESHOLD,
        threshold_type=DEFAULT_THRESHOLD_TYPE,
        max_pixels=DEFAULT_MAX_PIXELS,
        coords_array=None,
        filter_size=DEFAULT_FILTER_SIZE,
        fill=None):

    """
    Parameters
    ----------

    masked_image : ndarray
        Image with beam removed (zeroed)

    threshold : int
        Any pixels with photon count greater than ``threshold`` are
        considered hot spots.

    filter_size : int
        Size of filter ``filter_size`` by ``filter_size``.

    method : str
        Choice of ``median`` which sets all pixels near the hot spot to the
        median value, or ``zero``, which sets all pixels near the hot spot to
        zero (not recommended if performing subsequent image processing).

    """
    fill_list = ["median", "zero", "nan"]
    if fill not in fill_list:
        raise ValueError(
                "Invalid fill value ({})! Choose from {}".format(
                    fill, fill_list))

    if coords_array is None:
        coords_array = find_outlier_pixel_values(
                masked_image, threshold=threshold,
                relative_threshold=relative_threshold, max_pixels=max_pixels,
                threshold_type=threshold_type)
    filtered_image = masked_image.copy()

    for hot_spot_coords in coords_array:
        hot_spot_roi_rows = slice(
                hot_spot_coords[0]-filter_size//2,
                hot_spot_coords[0]+filter_size//2+1)
        hot_spot_roi_cols = slice(
                hot_spot_coords[1]-filter_size//2,
                hot_spot_coords[1]+filter_size//2+1)

        hot_spot_roi = filtered_image[hot_spot_roi_rows, hot_spot_roi_cols]
        if fill == "zero":
            filtered_image[hot_spot_roi_rows, hot_spot_roi_cols] = 0
        elif fill == "median":
            filtered_image[hot_spot_roi_rows, hot_spot_roi_cols] = np.median(
                    hot_spot_roi)
        elif fill == "nan":
            filtered_image[hot_spot_roi_rows, hot_spot_roi_cols] = np.nan

    return filtered_image


class FilterOutlierPixelValues(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Adapted from scikit-learn transforms
    Repair dead pixel areas of detector
    Replace dead pixels by np.nan
    """

    def __init__(self, *,
            copy=True,
            center=None,
            beam_rmax=DEFAULT_BEAM_RMAX,
            threshold=DEFAULT_THRESHOLD,
            relative_threshold=DEFAULT_RELATIVE_THRESHOLD,
            threshold_type=DEFAULT_THRESHOLD_TYPE,
            max_pixels=DEFAULT_MAX_PIXELS,
            coords_array=None,
            filter_size=DEFAULT_FILTER_SIZE,
            fill=None,
            ):

        """
        Parameters
        ----------
        copy : bool
            Creates copy of array if True (default = False).

        center : {(float, float)}
            Center of diffraction pattern (row, column).

        beam_rmax : int
            Set radius around beam center to ignore when finding dead pixels.

        """
        self.copy = copy
        self.center = center
        self.beam_rmax = beam_rmax
        self.threshold = threshold
        self.relative_threshold = relative_threshold
        self.threshold_type = threshold_type
        self.max_pixels = max_pixels
        self.coords_array = coords_array
        self.filter_size = filter_size
        self.fill = fill

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

    def transform(self, X, copy=True):
        """Parameters
        ----------
        X : {array-like}, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        center = self.center
        beam_rmax = self.beam_rmax
        threshold = self.threshold
        relative_threshold = self.relative_threshold
        threshold_type = self.threshold_type
        max_pixels = self.max_pixels
        coords_array = self.coords_array
        filter_size = self.filter_size
        fill = self.fill

        if copy:
            X = X.copy()

        # Loop over all samples using batches
        for idx in range(X.shape[0]):
            image = X[idx, ...].reshape(X.shape[1:])

            if not center:
                center = find_center(image)

            output_image = filter_outlier_pixel_values(
                    image, center=center,
                    threshold=threshold,
                    relative_threshold=relative_threshold,
                    threshold_type=threshold_type,
                    max_pixels=max_pixels,
                    coords_array=None,
                    filter_size=filter_size,
                    fill=None)

            X[idx, ...] = output_image

        return X

def filter_outlier_pixel_values_dir(
            input_path=None,
            output_path=None,
            overwrite=False,
            center=None,
            beam_rmax=DEFAULT_BEAM_RMAX,
            dead_pixel_threshold=DEFAULT_THRESHOLD,
            file_format=None,
            verbose=False):
    """
    Repairs dead pixels in images contained in a directory

    Parameters
    ----------

    input_path : str
        Path to input files to repair.

    output_path : str
        Output path to save repaired files.

    overwrite : bool
        Set to ``True`` to overwrite files in place.

    beam_rmax : int
        Set radius around beam center to ignore when finding dead pixels.

    dead_pixel_threshold : int
        Pixels above this threshold value are considered dead pixels.

    file_format : str
        Choice of "txt" (default), "tiff", or "npy".

    verbose : bool
        Print information about the repair process.

    Notes
    -----

    Repairs dead pixels on the detector, which have very high pixel values.

    Replaces dead pixels with nan values.

    Blocks beam area when finding dead pixels.

    Saves repaired files to specified output directory or
    overwrites data in place.

    """

    parent_path = os.path.dirname(input_path)

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    if not file_format:
        file_format = "txt"

    if input_path:
        # Given single input path
        # Get filepath list
        if file_format == "txt":
            file_pattern = "*.txt"
        elif file_format == "npy":
            file_pattern = "*.npy"
        elif file_format == "tiff":
            file_pattern = "*.tif*" # finds tiff or tif

        filepath_list = glob.glob(os.path.join(input_path, file_pattern))
        # Sort files list
        filepath_list.sort()

    if verbose is True:
        print("Found {} files to repair.".format(len(filepath_list)))

    # Store output directory info
    # Create output directory if it does not exist
    if output_path:
        pass
    elif overwrite and not output_path:
        output_path = input_path
    else:
        # Create preprocessing results directory
        results_dir = "preprocessed_results"
        results_path = os.path.join(parent_path, results_dir)

        # Create timestamped output directory
        input_dir = os.path.basename(input_path)
        output_dir = "preprocessed_results_{}".format(
                timestamp)
        output_path = os.path.join(results_path, output_dir)

    # Set copy for call to dead_pixel_repair
    copy = overwrite

    os.makedirs(output_path, exist_ok=True)

    if verbose is True:
        print("Saving data to\n{}".format(output_path))

    # Loop over files list
    for filepath in filepath_list:
        filename = os.path.basename(filepath)
        if file_format in ["txt", "tif", "tiff"]:
            image = np.loadtxt(filepath, dtype=np.float64)
        elif file_format == "npy":
            image = np.load(filepath, allow_pickle=True)

        if not center:
            center = find_center(image)

        output_image = dead_pixel_repair(
                image, copy=copy, center=center, beam_rmax=beam_rmax,
                dead_pixel_threshold=dead_pixel_threshold)

        # Save results
        data_output_filepath = os.path.join(output_path, filename)
        if file_format in ["txt", "tif", "tiff"]:
            np.savetxt(data_output_filepath, output_image)
        elif file_format == "npy":
            np.save(data_output_filepath, output_image)


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
            "--center", type=str,
            default=None,
            help="Set diffraction pattern center for entire dataset.")
    parser.add_argument(
            "--visualize", action="store_true",
            help="Visualize plots.")
    parser.add_argument(
            "--overwrite", action="store_true",
            help="Overwrite files.")
    parser.add_argument(
            "--file_format", type=str, default=None, required=False,
            help="The file format to use (txt, tiff, or npy).")
    parser.add_argument(
            "--verbose", action="store_true",
            help="Verbose output.")

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
    overwrite = args.overwrite
    file_format = args.file_format
    verbose = args.verbose
    center_kwarg = args.center
    center = center_kwarg.strip("()").split(",") if center_kwarg else None

    if input_path and find_dead_pixels:
        dead_pixel_repair_dir(
            input_path=input_path,
            output_path=output_path,
            overwrite=overwrite,
            center=center,
            beam_rmax=beam_rmax,
            dead_pixel_threshold=dead_pixel_threshold,
            file_format=file_format,
            verbose=verbose,
            )
    else:
        raise NotImplementedError()
