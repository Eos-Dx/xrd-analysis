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

DEFAULT_BEAM_RMAX = 0
DEFAULT_THRESHOLD = 0.75
DEFAULT_ABSOLUTE = False
DEFAULT_LIMIT_TYPE = "max"
DEFAULT_FILTER_SIZE = 5
DEFAULT_FILL_METHOD = "median"


def find_outlier_pixel_values(
        image : np.ndarray,
        mask : np.ndarray = None,
        threshold : float = DEFAULT_THRESHOLD,
        absolute : bool = DEFAULT_ABSOLUTE,
        limit_type : str = DEFAULT_LIMIT_TYPE,
        ):
    """
    Filters outlier pixels with values that are either too high or too low.

    Parameters
    ----------

    image : ndarray
        Input image.

    mask : ndarray
        Areas = 1 will be masked (ignored). Must be same shape as input image.

    threshold : number
        Threshold to use for detecting outlier pixel values. If relative limit
        type is used, threshold must be in the range [0, 1].

    absolute : bool
        Use absolute threshold (default = False).

    limit_type : str
        Choice of ``max`` (default) or ``min``. If ``max``, pixels with values above the
        threshold will be identified as outliers.

    Returns
    -------

    coords_array : ndarray
        Coordinates of outliers in the following paired format:
            np.array([array of row indices, array of column indices])

    """
    if type(mask) == np.ndarray:
        working_image = image.copy()
        working_image[mask] = np.nan
    else:
        working_image = image

    if absolute:
        intensity_threshold = threshold
    else:
        intensity_threshold = threshold * np.nanmax(working_image)

    if limit_type == "max":
        coords_array = np.array(
                np.where(working_image > intensity_threshold)).T
    elif limit_type == "min":
        coords_array = np.array(
                np.where(working_image < intensity_threshold)).T
    else:
        raise ValueError(
                "``threshold_type`` must be ``min`` or ``max``.")

    return np.asanyarray(coords_array)

def filter_outlier_pixel_values(
        image : np.ndarray,
        mask : np.ndarray = None,
        threshold : float = DEFAULT_THRESHOLD,
        absolute : bool = DEFAULT_ABSOLUTE,
        limit_type : str = DEFAULT_LIMIT_TYPE,
        coords_array : np.ndarray = None,
        filter_size : int = DEFAULT_FILTER_SIZE,
        fill_method : str = DEFAULT_FILL_METHOD,
        ):

    """
    Filters outlier pixels with values that are either too high or too low.

    Parameters
    ----------

    image : ndarray
        Input image.

    mask : ndarray
        Areas = 1 will be masked (ignored). Must be same shape as input image.

    threshold : number
        Threshold to use for detecting outlier pixel values. If relative limit
        type is used, threshold must be in the range [0, 1].

    absolute : bool
        Use absolute threshold (default = False).

    limit_type : str
        Choice of ``max`` (default) or ``min``. If ``max``, pixels with values above the
        threshold will be identified as outliers.

    coords_array : ndarray
        Coordinates of outliers in the following paired format:
            np.array([array of row indices, array of column indices])

    filter_size : int
        Size of filter to use for pixels with outlier values.

    fill_method : str
        Method to fill in region of interest surrounding pixels with outlier
        values. Choice of ``median`` (default), ``zero``, or ``nan``.

    Returns
    -------

    filtered_image : ndarray
        Image with outliers removed.

    """
    # Check fill method
    FILL_METHOD_LIST = ["median", "zero", "nan"]
    if fill_method not in FILL_METHOD_LIST:
        raise ValueError(
                "Invalid fill method ({})! Choose from {}".format(
                    fill_method, FILL_METHOD_LIST))

    # Find outlier pixel values if coordinates not provided
    if type(coords_array) is type(None):
        coords_array = find_outlier_pixel_values(
                image,
                mask=mask,
                threshold=threshold,
                absolute=absolute,
                limit_type=limit_type,
                )

    # Work from unmasked image
    filtered_image = image.copy()

    # Filter outliers
    for outlier_coords in coords_array:
        # Extract region of interest slices based on filter size
        outlier_roi_rows = slice(
                int(outlier_coords[0]-filter_size//2),
                int(outlier_coords[0]+filter_size//2+1))
        outlier_roi_cols = slice(
                int(outlier_coords[1]-filter_size//2),
                int(outlier_coords[1]+filter_size//2+1))

        # Get outlier region of interest
        outlier_roi = filtered_image[outlier_roi_rows, outlier_roi_cols].copy()

        if fill_method == "zero":
            # Set values in region of interest around outlier pixel to zero
            filtered_image[outlier_roi_rows, outlier_roi_cols] = 0
        elif fill_method in ["median", "mean"]:
            # Set values in region of interest around outlier pixel to median
            # of neighbors (border)

            # Get borders
            border_roi_rows = slice(
                    int(outlier_coords[0]-filter_size//2-1),
                    int(outlier_coords[0]+filter_size//2+2))
            border_roi_cols = slice(
                    int(outlier_coords[1]-filter_size//2-1),
                    int(outlier_coords[1]+filter_size//2+2))

            # Get the border region of interest
            border_roi = filtered_image[border_roi_rows, border_roi_cols].copy()
            # Set interior to nan
            border_roi[1:-1, 1:-1] = np.nan
            # Set outlier region of interest to median of border region of 
            # interest
            if fill_method == "median":
                filtered_image[outlier_roi_rows, outlier_roi_cols] = \
                        np.nanmedian(border_roi)
            elif fill_method == "mean":
                filtered_image[outlier_roi_rows, outlier_roi_cols] = \
                        np.nanmean(border_roi)
        elif fill_method == "nan":
            # Set values in region of inteest around outlier pixel to nan
            filtered_image[outlier_roi_rows, outlier_roi_cols] = np.nan

    return filtered_image


class FilterOutlierPixelValues(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Adapted from scikit-learn transforms
    Repair dead pixel areas of detector
    Replace dead pixels by np.nan
    """

    def __init__(self, *,
            copy=True,
            center=None,
            beam_rmax : int = DEFAULT_BEAM_RMAX,
            threshold : float = DEFAULT_THRESHOLD,
            absolute : bool = DEFAULT_ABSOLUTE,
            limit_type : str = DEFAULT_LIMIT_TYPE,
            coords_array : np.ndarray = None,
            filter_size : int = DEFAULT_FILTER_SIZE,
            fill_method : str = DEFAULT_FILL_METHOD,
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
        self.absolute = absolute
        self.limit_type = limit_type
        self.coords_array = coords_array
        self.filter_size = filter_size
        self.fill_method = fill_method

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

    def transform(self, X, copy=True, mask : np.ndarray = None):
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
        absolute = self.absolute
        limit_type = self.limit_type
        coords_array = self.coords_array
        filter_size = self.filter_size
        fill_method= self.fill_method

        if copy:
            X = X.copy()

        # Loop over all samples using batches
        for idx in range(X.shape[0]):
            image = X[idx, ...].reshape(X.shape[1:])

            if not center:
                center = find_center(image)

            # Beam masking
            if beam_rmax > 0 and type(mask) == type(None):
                # Block out the beam
                mask = create_circular_mask(
                        image.shape[0], image.shape[1], center=center, rmax=beam_rmax)

            filtered_image = filter_outlier_pixel_values(
                    image,
                    mask=mask,
                    threshold=threshold,
                    absolute=absolute,
                    limit_type=limit_type,
                    coords_array=coords_array,
                    filter_size=filter_size,
                    fill_method=fill_method,
                    )

            X[idx, ...] = filtered_image

        return X

def filter_outlier_pixel_values_dir(
            input_path=None,
            output_path=None,
            overwrite=False,
            center=None,
            beam_rmax=DEFAULT_BEAM_RMAX,
            threshold : float = DEFAULT_THRESHOLD,
            absolute : bool = DEFAULT_ABSOLUTE,
            limit_type = DEFAULT_LIMIT_TYPE,
            coords_array : np.ndarray = None,
            filter_size : int = DEFAULT_FILTER_SIZE,
            fill_method : str = DEFAULT_FILL_METHOD,
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

        # Beam masking
        if beam_rmax > 0:
            # Block out the beam
            mask = create_circular_mask(
                    image.shape[0], image.shape[1], center=center, rmax=beam_rmax)

        else:
            mask = None

        filtered_image = filter_outlier_pixel_values(
                image,
                mask=mask,
                threshold=threshold,
                absolute=absolute,
                limit_type=limit_type,
                coords_array=coords_array,
                filter_size=filter_size,
                fill_method=fill_method)

        # Save results
        data_output_filepath = os.path.join(output_path, filename)
        if file_format in ["txt", "tif", "tiff"]:
            np.savetxt(data_output_filepath, filtered_image)
        elif file_format == "npy":
            np.save(data_output_filepath, filtered_image)


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
            "--threshold", type=float, required=False,
            help="Threshold to use to identify outlier pixel values.")
    parser.add_argument(
            "--limit_type", type=str, default="max",
            help="Threshold limit type. Choice of ``max`` (default) or ``min``.")
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
    beam_rmax = args.beam_rmax
    threshold = args.threshold
    limit_type = args.limit_type
    visualize = args.visualize
    overwrite = args.overwrite
    file_format = args.file_format
    verbose = args.verbose
    center_kwarg = args.center
    center = center_kwarg.strip("()").split(",") if center_kwarg else None

    filter_outlier_pixel_values_dir(
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
        center=center,
        beam_rmax=beam_rmax,
        threshold=threshold,
        limit_type=limit_type,
        file_format=file_format,
        verbose=verbose,
        )
