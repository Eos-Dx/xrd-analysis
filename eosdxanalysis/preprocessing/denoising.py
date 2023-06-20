"""
Methods for denoising the diffraction pattern
"""

import numpy as np
from scipy.ndimage import generic_filter
from eosdxanalysis.preprocessing.utils import create_circular_mask


def find_hot_spots(
        masked_image, threshold=0.75, detection_method="relative",
        max_hot_spots=10):
    """
    Parameters
    ----------

    masked_image : ndarray
        Image with beam removed (zeroed)

    threshold : number
        Any pixels with photon count greater than ``intensity_threshold`` are
        considered hot spots. If using ``relative`` detection method,
        ``intensity_threshold = threshold * max(masked_image)``. If using
        ``absolute`` detection method, ``intensity_threshold = threshold``.

    detection_method : str
        Choice of ``relative`` or ``absolute`` threshold.

    max_hot_spots : int
        If more than ``max_hot_spots`` are found, only the first
        ``max_hot_spots`` are returned.

    Returns
    -------

    hot_spot_coords_array

    """
    if detection_method == "absolute":
        intensity_threshold = threshold
    elif detection_method == "relative":
        intensity_threshold = threshold * np.max(masked_image)
    else:
        raise ValueError(
                "Invalid detection method ({})!".format(detection_method))

    bright_spot_coords_array = np.array(
            np.where(masked_image > intensity_threshold)).T

    # Check if more than the maximum number of hot spots were found
    if len(bright_spot_coords_array) > max_hot_spots:
        # Sort by intensity, take the first ``max_hot_spots``
        bright_spot_intensities_all = np.sort(
                masked_image[bright_spot_coords_array])[::-1]
        bright_spot_intensities = bright_spot_intensities_all[:max_hot_spots]

        # Get the coordinates of the first ``max_hot_spots``
        bright_spot_coords_array = np.array(
                np.where(
                    masked_image == bright_spot_intensities)).T

    hot_spot_coords_array = bright_spot_coords_array

    return hot_spot_coords_array

def filter_hot_spots(
        masked_image, threshold=0.75, detection_method="relative",
        max_hot_spots=10, hot_spot_coords_array=None, filter_size=5,
        fill="median"):
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

    if hot_spot_coords_array is None:
        hot_spot_coords_array = find_hot_spots(
                masked_image, threshold=threshold, detection_method=detection_method,
                max_hot_spots=max_hot_spots)
    filtered_image = masked_image.copy()

    for hot_spot_coords in hot_spot_coords_array:
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
