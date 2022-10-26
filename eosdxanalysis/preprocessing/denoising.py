"""
Methods for denoising the diffraction pattern
"""

import numpy as np
from scipy.ndimage import generic_filter
from eosdxanalysis.preprocessing.utils import create_circular_mask


def find_hot_spots(masked_image, threshold):
    """
    Parameters
    ----------

    masked_image : ndarray
        Image with beam removed (zeroed)

    threshold : int
        Any pixels with photon count greater than ``threshold`` are
        considered hot spots.
    """
    hot_spot_coords_array= np.array(np.where(masked_image > threshold)).reshape(-1,2)
    return hot_spot_coords_array

def filter_hot_spots(masked_image, threshold, filter_size=5, method="ignore"):
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
        Choice of ``ignore`` which sets all pixels near the hot spot to zero,
        which gets ignored by Gaussian filtering, or ``median``, which sets
        all pixels near the hot spot to the median value.
    """
    hot_spot_coords_array = find_hot_spots(masked_image, threshold)
    filtered_image = masked_image.copy()

    for hot_spot_coords in hot_spot_coords_array:
        hot_spot_roi_rows = slice(
                hot_spot_coords[0]-filter_size//2,
                hot_spot_coords[0]+filter_size//2+1)
        hot_spot_roi_cols = slice(
                hot_spot_coords[1]-filter_size//2,
                hot_spot_coords[1]+filter_size//2+1)

        hot_spot_roi = filtered_image[hot_spot_roi_rows, hot_spot_roi_cols]
        if method == "ignore":
            filtered_image[hot_spot_roi_rows, hot_spot_roi_cols] = 0
        elif method == "median":
            filtered_image[hot_spot_roi_rows, hot_spot_roi_cols] = np.median(
                    hot_spot_roi)

    return filtered_image
