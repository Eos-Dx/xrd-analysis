from typing import Tuple

import numpy as np
from skimage.measure import label, regionprops


def get_center(data: np.ndarray, threshold=3.0) -> Tuple[float]:
    """
    Takes SAXS data and determines the center of the beam.

    Parameters:
    - data : numpy.ndarray
        The input SAXS data.
    - threshold : float, optional
        The threshold factor for identifying the center of the beam.
        Defaults to 3.0 times the average value of the input data.

    Returns:
    - center : tuple of float
        The coordinates of the center of the beam in the input data.
        If no center is found, returns (np.NaN, np.NaN).
    """
    average_value = np.nanmean(data)

    # Set the threshold to be X higher than the average value
    threshold = threshold * average_value
    binary_image = data > threshold

    # Label connected regions
    labeled_image = label(binary_image)

    # Get region properties for all labeled regions
    regions = regionprops(labeled_image)

    # Find the largest region
    max_area = 0
    max_region = None

    for region in regions:
        if region.area > max_area:
            max_area = region.area
            max_region = region

    # Get the centroid of the largest region
    if max_region is not None:
        center = max_region.centroid
        center = (center[0], center[1])
    else:
        center = (np.NaN, np.NaN)

    return center
