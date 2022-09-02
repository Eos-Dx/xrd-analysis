"""
Code to find a peak by taking a dot-product with a Gaussian
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm

def find_peaks_2d(image, window_size=5):
    """
    Given an image, find 2D peaks using a Gaussian curve fitting procedure

    Inputs:
    - image: 2D numpy array
    - window_size: int
    """
    # Check that window size is not smaller than the image size
    if window_size > image.shape[0] or window_size > image.shape[1]:
        raise ValueError("Window size must be less than image size.")

    # Create a Gaussian window
    x = np.linspace(-1, 1, window_size)
    y = np.linspace(-1, 1, window_size)
    xv, yv = np.meshgrid(x, y)
    pos =  np.dstack((xv, yv))
    rv = multivariate_normal([0.0, 0.0], [[1, 0.0], [0.0, 1]])
    gaussian = rv.pdf(pos)

    # Now take the inner product of the gaussian window
    # within the roi, and find the maximum location
    peak_location_list = []
    dot_product_max = 0
    for idx in range(image.shape[0]-window_size+1):
        for jdx in range(image.shape[0]-window_size+1):
            window = image[idx:idx+window_size, jdx:jdx+window_size]
            dot_product = window.ravel() @ gaussian.ravel()

            # Check if dot product is new maximum
            if np.greater(dot_product, dot_product_max):
                dot_product_max = dot_product
                # Store the new peak location
                new_peak_location = np.array([idx+window_size/2-0.5,
                                            jdx+window_size/2-0.5])
                # Start a new list
                peak_location_list = [new_peak_location]
            # Check if dot product is close to the maximum
            elif np.isclose(dot_product, dot_product_max):
                # Add the new peak location
                new_peak_location = np.array([idx+window_size/2-0.5,
                                            jdx+window_size/2-0.5])
                peak_location_list.append(new_peak_location)
    return np.array(peak_location_list)

def find_peaks_1d(radial_profile, window_size=3, peak_width=1):
    """
    Find a single peak in a 1D array
    This uses index notation.

    Examples
    --------
    - [0, 1, 0] has peak location at index 1
    - [1, 1] has peak location at index 0.5 (between 0 and 1)

    peak_width is the standard deviation
    """
    peak_location_list = []
    dot_product_max = 0

    x = np.linspace(-1, 1, window_size)
    gaussian = norm.pdf(x, scale=peak_width)
    radial_profile = np.squeeze(radial_profile)

    for idx in range(len(radial_profile)-window_size+1):
        window = radial_profile[idx:idx+window_size]
        dot_product = np.dot(window, gaussian)

        # Check if dot product is new maximum
        if np.greater(dot_product, dot_product_max):
            dot_product_max = dot_product
            # Store the new peak location
            new_peak_location = idx+window_size/2-0.5
            peak_location_list = [new_peak_location]
        # Check if dot product is close to the maximum
        elif np.isclose(dot_product, dot_product_max):
            # Add the new peak location
            new_peak_location = idx+window_size/2-0.5
            peak_location_list.append(new_peak_location)

    peak_location_array = np.array(peak_location_list)

    return peak_location_array


def sub_pixel_peak_location(image, peak_location_coords, window_size=5):
    """
    Given an image and the 2D coordinates in index notation,
    find the sub-pixel peak location using center of mass calculation.

    Input:
    - image: 2D numpy array
    - peak_location_coords: integers (row, column) array notation
    - window_size: integer (size of NxN window to calculate center of mass)
    """

    return peak_location


if __name__ == "__main__":
    pass
