"""
Code to find a peak by taking a dot-product with a Gaussian
"""

import numpy as np
from scipy.stats import multivariate_normal

def find_2d_peak(image, roi=None, window_size=3):
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
    for idx in range(image.shape[0]-window_size):
        for jdx in range(image.shape[0]-window_size):
            dot_product = image[idx:idx+window_size,
                                jdx:jdx+window_size].ravel() @ gaussian.ravel()
            if dot_product > dot_product_max:
                # Store the new peak location
                new_peak_location = np.array([idx+window_size//2,
                                            jdx+window_size//2])
                peak_location_list = [new_peak_location]
            if dot_product == dot_product_max:
                # Add the new peak location
                new_peak_location = np.array([idx+window_size//2,
                                            jdx+window_size//2])
                peak_location_list.append(new_peak_location)

    # If we have multiple peak locations, return the centroid
    peak_location_array = np.array(peak_location_list)
    peak_location = np.mean(peak_location_array, axis=0)

    return peak_location

def find_1d_peak(radial_profile, window_size=3):
    pass

def find_all_1d_peaks(radial_profile, window_size=3):
    # Ensure the window size is not smaller than the input size
    if window_size > len(radial_profile):
        raise ValueError("Window size must be less than input size.")

    # Create a 1D Gaussian window


if __name__ == "__main__":
    pass
