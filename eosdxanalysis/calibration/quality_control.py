"""
The file is to provide automated quality control tool for technicians to check the initial beam quality
It checks the following:
  1. The beam detector alignment
  2. The beam aperture alignment
"""

import numpy as np

def find_weighted_centroid(input_image):
    # The weighted sum of all data coordinates divided by the total weight
    sum = np.array([0, 0])
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            sum = sum + np.array([i, j])*input_image[i][j]

    row_weighted_centroid, col_weighted_centroid = sum/input_image.sum()

    return row_weighted_centroid, col_weighted_centroid

def check_beam_detector_alignment(input_filepath):
    """
    Prints the beam position
    """

    # load the image
    image = np.loadtxt(input_filepath)

    # calculating the weighted centroid
    weighted_centroid = find_weighted_centroid(image)
    print(weighted_centroid)

    return weighted_centroid

if __name__ == '__main__':
    """
    Check if the machine has proper beam-detector alignment
    """
    measurement_filepath = input("Enter the full path to the beam-only measurement file:\n")

    check_beam_detector_alignment(measurement_filepath)