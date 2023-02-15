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

    weighted_centroid = sum/input_image.sum()

    return weighted_centroid

def check_beam_detector_alignment(input_filepath):
    """
    Prints the beam position
    """

    # load the image
    image = np.loadtxt(input_filepath)

    # calculating the weighted centroid
    weighted_centroid = find_weighted_centroid(image)
    print("Beam position: {:.1f}, {:.1f}".format(
        weighted_centroid[1], weighted_centroid.shape[0] - weighted_centroid[0]))

    # calculate error
    detector_center = np.array([127.5, 127.5])
    error = weighted_centroid - detector_center
    print("Beam position error: {:.1f}, {:.1f}".format(error[1], -error[0]))

    # check tolerance
    error_tol = np.array([5, 5])
    if (np.abs(error) > error_tol).any():
        print("Beam position is out of bounds!")
    else:
        print("Beam position is within tolerance.")

    return weighted_centroid

if __name__ == '__main__':
    """
    Check if the machine has proper beam-detector alignment
    """
    measurement_filepath = input("Enter the full path to the beam-only measurement file:\n")

    check_beam_detector_alignment(measurement_filepath)