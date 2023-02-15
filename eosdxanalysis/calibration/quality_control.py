"""
The file is to provide automated quality control tool for technicians to check the initial beam quality
It checks the following:
  1. The beam detector alignment
  2. The beam aperture alignment
"""

import numpy as np

def find_weighted_centroid(input_image):

    sum = np.array([0, 0])
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            sum = sum + np.array([i, j])*input_image[i][j]

    row_weighted_centroid, col_weighted_centroid = sum/(input_image.sum())

    return row_weighted_centroid, col_weighted_centroid

def beam_position(input_image):
    row_center, col_center = find_centroid(input_image)
    return row_center, col_center

