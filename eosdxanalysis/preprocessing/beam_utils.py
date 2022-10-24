"""
Beam removal
"""

import os
import glob

import numpy as np
from skimage.transform import warp_polar
from scipy.signal import find_peaks

from eosdxanalysis.preprocessing.utils import zerocross1d

import matplotlib.pyplot as plt


def azimuthal_integration(image, center=None, output_shape=(360,128)):
    """
    Performs azimuthal integration

    Parameters
    ----------

    image : ndarray

    output_shape : 2-tuple int

    Returns
    -------

    profile_1d : (n,1)-array float 
    """
    polar_image = warp_polar(
            image, center=center, radius=output_shape[1],
            output_shape=output_shape, preserve_range=True)
    profile_1d = np.sum(polar_image, axis=0)
    return profile_1d

def beam_radius(image, center=None, bounds=None):
    """
    Calculate the beam radius by finding the first location where the first
    derivative is positive and the second derivative crosses zero
    (positive to negative).

    Parameters
    ----------

    image : ndarray

    bounds : 2-tuple float
        (min, max)


    Returns
    -------

    radius : int
    """
    # Calculate the azimuthal integration profile
    profile_1d = azimuthal_integration(image, center)
    # Calculate the first and second derivatives
    grad1 = np.gradient(profile_1d)
    grad2 = np.gradient(grad1)

    # Find the first valley (maximum of negative profile)
    valleys = find_peaks(-profile_1d)

    # Check if the first valley was found
    try:
        first_valley = valleys[0][0]
    except IndexError as err:
        print("No valleys found!")
        raise err

    grad1_positive = np.where(grad1 > 0)

    # Find inflection points (zero-crossings of second derivative)
    inflection_points = zerocross1d(
            np.arange(grad2.size), grad2, getIndices=True)

    # Find the first inflection point past the first valley
    try:
        inflection_point_indices = inflection_points[1]

        # Find the first inflection point of interest
        past_first_valley = inflection_point_indices > first_valley

        # Add 1 to the index to get the start of the diffraction pattern
        # since zerocross1d returns the indx preceding the zero-crossing event
        first_inflection_poi = \
                inflection_point_indices[past_first_valley][0] + 1

        return first_inflection_poi 
    except:
        raise ValueError("Beam radius not found!")
