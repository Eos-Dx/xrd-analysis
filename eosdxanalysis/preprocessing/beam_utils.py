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


def azimuthal_integration(
        image, center=None, radius=None, num_points=360,
        start_angle=None, end_angle=None, scale=1):
    """
    Performs azimuthal integration

    Parameters
    ----------

    image : ndarray

    center : (num, num)

    radius : int

    num_points : int

    start_angle : float

    end_angle : float

    scale : int

    Returns
    -------

    profile_1d : (n,1)-array float 
    """
    # Set radius
    if not radius:
        radius = np.max(image.shape)/2
    if not num_points:
        num_points = 360
    if type(scale) != int:
        raise ValueError("Scale must be an integer")
    if (start_angle is None) and (end_angle is None):
        start_angle = -np.pi + np.pi/num_points/2
        end_angle = np.pi - np.pi/num_points/2
    if start_angle > end_angle:
        raise ValueError("Start angle must be greater than end angle")

    angle_range = np.arange(start_angle, end_angle, step=2*np.pi/num_points)

    # Perform a polar warp on the input image
    output_shape = (num_points, radius*scale)
    polar_image = warp_polar(
            image, center=center, radius=radius,
            output_shape=output_shape, preserve_range=True)
    # Take subset of 2*pi if start or end angle provided
    start_angle_idx = int(start_angle / (2*np.pi)  * num_points)
    end_angle_idx = int(end_angle / (2*np.pi) * num_points)
    polar_image_subset = polar_image[start_angle_idx:end_angle_idx, :]
    profile_1d = np.mean(polar_image_subset, axis=0)

    return profile_1d

def first_valley_location(image=None, center=None, profile_1d=None, bounds=None):
    """
    Calculate the first valley location given an input image and its center,
    or calculate the first valley location from the 1-d azimuthal integration
    profile.

    Parameters
    ----------

    image : ndarray

    center : (float, float)
        The center of the diffraction pattern in array notation ``(row, col)``.

    profile_1d : (n,1) ndarray of ``float`` values.
        1-d azimuthal integration profile

    bounds : (float, float)
        (min, max)


    Returns
    -------

    first_valley, profile_1d : (int, (n,1)-array float)
        Returns the first valley location and 1-d azimuthal integration
        profile. If first valley location is not found, then ``None`` is
        returned in its place.

    """
    # Calculate the azimuthal integration profile
    profile_1d = azimuthal_integration(image, center)

    # Find the first valley (maximum of negative profile)
    valleys = find_peaks(-profile_1d)

    # Check if the first valley was found
    try:
        first_valley = valleys[0][0]
        return first_valley, profile_1d
    except IndexError as err:
        return None, profile_1d


def beam_extent(
        image=None, center=None, profile_1d=None, first_valley=None,
        bounds=None):
    """
    Calculate the beam extent using the first valley location and the
    subsequent inflection point. The inflection point is used to indicate the
    extent of the beam (beam radius).

    Parameters
    ----------

    image : ndarray

    center : (float, float)
        The center of the diffraction pattern in array notation ``(row, col)``.

    profile_1d : (n,1) ndarray of ``float`` values.
        1-d azimuthal integration profile

    first_valley : int
        Pixel location of the first valley.

    bounds : (float, float)
        (min, max)


    Returns
    -------

    first_inflection_point, first_valley, profile_1d : (int, int, (n,1) ndarray float)
        Returns the inflection point after the first valley.
        If first valley or inflection point are not found then their values will be ``None``.

    """
    first_valley, profile_1d = first_valley_location(
            image, center=center, bounds=bounds)

    # Calculate the first and second derivatives
    grad1 = np.gradient(profile_1d)
    grad2 = np.gradient(grad1)

    grad1_positive = np.where(grad1 > 0)

    # Find inflection points (zero-crossings of second derivative)
    inflection_points = zerocross1d(
            np.arange(grad2.size), grad2, getIndices=True)

    # Find the first inflection point past the first valley
    try:
        inflection_point_indices = inflection_points[1]

        # Find the first inflection point of interest
        past_first_valley = inflection_point_indices > first_valley

        # Note: zerocross1d returns the index preceding the zero-crossing event
        first_inflection_point = \
                inflection_point_indices[past_first_valley][0]

        return first_inflection_point, first_valley, profile_1d
    except:
        return None, first_valley, profile_1d
