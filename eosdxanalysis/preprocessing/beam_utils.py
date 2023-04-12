"""
Beam removal
"""

import os
import glob

import numpy as np
from skimage.transform import warp_polar
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator

from eosdxanalysis.preprocessing.utils import zerocross1d

import matplotlib.pyplot as plt

AZIMUTHAL_POINT_COUNT_DEFAULT = 360

def azimuthal_integration(
        image, center=None, radius=None,
        azimuthal_point_count=AZIMUTHAL_POINT_COUNT_DEFAULT,
        start_angle=None, end_angle=None, res=1):
    """
    Performs azimuthal integration

    Parameters
    ----------

    image : ndarray
        Diffraction image.

    center : (num, num)
        Center of diffraction pattern.

    radius : int

    azimuthal_point_count : int
        Number of points in azimuthal dimension.

    start_angle : float
        Radians

    end_angle : float
        Radians

    res : int
        Resolution

    Returns
    -------

    profile_1d : (n,1)-array float 
        n = azimuthal_point_count
    """
    # Set radius
    if not radius:
        radius = np.max(image.shape)/2*res
    if not azimuthal_point_count:
        azimuthal_point_count = AZIMUTHAL_POINT_COUNT*res
    if type(res) != int:
        raise ValueError("Scale must be an integer")
    if (start_angle is None) and (end_angle is None):
        start_angle = -np.pi + 2*np.pi/azimuthal_point_count/2
        end_angle = np.pi - 2*np.pi/azimuthal_point_count/2
    if start_angle > end_angle:
        raise ValueError("Start angle must be greater than end angle")

    azimuthal_step = 2*np.pi/azimuthal_point_count

    azimuthal_space = np.linspace(
            start_angle,
            end_angle,
            num=AZIMUTHAL_POINT_COUNT_DEFAULT*res)
    radial_range = np.arange(radius)

    # Perform a polar warp on the input image for entire azimuthal range
    output_shape = (AZIMUTHAL_POINT_COUNT_DEFAULT*res, radius*res)
    polar_image = warp_polar(
            image, center=center, radius=radius,
            output_shape=output_shape, preserve_range=True)

    # Interpolate if subset is needed
    interp = RegularGridInterpolator((azimuthal_space, radial_range), polar_image)

    azimuthal_space_subset = np.linspace(
            start_angle,
            end_angle,
            num=azimuthal_point_count*res)

    AA, RR = np.meshgrid(azimuthal_space_subset, radial_range, indexing="ij", sparse=True)

    polar_image_subset = interp((AA, RR))

    # Calculate the mean
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
