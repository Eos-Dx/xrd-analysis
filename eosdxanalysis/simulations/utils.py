"""
Functions to help with simulations code
"""

import numpy as np

PIXEL_WIDTH = 55e-6 # 55 um in [meters]
WAVELENGTH = 1.5418e-10 # 1.5418 Angstrom in [meters]
DISTANCE = 10e-3 # Distance from sample to detector in [meters]


def feature_pixel_location(spacing, distance=DISTANCE, wavelength=WAVELENGTH,
            pixel_width=PIXEL_WIDTH):
    """
    Function to calculate the pixel distance from the center to a
    specified feature defined by spacing.

    Inputs:
    - spacing: [meters]
    - distance: sample-to-detector distance [meters]
    - wavelength: source wavelength [meters]

    Returns:
    - distance from center to feature location in pixel units

    Note that since our sampling rate corresponds to pixels,
    pixel units directly correspond to array indices,
    i.e.  distance of 1 pixel = distance of 1 array element
    """
    twoTheta = np.arcsin(wavelength/spacing)
    d_inv = distance * np.tan(twoTheta)
    d_inv_pixels = d_inv / PIXEL_WIDTH
    return d_inv_pixels

