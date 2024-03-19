"""
This file includes functions and classes essential for azimuthal integration
"""

from functools import cache

import pandas as pd
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector


@cache
def initialize_azimuthal_integrator(
    pixel_size, center_column, center_row, wavelength, sample_distance_mm
):
    """
    Initializes an azimuthal integrator with given parameters.

    Parameters:
    - detector : pyFAI.detectors.Detector
        The detector used for integration.
    - wavelength : float
        The wavelength of the incident X-ray beam.
    - sample_dist : float
        The sample-to-detector distance.
    - center : tuple
        A tuple of (x, y) coordinates representing the center point
        for integration.

    Returns:
    - ai : pyFAI.azimuthalIntegrator.AzimuthalIntegrator
        An instance of the PyFAI AzimuthalIntegrator.
    """
    detector = Detector(pixel_size, pixel_size)
    ai = AzimuthalIntegrator(detector=detector)
    ai.setFit2D(
        sample_distance_mm, center_column, center_row, wavelength=wavelength
    )
    return ai


def perform_azimuthal_integration(
    row: pd.Series, npt=256, mask=None, mode="1D"
):
    """
    Perform azimuthal integration on a single row of a DataFrame.

    Parameters:
    - row : pandas.Series
        A row from a pandas DataFrame, expected to contain 'measurement_data'
        (a 2D array), 'center' (a tuple of (x, y) representing the center of
        integration), 'wavelength' and 'calculated_distance'.
    - ai : pyFAI.azimuthalIntegrator.AzimuthalIntegrator
        The AzimuthalIntegrator used for integration.
    - npt: int
        number of points

    Returns:
    - q_range : numpy.ndarray
        The array of q values (momentum transfer) resulting
        from the integration.
    - I : numpy.ndarray
        The intensity values resulting from the integration.
    """
    interpolation_q_range = row.get("interpolation_q_range")
    pixel_size = row["pixel_size"] * (10**-6)
    data = row["measurement_data"]
    center = row["center"]
    center_column, center_row = center[1], center[0]
    sample_distance_mm = row["calculated_distance"] * (10**3)
    wavelength = row["wavelength"] * 10

    ai_cached = initialize_azimuthal_integrator(
        pixel_size, center_column, center_row, wavelength, sample_distance_mm
    )

    if mode == "1D":
        radial, intensity = ai_cached.integrate1d(
            data, npt, radial_range=interpolation_q_range, mask=mask
        )
        return radial, intensity
    elif mode == "2D":
        intensity, radial, azimuthal = ai_cached.integrate2d(
            data, npt, radial_range=interpolation_q_range, mask=mask
        )
        return radial, intensity, azimuthal
