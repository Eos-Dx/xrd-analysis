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
    Initializes or gets a cached azimuthal integrator with given parameters.

    Parameters:
    - pixel_size : float
        The size of a pixel on the detector, in meters.
    - center_column : float
        The column index of the center point for integration.
    - center_row : float
        The row index of the center point for integration.
    - wavelength : float
        The wavelength of the incident X-ray beam, in angstroms.
    - sample_distance_mm : float
        The sample-to-detector distance, in millimeters.

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
        A row from a pandas DataFrame, expected to contain the
        following columns:
            - 'measurement_data' : array_like
                A 2D array containing the measurement data.
            - 'center' : tuple
                A tuple of two floats representing the center coordinates
                (y, x)for the integration.
            - 'wavelength' : float
                The wavelength of the incident X-ray beam, in nanometers.
            - 'calculated_distance' : float
                The calculated distance from the sample to the detector, in
                meters.
            - 'pixel_size' : float
                The size of a pixel on the detector, in micrometers.
            - 'interpolation_q_range' : tuple or None, optional
                A tuple (q_min, q_max) specifying the range of q values
                for interpolation, where q is the momentum transfer,
                in reciprocal nanometers.
                If not provided, the full q range will be used.
    - npt : int, optional (default=256)
        Number of points for integration.
    - mask : array_like, optional
        Mask array to be applied during integration, 1-s are for pixels
        to be masked, 0-s for pixels to be integrated.
    - mode : {'1D', '2D'}, optional (default='1D')
        Mode of integration, '1D' for 1-dimensional integration and '2D' for
        2-dimensional integration.

    Returns:
    - radial : numpy.ndarray
        The array of radial q values (momentum transfer) resulting
        from the integration.
    - intensity : numpy.ndarray
        The intensity values resulting from the integration.
    - azimuthal : numpy.ndarray, optional
        The array of azimuthal angles (radians) for 2D integration, returned
        only if mode is '2D'.
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
