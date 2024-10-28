"""
This file includes functions and classes essential for azimuthal integration
"""

import os
from functools import cache

import pandas as pd
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector


@cache
def initialize_azimuthal_integrator_df(
    pixel_size, center_column, center_row, wavelength, sample_distance_mm
):
    """
    Initializes or gets a cached azimuthal integrator with the\
    given parameters.

    :param float pixel_size: The size of a pixel on the detector, in meters.
    :param float center_column: The column index of the center point for\
        integration.
    :param float center_row: The row index of the center point for integration.
    :param float wavelength: The wavelength of the incident X-ray beam, in\
        angstroms.
    :param float sample_distance_mm: The sample-to-detector distance, in\
        millimeters.

    :returns:
        - **pyFAI.azimuthalIntegrator.AzimuthalIntegrator**: An instance of\
            the PyFAI AzimuthalIntegrator.
    """

    detector = Detector(pixel_size, pixel_size)
    ai = AzimuthalIntegrator(detector=detector)
    ai.setFit2D(
        sample_distance_mm, center_column, center_row, wavelength=wavelength
    )
    return ai


@cache
def initialize_azimuthal_integrator_poni(file):
    """
    Initializes or gets a cached azimuthal integrator with the given\
    parameters.

    :param str file: Path to the poni file.

    :returns:
        - **pyFAI.azimuthalIntegrator.AzimuthalIntegrator**: An instance of\
            the PyFAI AzimuthalIntegrator.
    """

    ai = pyFAI.load(file)
    return ai


def perform_azimuthal_integration(
    row: pd.Series,
    npt=256,
    mask=None,
    mode="1D",
    calibration_mode="dataframe",
    thres=3,
    max_iter=5,
    poni_dir=None,
):
    """
    Perform azimuthal integration on a single row of a DataFrame.

    :param pandas.Series row: A row from a pandas DataFrame, expected to\
        contain the following columns:

        - **measurement_data** (*array_like*): A 2D array containing the\
            measurement data.
        - **center** (*tuple*): A tuple of two floats representing the center\
            coordinates (y, x) for the integration.
        - **wavelength** (*float*): The wavelength of the incident X-ray beam,\
            in nanometers.
        - **calculated_distance** (*float*): The calculated distance from the\
            sample to the detector, in meters.
        - **pixel_size** (*float*): The size of a pixel on the detector, in\
            micrometers.
        - **interpolation_q_range** (*tuple or None, optional*): A tuple\
            (q_min, q_max) specifying the range of q values for interpolation,\
            where q is the momentum transfer, in reciprocal nanometers.\
            If not provided, the full q range will be used.
        - **azimuthal_range** (*tuple or None, optional*): A tuple\
            (min_deg, max_deg) specifying the range of degrees for\
            integration. Degrees are in a range from -180 to 180.

        **OR**

        - **calibration_measurement_id** (*int*): Integer associated with the\
            ID of a specific calibration, used to name the poni file.
        - **measurement_data** (*array_like*): A 2D array containing the\
            measurement data.
        - **ponifile** (*str*): A string for poni file usage.
        - **interpolation_q_range** (*tuple or None, optional*): A tuple\
            (q_min, q_max) specifying the range of q values for interpolation,\
            where q is the momentum transfer, in reciprocal nanometers.\
            If not provided, the full q range will be used.
        - **azimuthal_range** (*tuple or None, optional*): A tuple\
            (min_deg, max_deg) specifying the range of degrees for\
            integration. Degrees are in a range from -180 to 180.

    :param int npt: Number of points for integration. Defaults to 256.
    :param array_like mask: Mask array to be applied during integration.\
        1s are for pixels to be masked, 0s for pixels to be integrated.\
        Defaults to None.
    :param str mode: Mode of integration. '1D' for 1-dimensional integration\
        and '2D' for 2-dimensional integration. Defaults to '1D'.
    :param str calibration_mode: Mode of calibration. 'dataframe' is used when\
        calibration values are columns in the dataframe, 'poni' is used when\
        calibration is in a poni file. Defaults to 'dataframe'.
    :param str or None poni_dir: Directory path containing .poni files for\
        calibration. Only applicable when calibration_mode is set to 'poni'.\
        Defaults to None.

    :returns:
        - **numpy.ndarray**: The array of radial q values (momentum transfer)\
            resulting from the integration.
        - **numpy.ndarray**: The intensity values resulting from the\
            integration.
        - **numpy.ndarray, optional**: The array of azimuthal angles (radians)\
            for 2D integration, returned only if mode is '2D'.
    """

    interpolation_q_range = row.get("interpolation_q_range")
    azimuthal_range = row.get("azimuthal_range")
    data = row["measurement_data"]

    if calibration_mode == "dataframe":
        pixel_size = row["pixel_size"] * (10**-6)
        center = row["center"]
        center_column, center_row = center[1], center[0]
        sample_distance_mm = row["calculated_distance"] * (10**3)
        wavelength = row["wavelength"] * 10

        ai_cached = initialize_azimuthal_integrator_df(
            pixel_size,
            center_column,
            center_row,
            wavelength,
            sample_distance_mm,
        )
    elif calibration_mode == "poni":
        calibration_measurement_id = row["calibration_measurement_id"]
        ai_cached = initialize_azimuthal_integrator_poni(
            os.path.join(poni_dir, f"{calibration_measurement_id}.poni")
        )

    if mode == "1D":
        radial, intensity = ai_cached.integrate1d(
            data,
            npt,
            radial_range=interpolation_q_range,
            azimuth_range=azimuthal_range,
            mask=mask,
        )
        return radial, intensity, ai_cached.dist
    elif mode == "2D":
        intensity, radial, azimuthal = ai_cached.integrate2d(
            data,
            npt,
            radial_range=interpolation_q_range,
            azimuth_range=azimuthal_range,
            mask=mask,
        )
        return radial, intensity, azimuthal, ai_cached.dist
    elif mode == "sigma_clip":
        radial, intensity, _ = ai_cached.sigma_clip_ng(
            data,
            npt,
            thres=thres,
            max_iter=max_iter,
            error_model="azimuth",
            radial_range=interpolation_q_range,
            azimuth_range=azimuthal_range,
            mask=mask,
        )
        return radial, intensity, ai_cached.dist
