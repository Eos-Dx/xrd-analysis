"""
This file includes functions and classes essential for azimuthal integration
"""

import os
from functools import cache

import numpy as np
import pandas as pd
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector
from scipy.stats import mstats

from xrdanalysis.data_processing.utility_functions import (
    generate_poni_from_text,
    perform_weighted_integration,
    prepare_angular_ranges,
)


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
def initialize_azimuthal_integrator_poni_text(ponifile_text):
    """
    Initializes or gets a cached azimuthal integrator based on ponifile text.

    :param str ponifile_text: Text content of the .poni file
    :returns: PyFAI AzimuthalIntegrator instance
    :rtype: pyFAI.azimuthalIntegrator.AzimuthalIntegrator
    """
    # Generate a temporary .poni file
    temp_poni_path = generate_poni_from_text(ponifile_text)

    try:
        # Load the integrator
        ai = pyFAI.load(temp_poni_path)
        return ai
    finally:
        # Always attempt to remove the temporary file
        try:
            os.unlink(temp_poni_path)
        except Exception:
            pass  # Ignore any errors in file deletion


def perform_azimuthal_integration(
    row: pd.Series,
    column: str,
    npt=256,
    mask=None,
    mode="1D",
    calibration_mode="dataframe",
    thres=3,
    max_iter=5,
    calc_cake_stats=False,
    angles=None,
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
    :param thres: Threshold for sigma clipping. Used only in "sigma_clip" \
    mode. Defaults to 3.
    :type thres: int
    :param max_iter: Maximum number of iterations for sigma clipping. \
    Used only in "sigma_clip" mode. Defaults to 5.
    :type max_iter: int
    :param calc_cake_stats: Whether to calculate cake statistics in "2D" mode.\
    Defaults to False.
    :type calc_cake_stats: bool
    :param angles: List of angle ranges for integration in "rotating_angles" \
    mode. Defaults to None.
    :type angles: list of tuples or None

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
    data = row[column]

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
        ai_cached = initialize_azimuthal_integrator_poni_text(row["ponifile"])

    center_x = ai_cached.poni2 / ai_cached.detector.pixel2
    center_y = ai_cached.poni1 / ai_cached.detector.pixel1

    if mode == "1D":
        result = ai_cached.integrate1d(
            data,
            npt,
            radial_range=interpolation_q_range,
            azimuth_range=azimuthal_range,
            error_model="azimuthal",
            mask=mask,
        )
        return (
            result.radial,
            result.intensity,
            result.sigma,
            result.std,
            ai_cached.dist,
            center_x,
            center_y,
        )
    elif mode == "2D":
        result = ai_cached.integrate2d(
            data,
            npt,
            radial_range=interpolation_q_range,
            azimuth_range=azimuthal_range,
            mask=mask,
        )

        mean_col = variance_col = std_col = skewness_col = kurtosis_col = None

        if calc_cake_stats:
            masked_array = np.ma.masked_equal(result.intensity, 0)

            # Mean along columns, ignoring masked values
            mean_col = masked_array.mean(axis=0)

            # Variance along columns, ignoring masked values
            variance_col = masked_array.var(axis=0)

            # SRD along columns, ignoring masked values
            std_col = masked_array.std(axis=0)

            # Skewness along columns, ignoring masked values
            skewness_col = mstats.skew(masked_array, axis=0)

            # Kurtosis along columns, ignoring masked values
            kurtosis_col = mstats.kurtosis(masked_array, axis=0)

        return (
            result.radial,
            result.intensity,
            result.azimuthal,
            ai_cached.dist,
            center_x,
            center_y,
            mean_col,
            variance_col,
            std_col,
            skewness_col,
            kurtosis_col,
        )
    elif mode == "sigma_clip":
        result = ai_cached.sigma_clip_ng(
            data,
            npt,
            thres=thres,
            max_iter=max_iter,
            error_model="azimuthal",
            radial_range=interpolation_q_range,
            azimuth_range=azimuthal_range,
            mask=mask,
        )
        return (
            result.radial,
            result.intensity,
            result.sigma,
            result.std,
            ai_cached.dist,
            center_x,
            center_y,
        )
    elif mode == "rotating_angles":
        results = []
        for start_angle, end_angle in angles:
            # Get processed ranges and their properties
            range_info = prepare_angular_ranges(start_angle, end_angle)

            # Perform integration and get combined results
            radial, intensity, sigma, std = perform_weighted_integration(
                data, ai_cached, range_info, npt, interpolation_q_range, mask
            )

            # Append the final result
            results.append(
                (
                    (start_angle, end_angle),
                    radial,
                    intensity,
                    sigma,
                    std,
                )
            )

        return results, ai_cached.dist, center_x, center_y


def calculate_deviation(
    row: pd.Series,
    above_limits=[1.2],
    below_limits=[0.8],
    npt=256,
    mask=None,
):
    """
    Calculate deviation of measurements against predefined limits.

    :param row: A pandas Series containing data and configuration for \
    deviation calculation.
    :type row: pd.Series
    :param above_limits: A list of upper threshold multipliers for deviation \
    calculation. Defaults to [1.2].
    :type above_limits: list[float]
    :param below_limits: A list of lower threshold multipliers for deviation \
    calculation. Defaults to [0.8].
    :type below_limits: list[float]
    :param npt: Number of points for azimuthal integration. Defaults to 256.
    :type npt: int
    :param mask: An optional mask to exclude specific data points. \
    Defaults to None.
    :type mask: np.ndarray or None
    :return: Results containing above and below deviation images, and their \
    respective averages.
    :rtype: tuple
    """

    interpolation_q_range = row.get("interpolation_q_range")
    data = row["measurement_data"]
    azimuthal_range = row.get("azimuthal_range")

    ai_cached = initialize_azimuthal_integrator_poni_text(row["ponifile"])

    bins, _ = ai_cached.integrate1d(
        data,
        npt,
        radial_range=interpolation_q_range,
        azimuth_range=azimuthal_range,
        mask=mask,
    )

    # Calculate radial array and digitize bin indices
    radial = ai_cached.array_from_unit(
        data.shape, "center", "q_nm^-1", scale=True
    )
    bin_indices = np.digitize(radial, bins)

    # Calculate the average intensity in each bin (distance) using vectorized
    # operations
    unique_distances = np.unique(bin_indices)
    image_with_averages = np.zeros_like(data, dtype=float)

    # Fill in the averages for each unique distance value
    for dist in unique_distances:
        mask = bin_indices == dist
        image_with_averages[mask] = np.mean(data[mask])

    # Prepare arrays to store images and distance averages for each threshold
    images_above = []
    images_below = []
    all_distance_averages_higher = []
    all_distance_averages_lower = []

    # Calculate images and averages for each above limit
    for above_limit in above_limits:
        higher_than_average = data > image_with_averages * above_limit
        image_above = np.where(higher_than_average, data, 0)
        images_above.append(image_above)

        # Calculate the average for values above the threshold in each distance
        averages_higher = [
            (
                np.mean(data[(bin_indices == dist) & higher_than_average])
                if np.any((bin_indices == dist) & higher_than_average)
                else np.nan
            )
            for dist in unique_distances
        ]
        all_distance_averages_higher.append(np.array(averages_higher))

    # Calculate images and averages for each below limit
    for below_limit in below_limits:
        lower_than_average = (data < image_with_averages * below_limit) & (
            data != 0
        )
        image_below = np.where(lower_than_average, data, 0)
        images_below.append(image_below)

        # Calculate the average for values below the threshold in each distance
        averages_lower = [
            (
                np.mean(data[(bin_indices == dist) & lower_than_average])
                if np.any((bin_indices == dist) & lower_than_average)
                else np.nan
            )
            for dist in unique_distances
        ]
        all_distance_averages_lower.append(np.array(averages_lower))

    return (
        above_limits,
        images_above,
        all_distance_averages_higher,
        below_limits,
        images_below,
        all_distance_averages_lower,
    )


def calculate_deviation_cake(
    row: pd.Series,
    above_limits=[1.2],
    below_limits=[0.8],
    npt=256,
    mask=None,
):
    """
    Calculate deviation in the "cake" representation of azimuthal data.

    :param row: A pandas Series containing data and configuration for \
    deviation calculation.
    :type row: pd.Series
    :param above_limits: A list of upper threshold multipliers for deviation \
    calculation. Defaults to [1.2].
    :type above_limits: list[float]
    :param below_limits: A list of lower threshold multipliers for deviation \
    calculation. Defaults to [0.8].
    :type below_limits: list[float]
    :param npt: Number of points for azimuthal integration. Defaults to 256.
    :type npt: int
    :param mask: An optional mask to exclude specific data points. \
    Defaults to None.
    :type mask: np.ndarray or None
    :return: Results containing above and below deviation "cake" images, \
    reverse transformations and their respective averages.
    :rtype: tuple
    """

    azimuthal_range = row.get("azimuthal_range")
    interpolation_q_range = row.get("interpolation_q_range")
    data = row["measurement_data"]

    ai_cached = initialize_azimuthal_integrator_poni_text(row["ponifile"])

    # Extract necessary values
    result = ai_cached.integrate2d(
        data,
        npt,
        mask=mask,
        azimuth_range=azimuthal_range,
        radial_range=interpolation_q_range,
    )

    cake_image = result.intensity

    # Mask the cake image to handle zeros
    masked_array = np.ma.masked_equal(cake_image, 0)
    column_means = np.ma.mean(masked_array, axis=0)

    # Elongate column_means to match the shape of cake_image
    column_means_elongated = np.tile(column_means, (cake_image.shape[0], 1))

    # Prepare result holders for images and means for each above and below
    # limit
    images_above = []
    cakes_above = []
    means_above = []
    images_below = []
    cakes_below = []
    means_below = []

    # Iterate over each limit in above_limits
    for limit in above_limits:
        # Calculate the mask for values greater than the limit times the
        # corresponding column means
        mask_above = cake_image > (column_means_elongated * limit)

        # Create a masked array for values above the limit
        intensity_above = np.ma.masked_array(
            cake_image, mask=~mask_above
        )  # Mask all values not above

        cake_above = np.where(mask_above, cake_image, 0)

        # Create the image for values that are above the limit
        img_above = ai_cached.calcfrom2d(
            cake_above,
            result.radial,
            result.azimuthal,
            shape=data.shape,
            mask=mask,
            dim1_unit="q_nm^-1",
        )

        # Calculate the mean for the pixels above the limit, ensuring to take
        # the mean across columns
        mean_values = np.mean(intensity_above, axis=0)
        mean_values[mean_values == 0] = np.nan
        means_above.append(mean_values)

        cakes_above.append(cake_above)
        images_above.append(img_above)

    # Iterate over each limit in below_limits
    for limit in below_limits:
        # Calculate the mask for values less than the limit times the
        # corresponding column means
        mask_below = cake_image < (column_means_elongated * limit)

        # Create a masked array for values below the limit
        intensity_below = np.ma.masked_array(
            cake_image, mask=~mask_below
        )  # Mask all values not below

        cake_below = np.where(mask_below, cake_image, 0)

        # Create the image for values that are below the limit
        img_below = ai_cached.calcfrom2d(
            cake_below,
            result.radial,
            result.azimuthal,
            shape=data.shape,
            mask=mask,
            dim1_unit="q_nm^-1",
        )

        # Calculate the mean for the pixels below the limit, ensuring to
        # take the mean across columns
        mean_values = np.mean(intensity_below, axis=0)
        mean_values[mean_values == 0] = np.nan
        means_below.append(mean_values)

        cakes_below.append(cake_below)
        images_below.append(img_below)

    return (
        above_limits,
        cakes_above,
        images_above,
        means_above,
        below_limits,
        cakes_below,
        images_below,
        means_below,
    )
