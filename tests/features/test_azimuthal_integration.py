"""
Test for azimuthal integration
"""

import numpy as np
import pandas as pd
import pytest

from src.features.azimuthal_integration import (
    AzimuthalIntegration,
    azimuthal_integration,
)


@pytest.mark.parametrize(
    "center,array,expected",
    [
        ((0, 0), np.array([[np.nan]]), np.array([0])),
        ((0, 0), np.array([[1, 2], [3, np.nan]]), np.array([1, 1.6666667])),
        (
            (0, 0),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, np.nan]]),
            np.array([1, 3.6666667, 6, 0]),
        ),  # Example case
        (
            (2, 2),
            np.array(
                [
                    [np.nan, 2, 3, 4, 5],
                    [6, 7, 8, np.nan, 10],
                    [np.nan, 12, 13, 14, np.nan],
                    [16, np.nan, 18, 19, 20],
                    [21, 22, 23, 24, np.nan],
                ]
            ),
            np.array([13, 9.75, 10.833333, 6.5]),
        ),
        # Add more cases as needed
    ],
)
def test_azimuthal_integration_parameterized_nans(center, array, expected):
    """
    Test azimuthal integration handling of NaN values within the input array.

    This test verifies that the azimuthal_integration function correctly
    processes arrays containing NaN values, ensuring they are
    replaced with zeroes.
    """
    result = azimuthal_integration(array, center)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "center,array,expected",
    [
        ((0, 0), np.array([[1]]), np.array([1])),
        ((0, 0), np.array([[1, 2], [3, 4]]), np.array([1, 3])),
        (
            (0, 0),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([1, 3.6666667, 6, 9]),
        ),  # Example case
        (
            (2, 2),
            np.array(
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                ]
            ),
            np.array([13, 13, 13, 13]),
        ),
        # Add more cases as needed
    ],
)
def test_azimuthal_integration_parameterized(center, array, expected):
    """
    Test azimuthal integration with various input arrays and centers.

    This test checks the azimuthal_integration function against
    a set of parameterized cases to ensure accurate computation
    of azimuthal integrated values across different data setups
    and center points.
    """
    result = azimuthal_integration(array, center)
    np.testing.assert_array_almost_equal(result, expected)


def test_AzimuthalIntegration_transform():
    """
    Test the transform method of the AzimuthalIntegration class.

    Ensures that the transform method correctly adds a 'profile'
    column to a DataFrame containing measurement data and
    centers, indicating successful application of azimuthal
    integration to each data row.
    """
    data = {
        "measurement_data": [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
        ],
        "center": [(0, 0), (1, 1)],
    }
    df = pd.DataFrame(data)
    transformer = AzimuthalIntegration()
    transformed_df = transformer.transform(df)

    # Check if 'profile' column is added
    assert "profile" in transformed_df.columns


def test_transform_input_validation():
    """
    Test input validation within the transform
    method of AzimuthalIntegration.

    Verifies that the transform method raises a
    TypeError when provided with input that is not a pandas
    DataFrame, ensuring the method's robustness and correct error handling.
    """
    transformer = AzimuthalIntegration()
    with pytest.raises(TypeError):
        transformer.transform(np.array([1, 2, 3]))
