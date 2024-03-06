"""
Test for azimuthal integration
"""

import numpy as np
import pandas as pd
import pytest
from pyFAI.detectors import Detector

from src.features.azimuthal_integration import (
    AzimuthalIntegration,
    azimuthal_integration,
    azimuthal_integration_row,
    initialize_azimuthal_integrator,
)


@pytest.fixture
def sample_row():
    return pd.Series(
        {
            "measurement_data": np.random.rand(100, 100),
            "center": (50, 50),
            "pixel_size": 0.1,
            "calculated_distance": 100,
            "wavelength": 1e-10,
        }
    )


def test_azimuthal_integration_row(sample_row):
    """
    Test azimuthal_integration_row function works on individual rows
    """

    q_range, intensity = azimuthal_integration_row(sample_row)

    assert isinstance(q_range, np.ndarray)
    assert isinstance(intensity, np.ndarray)


def test_azimuthal_integration():
    """
    Test azimuthal_integration function
    """
    detector = Detector(55e-6, 55e-6)
    ai = initialize_azimuthal_integrator(detector, 1e-10, 25, (128, 128))
    data = np.random.rand(100, 100)
    q_range, intensity = azimuthal_integration(data, ai)

    assert isinstance(q_range, np.ndarray)
    assert isinstance(intensity, np.ndarray)


def test_azimuthal_integration_transform():
    """
    Test azimuthal_integration_transform method, to see whether Transformer
    works with DataFrames
    """
    df = pd.DataFrame(
        {
            "measurement_data": [np.random.rand(100, 100) for _ in range(10)],
            "center": [(50, 50) for _ in range(10)],
            "pixel_size": [0.1 for _ in range(10)],
            "calculated_distance": [100 for _ in range(10)],
            "wavelength": [1e-10 for _ in range(10)],
        }
    )

    transformer = AzimuthalIntegration(pixel_size=0.1)
    transformed_df = transformer.transform(df)

    q_range_array_count = (
        transformed_df["q_range"]
        .apply(lambda x: isinstance(x, np.ndarray))
        .sum()
    )
    profile_array_count = (
        transformed_df["profile"]
        .apply(lambda x: isinstance(x, np.ndarray))
        .sum()
    )

    assert q_range_array_count == len(df)
    assert profile_array_count == len(df)

    assert "q_range" in transformed_df.columns
    assert "profile" in transformed_df.columns
