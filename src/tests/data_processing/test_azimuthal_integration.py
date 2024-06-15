"""Tests for azimuthal integration.py"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from xrdanalysis.data_processing.azimuthal_integration import (
    initialize_azimuthal_integrator_df,
    initialize_azimuthal_integrator_poni,
    perform_azimuthal_integration,
)


@patch(("xrdanalysis.data_processing.azimuthal_integration.Detector"))
@patch("xrdanalysis.data_processing.azimuthal_integration.AzimuthalIntegrator")
def test_initialize_azimuthal_integrator_df(
    mock_azimuthal_integrator, mock_detector
):
    """Test azimuthal integrator creation from a dataframe"""
    pixel_size = 0.0001
    center_column = 1024
    center_row = 1024
    wavelength = 1.54
    sample_distance_mm = 1000

    # Mock instance
    ai_mock_instance = mock_azimuthal_integrator.return_value
    ai_mock_instance.setFit2D.return_value = None

    # Call the function to test
    result = initialize_azimuthal_integrator_df(
        pixel_size, center_column, center_row, wavelength, sample_distance_mm
    )

    # Assertions to check if the Detector and AzimuthalIntegrator were
    # called correctly
    mock_detector.assert_called_once_with(pixel_size, pixel_size)
    mock_azimuthal_integrator.assert_called_once_with(
        detector=mock_detector.return_value
    )
    ai_mock_instance.setFit2D.assert_called_once_with(
        sample_distance_mm, center_column, center_row, wavelength=wavelength
    )
    assert result == ai_mock_instance


@patch("pyFAI.load", autospec=True)
def test_initialize_azimuthal_integrator_poni(mock_pyfai_load):
    """Test azimuthal integrator creation from a poni file"""
    poni_file = "path/to/poni/file.poni"
    ai_mock_instance = MagicMock(spec=AzimuthalIntegrator)
    mock_pyfai_load.return_value = ai_mock_instance

    result = initialize_azimuthal_integrator_poni(poni_file)

    mock_pyfai_load.assert_called_once_with(poni_file)
    assert result == ai_mock_instance


class TestAzimuthalIntegration(unittest.TestCase):
    """Tests for perform_azimuthal_integration"""

    @patch(
        (
            "xrdanalysis.data_processing."
            "azimuthal_integration.initialize_azimuthal_integrator_df"
        )
    )
    def test_azimuthal_integration_dataframe(self, mock_initialize_ai_df):
        """Test azimuthal integrator through DF"""
        # Mocked integrator object
        mock_ai = mock_initialize_ai_df.return_value
        mock_ai.dist = np.array([0.1, 0.1, 0.1])
        mock_ai.integrate1d.return_value = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        )

        row = pd.Series(
            {
                "measurement_data": np.random.rand(10, 10),
                "center": (5, 5),
                "wavelength": 1.54,
                "calculated_distance": 0.1,
                "pixel_size": 100,
                "interpolation_q_range": (0, 5),
            }
        )

        radial, intensity, dist = perform_azimuthal_integration(row)

        mock_initialize_ai_df.assert_called_once()
        mock_ai.integrate1d.assert_called_once()
        np.testing.assert_array_equal(radial, np.array([1, 2, 3]))
        np.testing.assert_array_equal(intensity, np.array([4, 5, 6]))
        np.testing.assert_array_equal(dist, np.array([0.1, 0.1, 0.1]))

    @patch(
        (
            "xrdanalysis.data_processing."
            "azimuthal_integration.initialize_azimuthal_integrator_poni"
        )
    )
    def test_azimuthal_integration_poni(self, mock_initialize_ai_poni):
        """Test azimuthal integrator through poni"""
        # Mocked integrator object
        mock_ai = mock_initialize_ai_poni.return_value
        mock_ai.dist = np.array([125, 126, 127])
        mock_ai.integrate1d.return_value = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        )

        row = pd.Series(
            {
                "measurement_data": np.random.rand(10, 10),
                "calibration_measurement_id": 123,
                "ponifile": "test.poni",
                "interpolation_q_range": (0, 5),
            }
        )

        radial, intensity, dist = perform_azimuthal_integration(
            row, calibration_mode="poni", poni_dir="/path/to/poni"
        )

        mock_initialize_ai_poni.assert_called_once_with(
            "/path/to/poni/123.poni"
        )
        mock_ai.integrate1d.assert_called_once()
        np.testing.assert_array_equal(radial, np.array([1, 2, 3]))
        np.testing.assert_array_equal(intensity, np.array([4, 5, 6]))
        np.testing.assert_array_equal(dist, np.array([125, 126, 127]))

    @patch(
        (
            "xrdanalysis.data_processing."
            "azimuthal_integration.initialize_azimuthal_integrator_df"
        )
    )
    def test_azimuthal_integration_2d_mode(self, mock_initialize_ai_df):
        """Test azimuthal integrator in 2D"""
        # Mocked integrator object
        mock_ai = mock_initialize_ai_df.return_value
        mock_ai.dist = np.array([0.1, 0.1, 0.1])
        mock_ai.integrate2d.return_value = (
            np.array([4, 5, 6]),
            np.array([1, 2, 3]),
            np.array([7, 8, 9]),
        )

        row = pd.Series(
            {
                "measurement_data": np.random.rand(10, 10),
                "center": (5, 5),
                "wavelength": 1.54,
                "calculated_distance": 0.1,
                "pixel_size": 100,
                "interpolation_q_range": (0, 5),
            }
        )

        radial, intensity, azimuthal, dist = perform_azimuthal_integration(
            row, mode="2D"
        )

        mock_initialize_ai_df.assert_called_once()
        mock_ai.integrate2d.assert_called_once()
        np.testing.assert_array_equal(radial, np.array([1, 2, 3]))
        np.testing.assert_array_equal(intensity, np.array([4, 5, 6]))
        np.testing.assert_array_equal(azimuthal, np.array([7, 8, 9]))
        np.testing.assert_array_equal(dist, np.array([0.1, 0.1, 0.1]))

    @patch(
        (
            "xrdanalysis.data_processing."
            "azimuthal_integration.initialize_azimuthal_integrator_df"
        )
    )
    def test_azimuthal_integration_with_mask(self, mock_initialize_ai_df):
        """Test azimuthal integrator with mask"""
        # Mocked integrator object
        mock_ai = mock_initialize_ai_df.return_value
        mock_ai.dist = np.array([0.1, 0.1, 0.1])
        mock_ai.integrate1d.return_value = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        )

        row = pd.Series(
            {
                "measurement_data": np.random.rand(10, 10),
                "center": (5, 5),
                "wavelength": 1.54,
                "calculated_distance": 0.1,
                "pixel_size": 100,
                "interpolation_q_range": (0, 5),
            }
        )

        mask = np.random.randint(0, 2, (10, 10))

        radial, intensity, dist = perform_azimuthal_integration(row, mask=mask)

        mock_initialize_ai_df.assert_called_once()
        mock_ai.integrate1d.assert_called_once_with(
            row["measurement_data"], 256, radial_range=(0, 5), mask=mask
        )
        np.testing.assert_array_equal(radial, np.array([1, 2, 3]))
        np.testing.assert_array_equal(intensity, np.array([4, 5, 6]))
        np.testing.assert_array_equal(dist, np.array([0.1, 0.1, 0.1]))

    @patch(
        (
            "xrdanalysis.data_processing."
            "azimuthal_integration.initialize_azimuthal_integrator_df"
        )
    )
    def test_azimuthal_integration_missing_optional_fields(
        self, mock_initialize_ai_df
    ):
        """Test azimuthal integrator without optional fields
        (are the default arguments provided?)"""
        # Mocked integrator object
        mock_ai = mock_initialize_ai_df.return_value
        mock_ai.dist = np.array([0.1, 0.1, 0.1])
        mock_ai.integrate1d.return_value = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        )

        row = pd.Series(
            {
                "measurement_data": np.random.rand(10, 10),
                "center": (5, 5),
                "wavelength": 1.54,
                "calculated_distance": 0.1,
                "pixel_size": 100,
            }
        )

        radial, intensity, dist = perform_azimuthal_integration(row)

        mock_initialize_ai_df.assert_called_once()
        mock_ai.integrate1d.assert_called_once_with(
            row["measurement_data"], 256, radial_range=None, mask=None
        )
        np.testing.assert_array_equal(radial, np.array([1, 2, 3]))
        np.testing.assert_array_equal(intensity, np.array([4, 5, 6]))
        np.testing.assert_array_equal(dist, np.array([0.1, 0.1, 0.1]))
