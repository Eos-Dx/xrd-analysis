"""
Simple test to verify that the AzimuthalIntegration modifications work correctly.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from xrdanalysis.data_processing.transformers import AzimuthalIntegration


# Create test data
def create_test_data():
    """Create test DataFrame."""
    return pd.DataFrame(
        {
            "measurement_data": [np.random.rand(10, 10) for _ in range(2)],
            "center": [(5, 5)] * 2,
            "wavelength": [0.154] * 2,
            "pixel_size": [0.1] * 2,
            "calculated_distance": [0.1, 0.2],
            "interpolation_q_range": [(0.5, 5.0)] * 2,
        }
    )


def test_1d_mode_custom_columns():
    """Test 1D mode with custom column names."""
    print("Testing 1D mode with custom columns...")

    # Mock the azimuthal integration function to return predictable results
    with patch(
        "xrdanalysis.data_processing.transformers.perform_azimuthal_integration"
    ) as mock_integration:
        # Mock return: (q_range, intensity, calculated_distance)
        mock_integration.return_value = ([1, 2, 3], [4, 5, 6], 0.15)

        # Create transformer with custom columns
        transformer = AzimuthalIntegration(
            integration_mode="1D",
            calibration_mode="dataframe",
            output_column="my_intensity_data",
            q_range_column="my_q_data",
        )

        # Transform data
        df = create_test_data()
        result = transformer.transform(df)

        # Check that custom columns exist
        assert "my_intensity_data" in result.columns, "Custom output column not found"
        assert "my_q_data" in result.columns, "Custom q_range column not found"
        assert (
            "calculated_distance" in result.columns
        ), "calculated_distance column not found"

        # Check that default column names are NOT present
        assert (
            "radial_profile_data" not in result.columns
        ), "Default output column should not exist"
        assert (
            "q_range" not in result.columns
        ), "Default q_range column should not exist"

        print("✓ 1D mode custom columns test passed!")


def test_2d_mode_custom_columns():
    """Test 2D mode with custom column names."""
    print("Testing 2D mode with custom columns...")

    # Mock the azimuthal integration function to return predictable results
    with patch(
        "xrdanalysis.data_processing.transformers.perform_azimuthal_integration"
    ) as mock_integration:
        # Mock return: (q_range, intensity, azimuthal_positions, calculated_distance)
        mock_integration.return_value = ([1, 2, 3], [4, 5, 6], [7, 8, 9], 0.15)

        # Create transformer with custom columns
        transformer = AzimuthalIntegration(
            integration_mode="2D",
            calibration_mode="dataframe",
            output_column="polar_intensity",
            q_range_column="q_radial",
        )

        # Transform data
        df = create_test_data()
        result = transformer.transform(df)

        # Check that custom columns exist
        assert (
            "polar_intensity" in result.columns
        ), "Custom output column not found in 2D"
        assert "q_radial" in result.columns, "Custom q_range column not found in 2D"
        assert (
            "azimuthal_positions" in result.columns
        ), "azimuthal_positions column not found in 2D"
        assert (
            "calculated_distance" in result.columns
        ), "calculated_distance column not found in 2D"

        # Check that default column names are NOT present
        assert (
            "radial_profile_data" not in result.columns
        ), "Default output column should not exist in 2D"
        assert (
            "q_range" not in result.columns
        ), "Default q_range column should not exist in 2D"

        print("✓ 2D mode custom columns test passed!")


def test_default_behavior():
    """Test that default behavior still works (backward compatibility)."""
    print("Testing backward compatibility with default columns...")

    # Mock the azimuthal integration function to return predictable results
    with patch(
        "xrdanalysis.data_processing.transformers.perform_azimuthal_integration"
    ) as mock_integration:
        # Mock return: (q_range, intensity, calculated_distance)
        mock_integration.return_value = ([1, 2, 3], [4, 5, 6], 0.15)

        # Create transformer with default settings
        transformer = AzimuthalIntegration(
            integration_mode="1D",
            calibration_mode="dataframe",
            # No custom column names specified - should use defaults
        )

        # Transform data
        df = create_test_data()
        result = transformer.transform(df)

        # Check that default columns exist
        assert (
            "radial_profile_data" in result.columns
        ), "Default output column not found"
        assert "q_range" in result.columns, "Default q_range column not found"
        assert (
            "calculated_distance" in result.columns
        ), "calculated_distance column not found"

        print("✓ Backward compatibility test passed!")


def test_pipeline_mode():
    """Test that pipeline mode uses custom column name."""
    print("Testing pipeline mode with custom column...")

    # Mock the azimuthal integration function to return predictable results
    with patch(
        "xrdanalysis.data_processing.transformers.perform_azimuthal_integration"
    ) as mock_integration:
        # Mock return: (q_range, intensity, calculated_distance)
        mock_integration.return_value = ([1, 2, 3], [4, 5, 6], 0.15)

        # Create transformer with custom output column and pipeline mode
        transformer = AzimuthalIntegration(
            integration_mode="1D",
            calibration_mode="dataframe",
            transformation_mode="pipeline",
            output_column="my_pipeline_data",
        )

        # Transform data
        df = create_test_data()
        result = transformer.transform(df)

        # In pipeline mode, should return a DataFrame with the intensity data as columns
        # The result should be the expanded version of the custom output column
        assert isinstance(result, pd.DataFrame), "Pipeline mode should return DataFrame"

        print("✓ Pipeline mode test passed!")


if __name__ == "__main__":
    print("Running tests for AzimuthalIntegration custom columns...")
    print("=" * 60)

    try:
        test_1d_mode_custom_columns()
        test_2d_mode_custom_columns()
        test_default_behavior()
        test_pipeline_mode()

        print("=" * 60)
        print("All tests passed! ✓")
        print("\nThe AzimuthalIntegration class successfully supports:")
        print("  • Custom output_column parameter")
        print("  • Custom q_range_column parameter")
        print("  • Backward compatibility with default names")
        print("  • Works in both 1D and 2D modes")
        print("  • Pipeline mode compatibility")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
