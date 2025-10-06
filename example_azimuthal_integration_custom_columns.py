"""
Example demonstrating the use of AzimuthalIntegration with customizable output columns.

This example shows how to modify the output column names in both default (1D) mode
and 2D mode using the new output_column and q_range_column parameters.
"""

import numpy as np
import pandas as pd

from xrdanalysis.data_processing.transformers import AzimuthalIntegration


# Create sample data
def create_sample_data():
    """Create sample DataFrame with measurement data."""
    data = {
        "measurement_data": [np.random.rand(256, 256) for _ in range(3)],
        "center": [(128, 128)] * 3,
        "wavelength": [0.154] * 3,
        "pixel_size": [0.1] * 3,
        "calculated_distance": [0.1, 0.2, 0.3],
        "interpolation_q_range": [(0.5, 5.0)] * 3,
        "ponifile": ["sample1.poni", "sample2.poni", "sample3.poni"],
    }
    return pd.DataFrame(data)


# Example 1: Default mode with custom output column
print("Example 1: Default 1D mode with custom output column")
print("=" * 50)

# Create sample data
df = create_sample_data()

# Create AzimuthalIntegration transformer with custom output column
azint_1d_custom = AzimuthalIntegration(
    calibration_mode="dataframe",
    integration_mode="1D",
    npt=200,
    output_column="custom_radial_data",  # Custom output column name
    q_range_column="custom_q_range",  # Custom q_range column name
)

print("Original columns:", list(df.columns))
print("AzimuthalIntegration parameters:")
print(f"  - integration_mode: {azint_1d_custom.integration_mode}")
print(f"  - output_column: {azint_1d_custom.output_column}")
print(f"  - q_range_column: {azint_1d_custom.q_range_column}")

# Note: In a real scenario, you would apply the transform like this:
# result_df = azint_1d_custom.transform(df)
# print("Resulting columns:", list(result_df.columns))
# You would see columns like: [...original columns..., 'custom_q_range', 'custom_radial_data', 'calculated_distance']

print("\nExpected output columns would include:")
print(f"  - {azint_1d_custom.q_range_column}")
print(f"  - {azint_1d_custom.output_column}")
print("  - calculated_distance")

print("\n" + "=" * 50)

# Example 2: 2D mode with custom output columns
print("Example 2: 2D mode with custom output columns")
print("=" * 50)

# Create AzimuthalIntegration transformer for 2D mode
azint_2d_custom = AzimuthalIntegration(
    calibration_mode="poni",
    integration_mode="2D",
    npt=200,
    angles=180,
    output_column="polar_data",  # Custom output column for 2D data
    q_range_column="q_range_2D",  # Custom q_range column name for 2D
)

print("AzimuthalIntegration 2D parameters:")
print(f"  - integration_mode: {azint_2d_custom.integration_mode}")
print(f"  - output_column: {azint_2d_custom.output_column}")
print(f"  - q_range_column: {azint_2d_custom.q_range_column}")

print("\nExpected output columns for 2D mode would include:")
print(f"  - {azint_2d_custom.q_range_column}")
print(f"  - {azint_2d_custom.output_column}")
print("  - azimuthal_positions")
print("  - calculated_distance")

print("\n" + "=" * 50)

# Example 3: Using the code you provided
print("Example 3: Your specific use case")
print("=" * 50)

# This matches your original code but with the new customizable parameters
qX, qY = 0.5, 5.0  # Example q range values
faulty_pixels_extended = [(10, 10), (20, 20)]  # Example faulty pixels

azint2D = AzimuthalIntegration(
    calibration_mode="poni",
    faulty_pixels=faulty_pixels_extended,
    integration_mode="2D",
    npt=200,
    angles=180,
    output_column="polar_data",  # Custom name instead of 'radial_profile_data'
    q_range_column="q_range_2D",  # Custom name instead of 'q_range'
)

print("Your configuration:")
print(f"  - calibration_mode: {azint2D.calibration_mode}")
print(f"  - integration_mode: {azint2D.integration_mode}")
print(f"  - npt: {azint2D.npt}")
print(f"  - angles: {azint2D.angles}")
print(f"  - output_column: {azint2D.output_column}")
print(f"  - q_range_column: {azint2D.q_range_column}")

# In your workflow, you would do:
# dfi = dfw.copy()
# dfi['interpolation_q_range'] = [(float(qX), float(qY))] * len(dfi)
# humans2D = azint2D.transform(dfi)
#
# The resulting DataFrame would have columns:
# - 'q_range_2D' instead of 'q_range'
# - 'polar_data' instead of 'radial_profile_data'
# - 'azimuthal_positions'
# - 'calculated_distance'

print("\nAfter transformation, the DataFrame would contain:")
print(f"  - {azint2D.q_range_column} (instead of 'q_range')")
print(f"  - {azint2D.output_column} (instead of 'radial_profile_data')")
print("  - azimuthal_positions")
print("  - calculated_distance")
print("  - ...other original columns...")

print("\n" + "=" * 50)
print("Summary:")
print("The AzimuthalIntegration class now supports:")
print("1. output_column parameter - customize the main output column name")
print("2. q_range_column parameter - customize the q_range column name")
print("3. Both parameters work in 1D and 2D modes")
print("4. Default values maintain backward compatibility")
