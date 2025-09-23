"""
GoodnessTransformer Usage Examples

This file demonstrates various usage scenarios for the updated GoodnessTransformer class,
including the new default 'polar_data' column and auto-detection functionality.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.xrdanalysis.data_processing.transformers import GoodnessTransformer


# Generate sample data for demonstration
def create_sample_data():
    """Create sample DataFrame with both polar_data and radial_profile_data columns."""
    np.random.seed(42)

    # Create sample 2D arrays (simulating XRD polar data)
    n_samples = 100
    n_angles = 360
    n_q_bins = 200

    data = []
    for i in range(n_samples):
        # Create a 2D array with some structure and noise
        polar_array = np.random.rand(n_angles, n_q_bins) * 1000 + 500
        # Add some periodic structure to make it more realistic
        for angle in range(n_angles):
            for q in range(n_q_bins):
                polar_array[angle, q] += (
                    200 * np.sin(angle * np.pi / 180) * np.cos(q * np.pi / 100)
                )

        # Create a slightly different version for radial_profile_data (legacy format)
        radial_array = polar_array * 0.9 + np.random.rand(n_angles, n_q_bins) * 100

        data.append(
            {
                "sample_id": f"sample_{i:03d}",
                "temperature": 20 + i * 0.1,  # Simulated temperature gradient
                "polar_data": polar_array,
                "radial_profile_data": radial_array,
            }
        )

    return pd.DataFrame(data)


def example_1_default_usage():
    """Example 1: Default usage with polar_data column (recommended)."""
    print("=" * 60)
    print("Example 1: Default Usage with polar_data")
    print("=" * 60)

    # Create sample data
    df = create_sample_data()
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Use GoodnessTransformer with default settings (uses polar_data)
    gt = GoodnessTransformer()
    print(f"Using column: '{gt.column}' (default)")

    # Transform the data
    df_transformed = gt.fit_transform(df)

    # Check results
    print(f"New column added: '{gt.output_col}'")
    print(f"Goodness scores (first 5): {df_transformed[gt.output_col].head().tolist()}")
    print(
        f"Goodness stats: mean={df_transformed[gt.output_col].mean():.2f}, "
        f"std={df_transformed[gt.output_col].std():.2f}"
    )
    print()


def example_2_explicit_column():
    """Example 2: Explicitly specifying the polar_data column."""
    print("=" * 60)
    print("Example 2: Explicit Column Specification")
    print("=" * 60)

    df = create_sample_data()

    # Explicitly specify polar_data column
    gt = GoodnessTransformer(column="polar_data", output_col="goodness_polar")
    print(f"Explicitly using column: '{gt.column}'")

    df_transformed = gt.fit_transform(df)

    print(f"Results stored in: '{gt.output_col}'")
    print(f"Sample goodness scores: {df_transformed[gt.output_col].head(3).tolist()}")
    print()


def example_3_legacy_column():
    """Example 3: Using legacy radial_profile_data column."""
    print("=" * 60)
    print("Example 3: Legacy radial_profile_data Column")
    print("=" * 60)

    df = create_sample_data()

    # Use legacy radial_profile_data column
    gt = GoodnessTransformer(column="radial_profile_data", output_col="goodness_legacy")
    print(f"Using legacy column: '{gt.column}'")

    df_transformed = gt.fit_transform(df)

    print(f"Results stored in: '{gt.output_col}'")
    print(f"Sample goodness scores: {df_transformed[gt.output_col].head(3).tolist()}")
    print()


def example_4_auto_detection():
    """Example 4: Auto-detection when requested column doesn't exist."""
    print("=" * 60)
    print("Example 4: Auto-Detection Fallback")
    print("=" * 60)

    # Create DataFrame with only polar_data
    df = create_sample_data()
    df_polar_only = df[["sample_id", "temperature", "polar_data"]].copy()
    print(f"DataFrame columns (polar_data only): {list(df_polar_only.columns)}")

    # Try to use radial_profile_data (doesn't exist) - should fall back to polar_data
    gt = GoodnessTransformer(column="radial_profile_data")
    print(f"Requested column: '{gt.column}' (doesn't exist)")
    print("Expected: Warning message and fallback to 'polar_data'")
    print()

    df_transformed = gt.fit_transform(df_polar_only)
    print(f"Transform successful! Results in: '{gt.output_col}'")
    print()


def example_5_advanced_parameters():
    """Example 5: Using advanced parameters with different settings."""
    print("=" * 60)
    print("Example 5: Advanced Parameter Configuration")
    print("=" * 60)

    df = create_sample_data()

    # Configure transformer with custom parameters
    gt = GoodnessTransformer(
        column="polar_data",
        skip_bins=20,  # Skip fewer low-q bins
        hf_cutoff_fraction=0.3,  # Higher frequency cutoff
        output_col="custom_goodness",
        save_dev=True,  # Save deviation matrices
        diff_col="deviation_matrices",
    )

    print("Custom configuration:")
    print(f"  Column: {gt.column}")
    print(f"  Skip bins: {gt.skip_bins}")
    print(f"  HF cutoff: {gt.hf_cutoff_fraction}")
    print(f"  Save deviations: {gt.save_dev}")

    df_transformed = gt.fit_transform(df)

    print(f"Output columns: {gt.output_col}, {gt.diff_col}")
    print(
        f"Deviation matrix shape (first sample): {df_transformed[gt.diff_col].iloc[0].shape}"
    )
    print(
        f"Custom goodness scores (first 3): {df_transformed[gt.output_col].head(3).tolist()}"
    )
    print()


def example_6_comparison():
    """Example 6: Compare results between polar_data and radial_profile_data."""
    print("=" * 60)
    print("Example 6: Column Comparison")
    print("=" * 60)

    df = create_sample_data()

    # Transform using both columns
    gt_polar = GoodnessTransformer(column="polar_data", output_col="goodness_polar")
    gt_radial = GoodnessTransformer(
        column="radial_profile_data", output_col="goodness_radial"
    )

    df_polar = gt_polar.fit_transform(df)
    df_comparison = gt_radial.fit_transform(df_polar)

    # Compare results
    correlation = df_comparison["goodness_polar"].corr(df_comparison["goodness_radial"])

    print("Comparison of goodness scores:")
    print(f"  Polar data mean: {df_comparison['goodness_polar'].mean():.3f}")
    print(f"  Radial data mean: {df_comparison['goodness_radial'].mean():.3f}")
    print(f"  Correlation: {correlation:.3f}")
    print(
        f"  Difference mean: {(df_comparison['goodness_polar'] - df_comparison['goodness_radial']).mean():.3f}"
    )
    print()


def example_7_pipeline_usage():
    """Example 7: Using GoodnessTransformer in a sklearn pipeline."""
    print("=" * 60)
    print("Example 7: Pipeline Usage")
    print("=" * 60)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = create_sample_data()

    # Create a pipeline with GoodnessTransformer
    pipeline = Pipeline(
        [
            ("goodness", GoodnessTransformer(column="polar_data")),
            # You could add other transformers here
        ]
    )

    # Transform data through pipeline
    df_transformed = pipeline.fit_transform(df)

    print("Pipeline transformation completed")
    print(f"Final columns: {list(df_transformed.columns)}")
    print(f"Goodness column statistics:")
    print(f"  Min: {df_transformed['goodness'].min():.3f}")
    print(f"  Max: {df_transformed['goodness'].max():.3f}")
    print(f"  Mean: {df_transformed['goodness'].mean():.3f}")
    print()


if __name__ == "__main__":
    print("GoodnessTransformer Usage Examples")
    print("==================================")
    print()

    # Run all examples
    example_1_default_usage()
    example_2_explicit_column()
    example_3_legacy_column()
    example_4_auto_detection()
    example_5_advanced_parameters()
    example_6_comparison()
    example_7_pipeline_usage()

    print("All examples completed successfully!")
