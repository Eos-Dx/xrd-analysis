"""
GoodnessFilter Transformer Example

This example demonstrates how to use the new GoodnessFilter transformer to clean
DataFrames by removing rows with low goodness scores, using different thresholds
for SAXS (0.5) and WAXS (0.3) measurements.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(__file__))
from src.xrdanalysis.data_processing.transformers import (
    GoodnessFilter,
    GoodnessTransformer,
)

print("=" * 60)
print("GoodnessFilter Transformer Example")
print("=" * 60)

# Create sample data with mixed SAXS and WAXS measurements
print("1. Creating sample data with SAXS and WAXS measurements...")
np.random.seed(42)

data = []
for i in range(20):
    # Create some 2D data for goodness calculation
    polar_array = np.random.rand(360, 100) * 1000 + 500

    # Alternate between SAXS and WAXS, with some having low quality
    measurement_type = "SAXS" if i % 2 == 0 else "WAXS"

    # Simulate some low-quality measurements
    quality_factor = 1.0
    if i in [3, 7, 11, 15]:  # Make some measurements lower quality
        quality_factor = 0.8
        polar_array = (
            polar_array * quality_factor + np.random.rand(*polar_array.shape) * 200
        )

    data.append(
        {
            "sample_id": f"Sample_{i+1:02d}",
            "type_measurement": measurement_type,
            "polar_data": polar_array,
        }
    )

df = pd.DataFrame(data)
print(f"   Created DataFrame with {len(df)} samples")
print(f"   SAXS samples: {len(df[df['type_measurement'] == 'SAXS'])}")
print(f"   WAXS samples: {len(df[df['type_measurement'] == 'WAXS'])}")
print()

# Step 1: Calculate goodness scores
print("2. Calculating goodness scores...")
gt = GoodnessTransformer()
df_with_goodness = gt.fit_transform(df)

# Show goodness distribution
print("   Goodness scores by type:")
for mtype in ["SAXS", "WAXS"]:
    subset = df_with_goodness[df_with_goodness["type_measurement"] == mtype]
    print(
        f"     {mtype}: mean={subset['goodness'].mean():.3f}, "
        f"min={subset['goodness'].min():.3f}, "
        f"max={subset['goodness'].max():.3f}"
    )
print()

# Step 2: Apply default filtering (WAXS > 0.3, SAXS > 0.5)
print("3. Applying default GoodnessFilter (WAXS > 0.3, SAXS > 0.5)...")
gf_default = GoodnessFilter(verbose=True)
df_filtered_default = gf_default.fit_transform(df_with_goodness)
print()

# Step 3: Apply custom filtering with different thresholds
print("4. Applying custom GoodnessFilter (WAXS > 0.2, SAXS > 0.4)...")
gf_custom = GoodnessFilter(waxs_threshold=0.2, saxs_threshold=0.4, verbose=True)
df_filtered_custom = gf_custom.fit_transform(df_with_goodness)
print()

# Step 4: Show detailed comparison
print("5. Detailed comparison:")
print("   Original data:")
for mtype in ["SAXS", "WAXS"]:
    subset = df_with_goodness[df_with_goodness["type_measurement"] == mtype]
    count = len(subset)
    avg_goodness = subset["goodness"].mean()
    print(f"     {mtype}: {count} samples, avg goodness = {avg_goodness:.3f}")

print("   After default filtering:")
for mtype in ["SAXS", "WAXS"]:
    subset = df_filtered_default[df_filtered_default["type_measurement"] == mtype]
    count = len(subset)
    if count > 0:
        avg_goodness = subset["goodness"].mean()
        print(f"     {mtype}: {count} samples, avg goodness = {avg_goodness:.3f}")
    else:
        print(f"     {mtype}: {count} samples")

print("   After custom filtering:")
for mtype in ["SAXS", "WAXS"]:
    subset = df_filtered_custom[df_filtered_custom["type_measurement"] == mtype]
    count = len(subset)
    if count > 0:
        avg_goodness = subset["goodness"].mean()
        print(f"     {mtype}: {count} samples, avg goodness = {avg_goodness:.3f}")
    else:
        print(f"     {mtype}: {count} samples")
print()

# Step 5: Pipeline usage example
print("6. Pipeline usage example:")
from sklearn.pipeline import Pipeline

# Create a pipeline that computes goodness and then filters
quality_pipeline = Pipeline(
    [
        ("compute_goodness", GoodnessTransformer()),
        (
            "filter_quality",
            GoodnessFilter(waxs_threshold=0.25, saxs_threshold=0.45, verbose=True),
        ),
    ]
)

# Apply pipeline to original data
df_pipeline_result = quality_pipeline.fit_transform(df)
print(f"   Pipeline result: {len(df_pipeline_result)} samples remaining")
print()

print("=" * 60)
print("Example completed successfully!")
print("=" * 60)
print()
print("Key features of GoodnessFilter:")
print("- Filters rows based on goodness scores")
print("- Different thresholds for SAXS (default: 0.5) and WAXS (default: 0.3)")
print("- Configurable column names and thresholds")
print("- Verbose mode for filtering statistics")
print("- Compatible with sklearn pipelines")
print("- Handles unknown measurement types conservatively")
