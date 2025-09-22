"""
Advanced GoodnessFilter Transformer Example

This example demonstrates the updated GoodnessFilter with dictionary-based
thresholds and configurable comparison rules.
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

print("=" * 70)
print("Advanced GoodnessFilter Transformer Example")
print("=" * 70)

# Create sample data with varied goodness scores
print("1. Creating sample data with varied goodness scores...")
np.random.seed(42)

data = []
for i in range(20):
    # Create some 2D data for goodness calculation
    polar_array = np.random.rand(360, 100) * 1000 + 500

    # Alternate between SAXS and WAXS
    measurement_type = "SAXS" if i % 2 == 0 else "WAXS"

    # Create some variation in quality to get different goodness scores
    if i in [2, 3, 6, 7]:  # Some lower quality
        polar_array = polar_array * 0.7 + np.random.rand(*polar_array.shape) * 300
    elif i in [10, 11, 14, 15]:  # Some medium quality
        polar_array = polar_array * 0.85 + np.random.rand(*polar_array.shape) * 200

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

# Calculate goodness scores
print("2. Calculating goodness scores...")
gt = GoodnessTransformer()
df_with_goodness = gt.fit_transform(df)

# Show goodness distribution
print("   Goodness score distribution:")
for mtype in ["SAXS", "WAXS"]:
    subset = df_with_goodness[df_with_goodness["type_measurement"] == mtype]
    scores = subset["goodness"].values
    print(f"     {mtype}: {scores}")
print()

# Example 1: Dictionary thresholds with '>' rule (keep good quality)
print("3. Example 1: Dictionary thresholds with '>' rule...")
print("   Keep samples with goodness > threshold")
thresholds_dict = {"SAXS": 50, "WAXS": 30}
gf1 = GoodnessFilter(thresholds=thresholds_dict, rule=">", verbose=True)
df_filtered_1 = gf1.fit_transform(df_with_goodness)
print()

# Example 2: Dictionary thresholds with '<' rule (keep poor quality)
print("4. Example 2: Dictionary thresholds with '<' rule...")
print("   Keep samples with goodness < threshold (for analysis of poor quality)")
thresholds_dict_low = {
    "SAXS": 85,
    "WAXS": 85,
}  # Higher thresholds to catch some samples
gf2 = GoodnessFilter(thresholds=thresholds_dict_low, rule="<", verbose=True)
df_filtered_2 = gf2.fit_transform(df_with_goodness)
print()

# Example 3: Mixed measurement types with default threshold
print("5. Example 3: Mixed measurement types with default threshold...")
# Add some unknown measurement type
df_mixed = df_with_goodness.copy()
df_mixed.loc[df_mixed.index[-2:], "type_measurement"] = "UNKNOWN"

gf3 = GoodnessFilter(
    thresholds={"SAXS": 75, "WAXS": 78},  # UNKNOWN will use default_threshold
    rule=">=",
    default_threshold=80,  # Default for unknown types
    verbose=True,
)
df_filtered_3 = gf3.fit_transform(df_mixed)
print()

# Example 4: Your specific use case - filter out poor quality
print("6. Example 4: Your specific use case...")
print("   Remove samples with goodness <= threshold (quality control)")
your_thresholds = {"SAXS": 50, "WAXS": 30}
gf_quality_control = GoodnessFilter(
    thresholds=your_thresholds,
    rule=">",  # Keep samples with goodness > threshold
    verbose=True,
)
df_quality_filtered = gf_quality_control.fit_transform(df_with_goodness)
print()

# Example 5: Pipeline with custom thresholds
print("7. Example 5: Pipeline with custom processing...")
from sklearn.pipeline import Pipeline

# Complete pipeline: compute goodness -> filter quality
quality_pipeline = Pipeline(
    [
        ("compute_goodness", GoodnessTransformer(column="polar_data")),
        (
            "filter_quality",
            GoodnessFilter(
                thresholds={"SAXS": 60, "WAXS": 40}, rule=">=", verbose=True
            ),
        ),
    ]
)

df_pipeline_result = quality_pipeline.fit_transform(df)
print(f"   Final pipeline result: {len(df_pipeline_result)} samples")
print()

# Show detailed results comparison
print("8. Detailed Results Summary:")
print(f"   Original data: {len(df_with_goodness)} samples")
print(f"   After quality filter (>): {len(df_filtered_1)} samples")
print(f"   After poor quality filter (<): {len(df_filtered_2)} samples")
print(f"   After mixed types filter (>=): {len(df_filtered_3)} samples")
print(f"   After your filter (>): {len(df_quality_filtered)} samples")
print(f"   After pipeline: {len(df_pipeline_result)} samples")
print()

print("=" * 70)
print("Advanced Example Completed Successfully!")
print("=" * 70)
print()
print("Key Features Demonstrated:")
print("✓ Dictionary-based thresholds: {'SAXS': 50, 'WAXS': 30}")
print("✓ Comparison rules: '>', '>=', '<', '<='")
print("✓ Default threshold for unknown measurement types")
print("✓ Verbose statistics showing filtering details")
print("✓ Pipeline integration")
print("✓ Flexible configuration for different use cases")
print()
print("Your Use Case:")
print("gf = GoodnessFilter(")
print("    thresholds={'SAXS': 50, 'WAXS': 30},")
print("    rule='>', # Keep samples with goodness > threshold")
print("    verbose=True")
print(")")
print("df_clean = gf.fit_transform(df_with_goodness)")
