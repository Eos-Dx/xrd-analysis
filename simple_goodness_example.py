"""
Simple GoodnessTransformer Example - Terminal Demo
"""

import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(__file__))
from src.xrdanalysis.data_processing.transformers import GoodnessTransformer

print("=" * 50)
print("GoodnessTransformer Simple Example")
print("=" * 50)

# Create simple sample data
print("1. Creating sample XRD data...")
np.random.seed(42)

# Create a small dataset with 5 samples
data = []
for i in range(5):
    # Create 2D polar data (360 angles x 100 q-bins)
    polar_array = np.random.rand(360, 100) * 1000 + 500

    data.append(
        {
            "sample_name": f"Sample_{i+1}",
            "temperature": 25 + i * 5,
            "polar_data": polar_array,
        }
    )

df = pd.DataFrame(data)
print(f"   Created DataFrame with {len(df)} samples")
print(f"   Columns: {list(df.columns)}")
print(f"   Polar data shape: {df['polar_data'].iloc[0].shape}")
print()

# Example 1: Basic usage with default settings
print("2. Basic usage (default: polar_data column)...")
gt = GoodnessTransformer()
df_result = gt.fit_transform(df)

print(f"   Added '{gt.output_col}' column")
print("   Goodness scores:")
for i, score in enumerate(df_result["goodness"]):
    print(f"     {df_result['sample_name'].iloc[i]}: {score:.3f}")
print()

# Example 2: Custom parameters
print("3. Custom parameters example...")
gt_custom = GoodnessTransformer(
    skip_bins=10,  # Skip fewer low-q bins
    hf_cutoff_fraction=0.2,  # Lower frequency cutoff
    output_col="custom_score",
)

df_custom = gt_custom.fit_transform(df)
print(f"   Added '{gt_custom.output_col}' column with custom settings")
print("   Custom scores vs default scores:")
for i in range(len(df)):
    default_score = df_result["goodness"].iloc[i]
    custom_score = df_custom["custom_score"].iloc[i]
    print(
        f"     {df['sample_name'].iloc[i]}: {default_score:.3f} vs {custom_score:.3f}"
    )
print()

# Example 3: Auto-detection demo
print("4. Auto-detection example...")
# Create DataFrame without polar_data, only with legacy column
df_legacy = df.copy()
df_legacy["radial_profile_data"] = df_legacy[
    "polar_data"
].copy()  # Copy data to legacy column
df_legacy = df_legacy.drop("polar_data", axis=1)  # Remove polar_data

print(f"   DataFrame columns: {list(df_legacy.columns)}")
print("   Trying to use 'polar_data' (doesn't exist)...")

gt_auto = GoodnessTransformer(column="polar_data")  # This will auto-fallback
df_auto = gt_auto.fit_transform(df_legacy)

print(f"   Successfully computed goodness scores!")
print(f"   First score: {df_auto['goodness'].iloc[0]:.3f}")
print()

# Example 4: Quick comparison
print("5. Quick statistics...")
print(
    f"   Default goodness - Mean: {df_result['goodness'].mean():.3f}, Std: {df_result['goodness'].std():.3f}"
)
print(
    f"   Custom goodness  - Mean: {df_custom['custom_score'].mean():.3f}, Std: {df_custom['custom_score'].std():.3f}"
)
print()

print("=" * 50)
print("Example completed successfully!")
print("=" * 50)
