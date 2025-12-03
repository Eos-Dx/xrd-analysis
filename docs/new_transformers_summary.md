# New XRD-Analysis Transformers and Visualization

## Overview

Three new components have been added to xrd-analysis to improve pancreatic cancer XRD analysis:

1. **FaultyPixelDetector** - Automatically detect faulty pixels in detectors
2. **MeasurementTypeClassifier** - Classify measurements as WAXS/SAXS/MIXED
3. **plot_measurements_by_type()** - Flexible visualization function

## 1. FaultyPixelDetector

**Location:** `xrdanalysis/data_processing/faulty_pixel_detection.py`

### Purpose
Automatically detects faulty pixels in primary and secondary detectors using multiple strategies:
- Dead/zero pixels (consistently zero across frames)
- Local region deviation (pixel vs 2×2/3×3 neighborhood blocks)
- Global statistical outliers (z-score based)
- Temporal consistency filtering

### Usage

```python
from xrdanalysis.data_processing import FaultyPixelDetector

detector = FaultyPixelDetector(
    region_size=2,
    outlier_n_std=3.0,
    zero_frac_threshold=0.6,
    temporal_consistency=0.7,
)

# Detect faulty pixels
fp_prim, fp_sec, stats = detector.detect(df, name_field='meas_name')

# Or use transform method (returns numpy arrays)
fp_prim_arr, fp_sec_arr, stats = detector.transform(df)

print(f"Primary faulty pixels: {len(fp_prim)}")
print(f"Secondary faulty pixels: {len(fp_sec)}")
```

### Parameters
- `region_size` (int): Block size for local checks (2 or 3)
- `outlier_n_std` (float): Z-score threshold for outliers
- `zero_frac_threshold` (float): Fraction of frames to call pixel dead
- `temporal_consistency` (float): Fraction of frames pixel must be abnormal

### Output
- `faulty_prim`, `faulty_sec`: Sets of (i, j) tuples
- `stats`: Dictionary with detection counts and metadata

## 2. MeasurementTypeClassifier

**Location:** `xrdanalysis/data_processing/measurement_type_classifier.py`

### Purpose
Classifies measurements as WAXS, SAXS, or MIXED based on calculated_distance.

### Usage

```python
from xrdanalysis.data_processing import MeasurementTypeClassifier

classifier = MeasurementTypeClassifier(
    distance_col='calculated_distance',
    output_col='type_measurement',
    waxs_threshold=0.05,
    saxs_threshold=0.1,
)

# Classify measurements
df_classified = classifier.fit_transform(df)

# Get statistics
stats = classifier.get_stats()
```

### Classification Rules
- **WAXS**: distance < 0.05 m
- **SAXS**: distance > 0.1 m
- **MIXED**: 0.05 ≤ distance ≤ 0.1 m (rejected)

### Parameters
- `distance_col`: Column with detector distance values
- `output_col`: Output column name
- `waxs_threshold`: Upper threshold for WAXS classification
- `saxs_threshold`: Lower threshold for SAXS classification
- `mixed_label`: Label for intermediate measurements

### Output
- Modified DataFrame with `type_measurement` column
- `.get_stats()` returns classification counts

## 3. plot_measurements_by_type()

**Location:** `xrdanalysis/visualization/measurement_plots.py`

### Purpose
Flexible visualization of measurements grouped by type with optional label-based coloring.

### Usage

```python
from xrdanalysis.visualization.measurement_plots import plot_measurements_by_type

# Basic usage
fig = plot_measurements_by_type(df)

# With custom coloring
color_map = {'HUMAN': 'red', 'MOUSE': 'blue', 'SHEEP': 'green'}
fig = plot_measurements_by_type(
    df,
    label_col='species',
    color_map=color_map,
    xlim_map={'WAXS': (0, 20), 'SAXS': (0.5, 2.0)},
    title_prefix='Grant 2 Pancreas'
)

# Auto-generated colors (no color_map provided)
fig = plot_measurements_by_type(
    df,
    label_col='species',
    # color_map=None  # Will auto-generate
)
```

### Parameters
- `type_col`: Column with measurement types (default: 'type_measurement')
- `q_col`: X-axis column (default: 'q_range')
- `y_col`: Y-axis column (default: 'radial_profile_data')
- `label_col`: Optional column for curve coloring
- `color_map`: Optional dict mapping labels to colors
- `title_prefix`: Title prefix for subplots
- `xlim_map`: Dict mapping type to (xmin, xmax) limits
- `logy`: Use log scale for y-axis
- `figsize_per_type`: Figure size per subplot

### Features
- ✅ Dynamic subplots (one per measurement type)
- ✅ Auto color generation or custom color mapping
- ✅ Per-type x-axis limits
- ✅ Flexible for any number of measurement types
- ✅ Auto legend showing color meanings

## Integration with xrd-analysis Pipeline

All transformers follow sklearn conventions and can be used in pipelines:

```python
from sklearn.pipeline import Pipeline
from xrdanalysis.data_processing import MeasurementTypeClassifier

pipeline = Pipeline([
    ('classifier', MeasurementTypeClassifier()),
    # ... additional steps
])

df_result = pipeline.fit_transform(df)
```

## Example Notebooks

See `Pancreas_pipeline_example.py` for a complete example showing:
1. Loading pancreatic cancer data
2. Detecting faulty pixels
3. Classifying measurements
4. Visualizing results

## Files Modified/Created

### Created
- `/src/xrdanalysis/data_processing/faulty_pixel_detection.py`
- `/src/xrdanalysis/data_processing/measurement_type_classifier.py`
- `/src/xrdanalysis/visualization/measurement_plots.py` (already existed)
- Example notebook: `Pancreas_pipeline_example.py`

### Modified
- `/src/xrdanalysis/data_processing/__init__.py` (added imports)

## Benefits

1. **Automatic faulty pixel detection** - No need for hardcoded pixel lists
2. **Per-detector masks** - Primary and secondary detectors use different masks
3. **Flexible classification** - Configurable thresholds for WAXS/SAXS
4. **Professional visualization** - Publication-ready plots with customization
5. **Pipeline integration** - Use with sklearn pipelines for reproducible workflows
