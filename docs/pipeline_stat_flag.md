# MLPipeline with Statistics Support

The `MLPipeline` class now supports a `stat` flag in its `transform()` method, allowing you to retrieve statistics from transformers during the data processing pipeline.

## Overview

When processing XRD data, you often want to track statistics from various transformation steps (faulty pixel detection, detector joining, measurement classification, etc.). The `stat` flag enables this functionality.

## Usage

### Basic Transform (without statistics)

```python
from xrdanalysis.data_processing.pipeline import MLPipeline
from xrdanalysis.data_processing import (
    FaultyPixelDetector,
    MeasurementTypeClassifier,
)
from xrdanalysis.data_processing.transformers import DetectorJoiner

# Create pipeline with transformers
pipeline = MLPipeline(
    data_wrangling_steps=[
        ('faulty_pixel_detection', FaultyPixelDetector()),
        ('detector_joining', DetectorJoiner()),
        ('type_classification', MeasurementTypeClassifier()),
    ]
)

# Transform without statistics
df_transformed = pipeline.transform(df, stat=False)
```

### Transform with Statistics

```python
# Transform with statistics
df_transformed, stats = pipeline.transform(df, stat=True)

# Access statistics
print(stats['faulty_pixel_detection'])
print(stats['detector_joining'])
print(stats['type_classification'])
```

## Example: Three-Transformer Pipeline

Based on the Pancreas data merging workflow:

```python
import pandas as pd
from pathlib import Path
from xrdanalysis.data_processing.pipeline import MLPipeline
from xrdanalysis.data_processing import (
    FaultyPixelDetector,
    MeasurementTypeClassifier,
)
from xrdanalysis.data_processing.transformers import DetectorJoiner
from xrdanalysis.data_processing.utility_functions import h5_to_df

# Load data
folder = Path("/path/to/data")
h5_path = folder / 'data.h5'
df_calib, df = h5_to_df(h5_path)

# Create pipeline with three transformers
pipeline = MLPipeline(
    data_wrangling_steps=[
        # 1. Faulty pixel detection
        ('faulty_pixel_detection', FaultyPixelDetector(
            region_size=2,
            outlier_n_std=3.0,
            zero_frac_threshold=0.6,
            temporal_consistency=0.7,
            debug=False
        )),
        
        # 2. Detector joining
        ('detector_joining', DetectorJoiner(
            name_field='meas_name',
            calibration_mode='poni',
            interpolation_q_range={
                'SAXS': (1, 2),
                'WAXS': (3, 21.0),
            },
            debug=False,
            npt=100,
            angles=90
        )),
        
        # 3. Type classification
        ('type_classification', MeasurementTypeClassifier(
            distance_col='calculated_distance',
            output_col='type_measurement',
            waxs_threshold=0.05,
            saxs_threshold=0.1,
        )),
    ]
)

# Transform with statistics
df_transformed, stats = pipeline.transform(df, stat=True)

# Print statistics from each transformer
for step_name, step_stats in stats.items():
    print(f"\n{step_name}:")
    print(step_stats)
```

## Statistics Output

### FaultyPixelDetector Statistics

```python
{
    'image_column': 'image',
    'n_frames_prim': 150,
    'n_frames_sec': 150,
    'faulty_prim': 42,
    'faulty_sec': 38,
}
```

### DetectorJoiner Statistics

The DetectorJoiner stores statistics in the dataframe attributes:
```python
# Access via dataframe attrs
join_stats = df_transformed.attrs.get('join_stats', {})
```

### MeasurementTypeClassifier Statistics

```python
{
    'waxs_count': 75,
    'saxs_count': 60,
    'mixed_count': 10,
    'unknown_count': 5,
    'total_count': 150,
}
```

## Return Value

- **`stat=False`** (default): Returns only the transformed DataFrame
  ```python
  df_transformed = pipeline.transform(df)
  ```

- **`stat=True`**: Returns tuple of (DataFrame, statistics dictionary)
  ```python
  df_transformed, stats = pipeline.transform(df, stat=True)
  ```

## Accessing Transformer Statistics Directly

You can also access statistics directly from transformers after transformation:

```python
# Get transformers from pipeline
for step_name, transformer in pipeline.data_wrangling_steps:
    if hasattr(transformer, 'stats_'):
        print(f"{step_name} stats: {transformer.stats_}")
    elif hasattr(transformer, 'get_stats'):
        print(f"{step_name} stats: {transformer.get_stats()}")
```

## Notes

- The `stat` flag collects statistics from transformers that have either:
  - A `stats_` attribute (e.g., `FaultyPixelDetector`, `MeasurementTypeClassifier`)
  - A `get_stats()` method
  
- Transformers without these attributes/methods won't contribute to the stats dictionary

- The statistics are collected AFTER the transformation is complete, so they reflect the final state of each transformer
