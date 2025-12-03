# Pipeline Dependency Fix Summary

## Problem Identified

The original pipeline had a **logical dependency issue**:

```python
# WRONG ORDER - DetectorJoiner came before MeasurementTypeClassifier
FaultyPixelDetector → DetectorJoiner → MeasurementTypeClassifier
```

**Why this was wrong:**
- `DetectorJoiner` needs `type_measurement` column to apply correct `interpolation_q_range` (WAXS vs SAXS)
- But `type_measurement` is created by `MeasurementTypeClassifier`
- The classifier was running LAST, so DetectorJoiner couldn't use it!

## Solution

### 1. **Reordered Pipeline Steps**

```python
# CORRECT ORDER
FaultyPixelDetector → MeasurementTypeClassifier → DetectorJoiner
```

**Rationale:**
1. **FaultyPixelDetector** - Detects and masks faulty pixels (no dependencies)
2. **MeasurementTypeClassifier** - Reads Distance from PONI files and classifies (needs PONI data)
3. **DetectorJoiner** - Uses `type_measurement` column to apply correct interpolation (needs classifier output)

### 2. **Rewrote MeasurementTypeClassifier**

#### Old Behavior
- Read from `calculated_distance` column (which may not exist yet)
- Used simple threshold parameters: `waxs_threshold=0.05`, `saxs_threshold=0.1`
- Only supported binary comparisons

#### New Behavior
- **Reads Distance directly from PONI file content** (stored in `ponifile` column)
- **Uses flexible dictionary-based rules** with operators:
  ```python
  type_rules={
      'WAXS': ('<', 0.05),
      'SAXS': ('>', 0.1),
      'INTERMEDIATE': ('between', (0.05, 0.1)),
  }
  ```
- **Supports operators:** `<`, `<=`, `>`, `>=`, `==`, `between`
- **Extracts Distance using regex** from PONI string format:
  ```
  Distance: 0.12358543419003888
  ```

### 3. **Updated Pipeline Configuration**

#### New Configuration
```python
pipeline = MLPipeline(
    data_wrangling_steps=[
        # Step 1: Detect faulty pixels
        ('faulty_pixel_detection', FaultyPixelDetector(
            region_size=2,
            outlier_n_std=3.0,
            zero_frac_threshold=0.6,
            temporal_consistency=0.7,
            debug=False
        )),
        
        # Step 2: Classify measurement type from PONI Distance
        ('type_classification', MeasurementTypeClassifier(
            poni_col='ponifile',  # Read from ponifile column
            output_col='type_measurement',
            type_rules={
                'WAXS': ('<', 0.05),
                'SAXS': ('>', 0.1),
                'INTERMEDIATE': ('between', (0.05, 0.1)),
            },
            unknown_label='UNKNOWN',
            debug=False
        )),
        
        # Step 3: Join detectors using type_measurement
        ('detector_joining', DetectorJoiner(
            name_field='meas_name',
            calibration_mode='poni',
            interpolation_q_range={
                'SAXS': (1, 2),      # Uses type_measurement='SAXS'
                'WAXS': (3, 21.0),   # Uses type_measurement='WAXS'
            },
            debug=False,
            npt=100,
            angles=90
        )),
    ]
)
```

## Changes to MeasurementTypeClassifier

### File: `/Users/sad/dev/xrd-analysis/src/xrdanalysis/data_processing/measurement_type_classifier.py`

#### New Parameters
```python
class MeasurementTypeClassifier:
    def __init__(
        self,
        poni_col: str = 'ponifile',           # NEW: column with PONI content
        output_col: str = 'type_measurement',
        type_rules: Dict = None,              # NEW: flexible rule dictionary
        unknown_label: str = 'UNKNOWN',       # NEW: label for unmatched
        debug: bool = False,                  # NEW: debug output
    ):
```

#### New Methods
1. **`_read_distance_from_poni(poni_content: str) -> float`**
   - Extracts Distance value from PONI file string using regex
   - Returns `np.nan` if not found

2. **`_apply_rule(distance, operator, threshold) -> bool`**
   - Applies classification rule with operators: `<`, `<=`, `>`, `>=`, `==`, `between`
   - Returns True if rule matches

#### New Statistics Output
```python
{
    'WAXS': 75,              # Count for each type in type_rules
    'SAXS': 60,
    'INTERMEDIATE': 10,
    'unknown_count': 5,      # Didn't match any rule
    'nan_distance_count': 2, # No Distance in PONI
    'total_count': 152,
}
```

## Changes to Pancreas_data_merging.py

### Updated Imports
```python
from xrdanalysis.data_processing.pipeline import MLPipeline
from xrdanalysis.data_processing.transformers import DetectorJoiner
```

### Updated Pipeline Cell
- Reordered transformers (FaultyPixel → Type → Detector)
- Updated MeasurementTypeClassifier configuration
- Added comments explaining the correct order

### Updated Statistics Display
- Shows correct order in markdown heading
- Displays new classifier statistics (WAXS, SAXS, INTERMEDIATE counts)
- Notes that classifier reads from PONI files

## Usage

```python
# Load data
df = joblib.load('compressed.joblib')  # Or use h5_to_df()

# Transform with pipeline
df_classified, stats = pipeline.transform(df, stat=True)

# Access statistics
print(stats['type_classification'])
# Output: {'WAXS': 75, 'SAXS': 60, 'INTERMEDIATE': 10, ...}
```

## Benefits

✅ **Correct dependency order** - Each step has what it needs from previous steps  
✅ **Reads from source** - Gets Distance directly from PONI files, not derived columns  
✅ **Flexible rules** - Easy to add new measurement types or change thresholds  
✅ **Better statistics** - Shows counts for all defined types including INTERMEDIATE  
✅ **DetectorJoiner works** - Now has `type_measurement` column to use for interpolation

## Testing

To verify the fix works:
1. Check that `type_measurement` column exists after step 2
2. Verify DetectorJoiner uses correct interpolation_q_range per type
3. Confirm statistics show WAXS/SAXS/INTERMEDIATE counts
4. Ensure no errors about missing `type_measurement` column
