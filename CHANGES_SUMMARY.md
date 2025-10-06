# AzimuthalIntegration Class Modifications

## Summary

The `AzimuthalIntegration` class in `src/xrdanalysis/data_processing/transformers.py` has been modified to support customizable output column names. This allows users to specify custom names for the output columns in both 1D and 2D integration modes.

## Changes Made

### 1. New Parameters Added

- **`output_column`** (str, default: `"radial_profile_data"`)
  - Customizes the name of the main output column containing integration results
  - In 1D mode: contains the integrated intensity data
  - In 2D mode: contains the 2D polar data

- **`q_range_column`** (str, default: `"q_range"`)
  - Customizes the name of the q-range output column
  - Contains the q-values (momentum transfer) for the integration

### 2. Modified Functions

- **`_map_1d()` function**: Updated to use `self.output_column` and `self.q_range_column` instead of hardcoded column names
- **`_map_2d()` function**: Updated to use `self.output_column` and `self.q_range_column` instead of hardcoded column names
- **Pipeline mode**: Updated to use `self.output_column` when returning data in pipeline transformation mode

### 3. Documentation Updates

- Updated class docstring to document the new parameters
- Updated `transform()` method docstring to mention customizable column names
- Maintained backward compatibility by using the original column names as defaults

## Usage Examples

### Example 1: Default Behavior (Backward Compatible)
```python
# Original behavior - no changes needed
azint = AzimuthalIntegration(
    calibration_mode='poni',
    integration_mode='1D',
    npt=200
)
# Output columns: 'q_range', 'radial_profile_data', 'calculated_distance'
```

### Example 2: Custom Column Names in 1D Mode
```python
azint = AzimuthalIntegration(
    calibration_mode='dataframe',
    integration_mode='1D',
    npt=200,
    output_column='intensity_data',
    q_range_column='momentum_transfer'
)
# Output columns: 'momentum_transfer', 'intensity_data', 'calculated_distance'
```

### Example 3: Custom Column Names in 2D Mode (Your Use Case)
```python
azint2D = AzimuthalIntegration(
    calibration_mode='poni',
    faulty_pixels=faulty_pixels_extended,
    integration_mode='2D',
    npt=200,
    angles=180,
    output_column='polar_data',       # Custom name instead of 'radial_profile_data'
    q_range_column='q_range_2D'      # Custom name instead of 'q_range'
)

# Usage in your workflow:
dfi = dfw.copy()
dfi['interpolation_q_range'] = [(float(qX), float(qY))] * len(dfi)
humans2D = azint2D.transform(dfi)

# Output columns: 'q_range_2D', 'polar_data', 'azimuthal_positions', 'calculated_distance'
```

## Benefits

1. **Flexibility**: Users can now customize output column names to match their specific workflows
2. **Clarity**: More descriptive column names can be used (e.g., `'polar_data'` instead of `'radial_profile_data'` for 2D data)
3. **Backward Compatibility**: Existing code continues to work without modification
4. **Consistency**: Both q_range and output columns can be customized consistently

## Files Modified

- `src/xrdanalysis/data_processing/transformers.py`: Main implementation
- Added example files:
  - `example_azimuthal_integration_custom_columns.py`: Usage examples
  - `test_custom_columns.py`: Test verification
  - `CHANGES_SUMMARY.md`: This summary

## Testing

All functionality has been tested including:
- ✅ 1D mode with custom columns
- ✅ 2D mode with custom columns
- ✅ Backward compatibility with default column names
- ✅ Pipeline mode compatibility
- ✅ Proper column mapping in all integration modes

The modifications are production-ready and maintain full backward compatibility.
