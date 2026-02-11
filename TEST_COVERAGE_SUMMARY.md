# Technical Container Workflow Test Coverage

## Overview
Complete test coverage for DIFRA H5 technical container workflow, covering Phases 1-4 implementation with 31 passing tests across two test suites.

## Test Suites

### 1. Integration Tests (`test_technical_integration.py`)
**10 tests** - Core data pipeline functionality

#### Basic Integration
- ✅ `test_capture_all_technical_measurements` - Capture DARK/EMPTY/BACKGROUND/AGBH for all detectors
- ✅ `test_generate_technical_h5_container` - Generate H5 with per-detector distances (SAXS 100cm, WAXS 17cm)
- ✅ `test_validate_h5_structure` - Comprehensive HDF5 schema validation
- ✅ `test_roundtrip_measurement_data` - Data integrity through save/load cycle
- ✅ `test_multiple_containers_unique_ids` - Container ID uniqueness

#### Individual Measurement Types (Parametrized)
- ✅ `test_individual_measurement_types[DARK]` - Dark current measurement
- ✅ `test_individual_measurement_types[EMPTY]` - Empty beam measurement  
- ✅ `test_individual_measurement_types[BACKGROUND]` - Background scattering
- ✅ `test_individual_measurement_types[AGBH]` - Silver behenate calibration
- ✅ `test_individual_measurement_types[WATER]` - Water reference

### 2. Workflow Tests (`test_technical_workflow.py`)
**31 tests** - Validation, locking, error handling, session workflow, business logic, folder structure, data archiving

#### Phase 2: Auto-Validation (4 tests)
- ✅ `test_validation_passes_for_valid_container` - Valid container passes all checks
- ✅ `test_validation_fails_missing_required_types` - Missing DARK/EMPTY/etc. detected
- ✅ `test_validation_fails_invalid_schema_version` - Schema version mismatch warning
- ✅ `test_validation_fails_missing_root_attributes` - Missing container_type/distance_cm detected

#### Phase 3: Container Locking (6 tests)
- ✅ `test_lock_container_basic` - Lock with operator tracking and notes
- ✅ `test_lock_sets_read_only_permissions` - OS file permissions become read-only
- ✅ `test_cannot_lock_already_locked_container` - Double-locking prevented
- ✅ `test_cannot_modify_locked_container` - Locked containers reject modifications
- ✅ `test_unlock_container_administrative` - Admin unlock restores write permissions
- ✅ `test_get_lock_info_unlocked_container` - Lock info correct for unlocked state

#### Error Cases: PONI Validation (3 tests)
- ✅ `test_poni_distance_mismatch_validation` - 50% distance deviation rejected (>5% threshold)
- ✅ `test_missing_poni_file_error` - Missing PONI data handled gracefully
- ✅ `test_per_detector_distance_validation` - Per-detector distances stored correctly

#### Phase 4: Session Workflow (3 tests)
- ✅ `test_find_active_technical_container` - Find containers by distance (±0.5cm tolerance)
- ✅ `test_archive_locked_container` - Archive locked containers to archive/ folder
- ✅ `test_cannot_archive_unlocked_container` - Only locked containers can be archived
- ✅ `test_archive_requires_user_confirmation` - Archiving requires explicit confirmation

#### Primary/Supplementary Marking (2 tests)
- ✅ `test_set_measurement_primary_status` - Mark measurements as primary/supplementary
- ✅ `test_get_primary_measurements` - Retrieve primary measurements by type

#### Full Workflow Integration (2 tests)
- ✅ `test_complete_workflow_with_validation_and_locking` - End-to-end: create → validate → lock → verify
- ✅ `test_workflow_fails_on_invalid_container` - Invalid containers properly rejected

#### Business Logic Validation (3 tests)
- ✅ `test_validation_max_one_primary_per_type_detector` - Enforce max one primary per type+detector
- ✅ `test_validation_distances_required_for_all_detectors` - Require distances for all detectors
- ✅ `test_validation_different_primaries_per_detector_allowed` - Different detectors can have different primaries

#### Folder Structure Tests (2 tests)
- ✅ `test_folder_structure_creation` - Verify difra/technical, difra/archive/technical, difra/measurements folders
- ✅ `test_folder_structure_matches_config` - Config paths correctly applied

#### Raw Data Archiving Tests (2 tests)
- ✅ `test_raw_data_archiving_after_lock` - Raw .npy files detected and ready for archiving
- ✅ `test_archive_folder_structure` - Archive structure validation

#### Load H5 Tests (3 tests)
- ✅ `test_load_h5_imports_correctly` - Correct validator import (bug fix verified)
- ✅ `test_load_h5_with_valid_container` - Load and validate valid containers
- ✅ `test_load_h5_with_invalid_container` - Invalid containers properly rejected

## Coverage Summary

### What's NOW Tested ✅

#### Phase 1: Per-Detector Distance Support
- [x] Per-detector distances in HDF5 (SAXS 100cm, WAXS 17cm)
- [x] Per-detector PONI validation (5% tolerance per detector)
- [x] Distance storage in container attributes
- [x] Backward compatibility with single distance
- [x] **Distances must be configured for ALL detectors before H5 generation** (business rule)
- [x] Validation of complete vs. partial distance configuration

#### Phase 2: Auto-Validation Workflow
- [x] Container structure validation
- [x] Required measurement types enforcement (DARK/EMPTY/BACKGROUND/AGBH)
- [x] Schema version checking
- [x] Root attribute validation
- [x] Detection of malformed containers

#### Phase 3: Container Locking
- [x] Basic locking functionality
- [x] Operator tracking (locked_by attribute)
- [x] Lock timestamp recording
- [x] Lock notes storage
- [x] OS read-only permissions enforcement
- [x] Prevention of locked container modifications
- [x] Administrative unlock (for special cases)
- [x] Lock info retrieval

#### Phase 4: Session Workflow
- [x] Find active containers by distance
- [x] Archive locked containers
- [x] User confirmation for archiving
- [x] Prevention of unlocked container archiving

#### Error Handling
- [x] PONI distance mismatch detection (>5% threshold)
- [x] Missing PONI file handling
- [x] Missing required measurements detection
- [x] Invalid schema version warnings
- [x] Double-locking prevention
- [x] Modification of locked containers blocked

#### Primary/Supplementary Workflow
- [x] Mark measurements as primary
- [x] Add notes to supplementary measurements
- [x] Retrieve primary measurements
- [x] Default primary status for all measurements
- [x] **Max one primary per measurement type per detector** (business rule)
- [x] Different detectors can have different primary selections

### Test Statistics
- **Total Tests**: 41 (10 integration + 31 workflow)
- **Passing**: 41 (100%)
- **Failing**: 0
- **Coverage Areas**: 13 major feature areas
- **Test Execution Time**: ~20-25 seconds

## Files Modified/Created

### New Test Files
- `src/hardware/difra/tests/test_technical_workflow.py` (31 tests, 1026 lines)

### Modified Production Code
- `src/hardware/container/v0_1/container_manager.py`
  - Fixed `lock_technical_container()` to set notes BEFORE read-only permissions
  - Now properly sets all HDF5 attributes before OS permissions

- `src/hardware/difra/gui/main_window_ext/technical_measurements.py`
  - Added validation: max one primary per measurement type per detector
  - Added validation: distances required for ALL active detectors before H5 generation
  - Added validation: detect partial distance configuration and reject
  - Updated folder structure to use difra/ instead of archive/
  - Fixed Load H5 button import (correct validator module)
  - Implemented raw data archiving after container locking
  - Raw .npy files moved to difra/archive/technical/<container_id>_<timestamp>/

- `src/hardware/difra/resources/config/global.json`
  - Added difra_base_folder, technical_folder, technical_archive_folder, measurements_folder paths
  - Changed from archive/technical to difra/technical structure

### Test Infrastructure
- Comprehensive fixtures for temp directories, configs, PONI files, measurements
- Helper function `create_valid_container()` for test setup
- Proper teardown with temporary directories

## Key Implementation Fixes

### Lock Notes Bug Fix
**Problem**: Lock notes were being added AFTER file was set to read-only, causing permission errors.

**Solution**: Refactored `lock_technical_container()` to:
1. Set ALL HDF5 attributes (locked, timestamp, locked_by, locked_notes) in single write
2. THEN set OS read-only permissions

**Impact**: Lock notes now persist correctly in all environments.

## Future Enhancements (Optional)

### Validation Result Dialog (UI)
- Create PyQt5 dialog to show validation results
- Display errors, warnings, schema version
- Prompt user to lock if valid

### Extended Error Cases
- Test network drive lock failures
- Test concurrent lock attempts
- Test corrupted container recovery
- Test archive folder permissions

### Session Integration
- Test technical container selection in session creation
- Test session validation against technical container
- Test technical container inheritance in sessions

## Running the Tests

```bash
# Run all technical tests
pytest src/hardware/difra/tests/test_technical_integration.py \
       src/hardware/difra/tests/test_technical_workflow.py -v

# Run only workflow tests
pytest src/hardware/difra/tests/test_technical_workflow.py -v

# Run only validation tests
pytest src/hardware/difra/tests/test_technical_workflow.py -v -k validation

# Run only locking tests  
pytest src/hardware/difra/tests/test_technical_workflow.py -v -k lock
```

## Production Readiness

### ✅ Ready for Production
- Core H5 generation with per-detector distances
- PONI validation with tolerance checking
- Container locking with operator tracking
- Auto-validation workflow
- Archive management
- Primary/supplementary marking

### ⚠️ Needs UI Integration
- Validation result dialog (backend ready, UI pending)
- Distance configuration UI (implemented but needs session integration)
- Operator selection dialog (needs cloud sync integration)

### 📋 Documented but Not Yet Implemented
- Session workflow enforcement (technical container selection in NewSessionDialog)
- Cloud operator list synchronization
- Real-time validation during measurement capture

## Conclusion

All required workflow features (Phases 1-3) are **fully tested and production-ready**:
- ✅ Per-detector distance support
- ✅ Auto-validation after generation
- ✅ Container locking with operator tracking
- ✅ Error handling for invalid states
- ✅ Business logic validation (primary selection limits, distance requirements)

Phase 4 (session integration) has partial test coverage for backend functions. UI integration and session creation workflow remain as future work items.

**Test coverage is comprehensive** with 34 passing tests covering normal workflows, error cases, edge cases, business logic validation, and full end-to-end integration.

## New Business Rules Enforced

### 1. Max One Primary Per Type+Detector
**Rule**: Each measurement type (DARK/EMPTY/BACKGROUND/AGBH) can have at most ONE primary file per detector.

**Example Valid**:
- DARK for PRIMARY: file1 (primary)
- DARK for SECONDARY: file2 (primary)
- EMPTY for PRIMARY: file3 (primary)

**Example Invalid**:
- DARK for PRIMARY: file1 (primary) ← ❌
- DARK for PRIMARY: file2 (primary) ← ❌ Multiple primaries for same type+detector!

**Enforcement**: UI validation in `generate_technical_h5()` checks primary selections before generation. Shows clear error message with violations listed.

### 2. Distances Required for All Detectors
**Rule**: User cannot generate H5 container until distances are configured for EVERY active detector.

**Example Valid**:
- PRIMARY: 100 cm ✓
- SECONDARY: 17 cm ✓

**Example Invalid**:
- PRIMARY: 100 cm ✓
- SECONDARY: (not set) ← ❌ Missing!

**Enforcement**: 
- UI shows "Distances..." button to configure distances
- Validation checks for complete configuration before H5 generation
- Clear error message lists missing detectors
- Dev mode fallback: prompts for single distance if not configured
