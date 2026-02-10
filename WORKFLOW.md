# DIFRA HDF5 Container Workflow

## Overview

This document describes the complete workflow for creating, populating, and validating DIFRA HDF5 session containers using the centralized `hardware.container` module (v0.1 beta).

## Current Architecture

- **Module**: `hardware.container.v0_1`
- **Schema Version**: 1.0 (DIFRA HDF5 Data Model)
- **Container Types**: Technical containers, Session containers
- **Status**: ✅ All 36 integration tests passing

---

## Workflow Steps

### Step 1: Technical Container Creation

**Purpose**: Create calibration and reference measurements for the session.

#### Current Implementation ✓

```python
from hardware.container.v0_1 import technical_container, schema

# Prepare PONI calibration data
pony_data = {
    'PRIMARY': (poni_content, 'primary.poni'),
    'SECONDARY': (poni_content, 'secondary.poni'),
}

# Prepare detector configuration
detector_config = [
    {
        'id': 'PRIMARY',
        'alias': 'PRIMARY',
        'type': 'Pilatus',
        'size': [256, 256],
        'pixel_size_um': 172.0,
    },
    # ... more detectors
]

# Prepare technical measurements (DARK, EMPTY, BACKGROUND, AGBH)
aux_measurements = {
    'DARK': {'PRIMARY': 'path/to/dark_primary.npy', ...},
    'EMPTY': {'PRIMARY': 'path/to/empty_primary.npy', ...},
    'BACKGROUND': {'PRIMARY': 'path/to/background_primary.npy', ...},
    'AGBH': {'PRIMARY': 'path/to/agbh_primary.npy', ...},
}

# Generate technical container
tech_id, tech_file = technical_container.generate_from_aux_table(
    folder=output_dir / 'technical',
    aux_measurements=aux_measurements,
    pony_data=pony_data,
    detector_config=detector_config,
    active_detector_ids=['PRIMARY', 'SECONDARY'],
    distance_cm=17.0,
)
```

**Tested**:
- ✓ PONI calibration files (PRIMARY, SECONDARY)
- ✓ Technical measurements (DARK, EMPTY, BACKGROUND, AGBH)
- ✓ Detector configuration
- ✓ Distance metadata (17cm)

#### Proposed Extensions

**Priority 1: Essential**
1. **Add WATER technical type**
   - Extend schema to support WATER measurements
   - Add to `schema.ALL_TECHNICAL_TYPES`
   - Test WATER in technical container generation

2. **Add faulty pixel masks**
   - Store bad pixel maps per detector
   - Link to detector configuration
   - Format: boolean mask or coordinate list

3. **Add detector flatfield corrections**
   - Store gain/offset maps
   - Per-detector calibration curves
   - Timestamp and validity period

**Priority 2: Multi-Distance**
4. **Multi-distance calibration support**
   - Store multiple distance calibrations
   - Per-distance PONI files
   - Automatic distance selection based on session

**Priority 3: Advanced**
5. **Beam profile measurements**
   - Beam intensity profile data
   - Spatial uniformity corrections
   - Temporal stability tracking

---

### Step 2: Session Container Creation

**Purpose**: Initialize a new session container for a sample/measurement run.

#### Current Implementation ✓

```python
from hardware.container.v0_1 import writer

session_id, session_file = writer.create_session_container(
    folder=output_dir / 'sessions',
    sample_id='WORKFLOW_TEST_SAMPLE',
    operator_id='test_user',
    site_id='test_facility',
    machine_name='DIFRA_TEST',
    beam_energy_keV=12.5,
    acquisition_date='2026-02-10',
)

# Embed technical data
writer.copy_technical_to_session(tech_file, session_file)
```

**Tested**:
- ✓ Sample ID, operator, site, machine metadata
- ✓ Beam energy
- ✓ Acquisition date
- ✓ Technical data embedding

#### Proposed Extensions

**Priority 1: Essential Metadata**
1. **Add patient_id for medical samples**
   ```python
   session_id, session_file = writer.create_session_container(
       ...,
       patient_id='PATIENT_12345',  # Optional medical identifier
   )
   ```

2. **Add sample_description**
   - Material type (bone, tissue, polymer, metal, etc.)
   - Sample preparation method
   - Sample dimensions
   - Sample orientation

3. **Add environmental conditions**
   ```python
   writer.add_environmental_conditions(
       session_file,
       temperature_celsius=23.5,
       humidity_percent=45.0,
       pressure_hPa=1013.25,
       timestamp=...,
   )
   ```

**Priority 2: Provenance**
4. **Add beam parameters**
   - Beam flux (photons/s)
   - Beam size (μm × μm)
   - Beam stability metrics
   - Monochromaticity (ΔE/E)

5. **Add audit trail**
   - Track every modification with timestamp
   - User ID for each operation
   - Software version information
   - Automated vs manual operations

**Priority 3: Extended Metadata**
6. **Add project/study metadata**
   - Project ID
   - Study protocol reference
   - Institutional review board approval (medical)
   - Grant/funding information

---

### Step 3: Image & Zone Management

**Purpose**: Define sample geometry and regions of interest.

#### Current Implementation ✓

```python
# Add sample image
image_data = np.random.rand(512, 512).astype(np.float32)
writer.add_image(session_file, 1, image_data)

# Add sample holder zone
writer.add_zone(
    session_file, 1,
    zone_role=schema.ZONE_ROLE_SAMPLE_HOLDER,
    geometry_px=[[100, 100], [400, 100], [400, 400], [100, 400]],
    holder_diameter_mm=25.0,
)
```

**Tested**:
- ✓ Single sample image (512×512)
- ✓ Sample holder zone definition
- ✓ Polygon geometry

#### Proposed Extensions

**Priority 1: Multi-Zone Support**
1. **Include/exclude zones**
   ```python
   # Include zone (measure only inside)
   writer.add_zone(
       session_file, 2,
       zone_role=schema.ZONE_ROLE_INCLUDE,
       geometry_px=[[150, 150], [350, 350]],
   )
   
   # Exclude zone (skip this region - damaged/artifact)
   writer.add_zone(
       session_file, 3,
       zone_role=schema.ZONE_ROLE_EXCLUDE,
       geometry_px=[[250, 250], [260, 260]],
       exclude_reason="sample_damage",
   )
   ```

2. **Multiple image support**
   ```python
   # Different angles, lighting conditions
   writer.add_image(session_file, 1, image_front, image_type='visible_front')
   writer.add_image(session_file, 2, image_side, image_type='visible_side')
   writer.add_image(session_file, 3, image_uv, image_type='uv_fluorescence')
   ```

**Priority 2: Geometric Shapes**
3. **Circular/elliptical zones**
   ```python
   writer.add_zone(
       session_file, 4,
       zone_role=schema.ZONE_ROLE_INCLUDE,
       shape='circle',
       geometry_px={'center': [256, 256], 'radius': 100},
   )
   ```

4. **Zone hierarchy**
   - Parent-child relationships
   - Nested zones for complex geometries
   - Zone inheritance (properties propagate)

**Priority 3: Advanced Imaging**
5. **Time-series imaging**
   - Before/during/after measurement
   - Track sample changes
   - Detect motion/deformation

6. **Image registration metadata**
   - Alignment parameters
   - Reference coordinate system
   - Transformation matrices

---

### Step 4: Point Definition

**Purpose**: Specify measurement locations on the sample.

#### Current Implementation ✓

```python
for pt_idx in range(1, 4):
    writer.add_point(
        session_file, pt_idx,
        pixel_coordinates=[100.0 + pt_idx * 50, 100.0],
        physical_coordinates_mm=[10.0 + pt_idx * 5, 10.0],
    )
    
    # Update status after measurement
    writer.update_point_status(session_file, pt_idx, schema.POINT_STATUS_MEASURED)
```

**Tested**:
- ✓ 3 measurement points
- ✓ Pixel coordinates
- ✓ Physical coordinates (mm)
- ✓ Point status updates

#### Proposed Extensions

**Priority 1: Point Generation**
1. **Grid generation utilities**
   ```python
   from hardware.container.v0_1 import point_generator
   
   # Regular grid
   points = point_generator.create_grid(
       zone_id='zone_001',
       spacing_mm=1.0,
       grid_type='square',  # or 'hexagonal'
   )
   
   # Adaptive grid (denser in regions of interest)
   points = point_generator.create_adaptive_grid(
       zone_id='zone_001',
       base_spacing_mm=2.0,
       roi_spacing_mm=0.5,
       roi_zones=['zone_002'],
   )
   ```

2. **Point metadata**
   ```python
   writer.add_point(
       session_file, pt_idx,
       pixel_coordinates=[100.0, 100.0],
       physical_coordinates_mm=[10.0, 10.0],
       priority=1,  # Measurement order
       estimated_time_s=30.0,
       focus_quality=0.95,
   )
   ```

**Priority 2: Point Management**
3. **Point groups/batches**
   - Group points for batch processing
   - Pause/resume at batch boundaries
   - Batch-level statistics

4. **Skip reasons tracking**
   ```python
   writer.update_point_status(
       session_file, pt_idx,
       schema.POINT_STATUS_SKIPPED,
       skip_reason='out_of_zone',  # or 'low_quality', 'damaged_area'
   )
   ```

**Priority 3: Quality Control**
5. **Point quality metrics**
   - Position accuracy (stage repeatability)
   - Focus score
   - Expected vs actual coordinates
   - Measurement complexity estimate

---

### Step 5: Measurement Capture

**Purpose**: Record diffraction patterns from detectors.

#### Current Implementation ✓

```python
measurement_data = {
    'PRIMARY': np.random.rand(256, 256).astype(np.float32),
    'SECONDARY': np.random.rand(256, 256).astype(np.float32),
}

detector_metadata = {
    'PRIMARY': {'integration_time_ms': 100.0, 'beam_energy_keV': 12.5},
    'SECONDARY': {'integration_time_ms': 100.0, 'beam_energy_keV': 12.5},
}

pony_alias_map = {'PRIMARY': 'PRIMARY', 'SECONDARY': 'SECONDARY'}

writer.add_measurement(
    session_file, pt_idx,
    measurement_data, detector_metadata, pony_alias_map,
)
```

**Tested**:
- ✓ 2 detectors (PRIMARY, SECONDARY)
- ✓ Raw signal data (256×256)
- ✓ Integration time metadata
- ✓ Beam energy metadata
- ✓ PONY calibration references

#### Proposed Extensions

**Priority 1: Correction Tracking**
1. **Dark/Empty/Background references**
   ```python
   writer.add_measurement(
       session_file, pt_idx,
       measurement_data, detector_metadata, pony_alias_map,
       correction_applied={
           'dark_subtracted': True,
           'empty_normalized': False,
           'background_subtracted': False,
       },
       correction_tech_events={
           'dark': 'tech_evt_001',
           'empty': 'tech_evt_002',
       },
   )
   ```

2. **Saturation handling**
   - Flag saturated pixels
   - Store saturation mask
   - Track dynamic range usage

**Priority 2: Advanced Acquisition**
3. **Multi-exposure measurements (HDR)**
   ```python
   # Store multiple exposures for HDR reconstruction
   writer.add_measurement_series(
       session_file, pt_idx,
       exposure_times_ms=[10, 50, 100, 500],
       measurement_data_list=[...],
   )
   ```

4. **Time-resolved measurements**
   - Pump-probe experiments
   - Time stamps per frame
   - Trigger/sync information

**Priority 3: In-Situ Measurements**
5. **Temperature-dependent measurements**
   ```python
   detector_metadata['temperature_celsius'] = 150.0
   detector_metadata['temperature_stable'] = True
   ```

6. **Stress/strain measurements**
   - Applied load tracking
   - Strain gauge readings
   - Time synchronization

7. **Cosmic ray detection**
   - Flag cosmic ray hits
   - Store hit locations
   - Cleaning status

---

### Step 6: Data Verification

**Purpose**: Validate container structure and data quality.

#### Current Implementation ✓

```python
from hardware.container import open_container

# Open and verify structure
container = open_container(session_file, validate=False)
metadata = container.get_metadata()
points = container.get_points()
measurements = container.get_measurements()
images = container.get_images()

# Check data readability
detector_data = container.get_detector_data(1, 1, 'PRIMARY')
```

**Tested**:
- ✓ Container structure validation
- ✓ Metadata integrity
- ✓ Data readability

#### Proposed Extensions

**Priority 1: Quality Checks**
1. **Statistical validation**
   ```python
   from hardware.container.v0_1 import quality_checks
   
   qc_report = quality_checks.validate_measurement_quality(
       session_file,
       checks=['signal_noise_ratio', 'detector_stats', 'saturation'],
   )
   
   # Report:
   # - Mean signal level
   # - Noise characteristics
   # - Saturation percentage
   # - Dead/hot pixel detection
   ```

2. **Detector sanity checks**
   - Dead pixel count vs expected
   - Hot pixel locations
   - Uniformity metrics
   - Gain stability

**Priority 2: Calibration Monitoring**
3. **Calibration drift detection**
   ```python
   drift_report = quality_checks.check_calibration_drift(
       session_file,
       reference_technical_file=tech_file,
   )
   ```

4. **Measurement reproducibility**
   - Repeat measurements comparison
   - Statistical consistency
   - Outlier detection

**Priority 3: Data Integrity**
5. **Checksums and corruption detection**
   ```python
   writer.add_checksums(session_file)
   validator.verify_checksums(session_file)
   ```

6. **Schema version compatibility**
   - Forward/backward compatibility checks
   - Migration recommendations
   - Deprecated feature warnings

---

## Implementation Roadmap

### Phase 1: Essential Extensions (Immediate)
**Goal**: Add most-requested features for production use

- [ ] Add WATER technical type
- [ ] Add include/exclude zones
- [ ] Add patient_id support
- [ ] Add environmental conditions
- [ ] Add correction tracking

**Estimated effort**: 2-3 weeks
**Tests to add**: ~10 new tests
**Documentation**: Update SESSION_CONTAINER_IMPLEMENTATION.md

### Phase 2: Analysis Pipeline (Next)
**Goal**: Enable automated analysis workflows

- [ ] Add point grid generation utilities
- [ ] Add multi-exposure support
- [ ] Add statistical quality checks
- [ ] Add skip reason tracking
- [ ] Add correction reference tracking

**Estimated effort**: 3-4 weeks
**Tests to add**: ~15 new tests
**Documentation**: Create ANALYSIS_PIPELINE.md

### Phase 3: Advanced Features (Future)
**Goal**: Support complex experimental scenarios

- [ ] Add multi-distance calibration
- [ ] Add time-series imaging
- [ ] Add HDR measurement support
- [ ] Add in-situ measurement metadata
- [ ] Add real-time validation
- [ ] Add cloud upload preparation

**Estimated effort**: 6-8 weeks
**Tests to add**: ~20 new tests
**Documentation**: Create ADVANCED_FEATURES.md

---

## Testing Strategy

### Current Test Coverage
- **36 tests** across session and technical integration
- **Coverage**: Core workflow (create → populate → validate)
- **Performance**: ~35s for full test suite

### Extended Test Plan

**Per Extension Category**:
1. Unit tests for new functions
2. Integration tests for workflows
3. Performance benchmarks
4. Regression tests for backward compatibility

**Test Data**:
- Synthetic data (current approach)
- Real detector data samples
- Edge cases (empty containers, corrupted data)
- Large-scale containers (100+ points, 1000+ measurements)

---

## Usage Examples

### Basic Workflow (Current)
```python
from hardware.container.v0_1 import writer, technical_container, schema

# 1. Create technical container
tech_id, tech_file = technical_container.generate_from_aux_table(...)

# 2. Create session container
session_id, session_file = writer.create_session_container(...)
writer.copy_technical_to_session(tech_file, session_file)

# 3. Add image and zones
writer.add_image(session_file, 1, image_data)
writer.add_zone(session_file, 1, zone_role=schema.ZONE_ROLE_SAMPLE_HOLDER, ...)

# 4. Add points
writer.add_point(session_file, 1, pixel_coordinates=[100, 100], ...)

# 5. Add measurements
writer.add_measurement(session_file, 1, measurement_data, detector_metadata, pony_alias_map)
writer.update_point_status(session_file, 1, schema.POINT_STATUS_MEASURED)

# 6. Validate
from hardware.container import open_container
container = open_container(session_file)
metadata = container.get_metadata()
```

### Extended Workflow (Proposed - Phase 1)
```python
# Create session with extended metadata
session_id, session_file = writer.create_session_container(
    folder=output_dir,
    sample_id='BONE_SAMPLE_001',
    operator_id='researcher_1',
    site_id='hospital_lab',
    machine_name='DIFRA_MEDICAL_01',
    beam_energy_keV=12.5,
    patient_id='PATIENT_12345',  # NEW
)

# Add environmental conditions (NEW)
writer.add_environmental_conditions(
    session_file,
    temperature_celsius=23.5,
    humidity_percent=45.0,
)

# Add multiple images (NEW)
writer.add_image(session_file, 1, image_visible, image_type='visible')
writer.add_image(session_file, 2, image_xray, image_type='xray_radiograph')

# Add include/exclude zones (NEW)
writer.add_zone(session_file, 1, zone_role=schema.ZONE_ROLE_SAMPLE_HOLDER, ...)
writer.add_zone(session_file, 2, zone_role=schema.ZONE_ROLE_INCLUDE, ...)
writer.add_zone(session_file, 3, zone_role=schema.ZONE_ROLE_EXCLUDE, 
                exclude_reason='sample_damage', ...)

# Add measurement with correction tracking (NEW)
writer.add_measurement(
    session_file, 1,
    measurement_data, detector_metadata, pony_alias_map,
    correction_applied={'dark_subtracted': True},
    correction_tech_events={'dark': 'tech_evt_001'},
)

# Quality checks (NEW)
from hardware.container.v0_1 import quality_checks
qc_report = quality_checks.validate_measurement_quality(session_file)
```

---

## Next Steps

**Immediate Action Items**:
1. Review this workflow document with team
2. Prioritize Phase 1 extensions based on user needs
3. Create GitHub issues for each extension
4. Set up development branch for Phase 1
5. Write detailed technical specs for top 3 priorities

**Questions to Resolve**:
1. Which extensions are most critical for current users?
2. Should we maintain backward compatibility with v0.1?
3. When to bump to v0.2 vs staying in v0.1?
4. What's the migration path for existing containers?

---

## References

- `SESSION_CONTAINER_IMPLEMENTATION.md` - Original implementation doc
- `DIFRA_HDF5_Data_Model_FINAL.md` - Schema specification
- `src/hardware/container/v0_1/` - Current implementation
- Test suite: `src/hardware/difra/tests/test_*integration.py`
