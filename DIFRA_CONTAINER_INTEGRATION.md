# DIFRA Container System Integration Guide

This guide explains how to integrate the HDF5 container system (`hardware/container/v0_1`) into the DIFRA GUI.

## Overview

The container system is **fully implemented and tested** (37/37 tests passing):
- ✅ Technical container management (24 tests)
- ✅ Analytical measurements (attenuation) (13 tests)
- ✅ PONI validation, locking, archiving
- ✅ Primary/supplementary marking
- ✅ Session containers with point linking

## Quick Start

### 1. Import the Container API

```python
from hardware.container.v0_1 import writer, schema
from hardware.container.v0_1.technical_container import generate_from_aux_table
from hardware.container.v0_1.container_manager import lock_container
```

### 2. Create Technical Container (from Aux Table UI)

```python
# When user finishes technical measurements selection in Aux Table
tech_id, tech_path = generate_from_aux_table(
    folder=session_folder,
    aux_measurements={
        "DARK": {"DET1": "/path/to/dark_det1.npy", "DET2": "/path/to/dark_det2.npy"},
        "EMPTY": {...},
        "BACKGROUND": {...},
        "AGBH": {...},
    },
    pony_data={
        "DET1": (poni_content_str, "DET1_17cm.poni"),
        "DET2": (poni_content_str, "DET2_17cm.poni"),
    },
    detector_config=detector_config_list,  # From global config
    active_detector_ids=["DET1", "DET2"],
    distance_cm=17.0,
    validate_poni=True,  # Validates distance within 5%
)

# Lock technical container when confirmed
lock_container(tech_path)
```

### 3. Create Session Container (Start of Sample Measurement)

```python
# When user starts a new sample measurement session
session_id, session_path = writer.create_session_container(
    folder=session_folder,
    sample_id="SAMPLE_001",
    operator_id="operator_name",
    site_id="DIFRA_LAB",
    machine_name="DIFRA-01",
    beam_energy_keV=17.5,
    acquisition_date="2024-01-15",
)

# Copy technical data (PONI, config, measurements) to session
writer.copy_technical_to_session(
    technical_file=tech_path,
    session_file=session_path,
    auto_lock=False,  # Already locked
)
```

### 4. Add Sample Images and Zones

```python
# Add sample image
writer.add_image(
    file_path=session_path,
    image_index=1,
    image_data=image_array,  # or path to .npy
    image_type="sample",
)

# Define sample holder zone (for point placement)
writer.add_zone(
    file_path=session_path,
    zone_index=1,
    zone_role="sample_holder",
    geometry_px={"center": [512, 512], "radius": 450},  # circle
    shape="circle",
    image_index=1,
    holder_diameter_mm=25.0,
)
```

### 5. Add Measurement Points

```python
# Add points where measurements will be taken
for idx, (px_x, px_y, mm_x, mm_y) in enumerate(point_list, start=1):
    writer.add_point(
        file_path=session_path,
        point_index=idx,
        pixel_coordinates=[px_x, px_y],
        physical_coordinates_mm=[mm_x, mm_y],
    )
```

### 6. Record Attenuation Measurements

```python
# Before sample measurements, record attenuation
# Step 1: Measure WITHOUT sample (I₀)
without_sample_data = {
    "DET1": detector1_frame_without,  # 2D numpy array
    "DET2": detector2_frame_without,
}

detector_metadata = {
    "DET1": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
    "DET2": {"integration_time_ms": 50.0, "beam_energy_keV": 17.5},
}

pony_alias_map = {"DET1": "DET1", "DET2": "DET2"}

i0_path = writer.add_analytical_measurement(
    file_path=session_path,
    measurement_data=without_sample_data,
    detector_metadata=detector_metadata,
    pony_alias_map=pony_alias_map,
    analysis_type="attenuation",
    timestamp_start="2024-01-15 10:00:00",
)

# Step 2: Measure WITH sample (I)
with_sample_data = {
    "DET1": detector1_frame_with,
    "DET2": detector2_frame_with,
}

i_path = writer.add_analytical_measurement(
    file_path=session_path,
    measurement_data=with_sample_data,
    detector_metadata=detector_metadata,
    pony_alias_map=pony_alias_map,
    analysis_type="attenuation",
    timestamp_start="2024-01-15 10:01:00",
)

# Step 3: Link both measurements to all points
for point_idx in range(1, num_points + 1):
    # Get measurement counters from return paths or HDF5 attributes
    i0_counter = 1  # First analytical measurement
    i_counter = 2   # Second analytical measurement
    
    writer.link_analytical_measurement_to_point(
        file_path=session_path,
        point_index=point_idx,
        analytical_measurement_index=i0_counter,
    )
    writer.link_analytical_measurement_to_point(
        file_path=session_path,
        point_index=point_idx,
        analytical_measurement_index=i_counter,
    )
```

### 7. Record Sample Measurements

```python
# For each point, record measurement
for point_idx in range(1, num_points + 1):
    measurement_data = {
        "DET1": detector1_frame,
        "DET2": detector2_frame,
    }
    
    detector_metadata = {
        "DET1": {
            "integration_time_ms": 1000.0,
            "beam_energy_keV": 17.5,  # Can vary per detector
        },
        "DET2": {
            "integration_time_ms": 1000.0,
            "beam_energy_keV": 17.5,
        },
    }
    
    meas_path = writer.add_measurement(
        file_path=session_path,
        point_index=point_idx,
        measurement_data=measurement_data,
        detector_metadata=detector_metadata,
        pony_alias_map=pony_alias_map,
    )
    
    # Update point status
    writer.update_point_status(
        file_path=session_path,
        point_index=point_idx,
        point_status="measured",
    )
```

## Integration Points in DIFRA GUI

### A. Technical Measurements Tab (`zone_technical_measurements/`)

**Current**: Saves measurements as separate `.npy` files
**New**: Use `generate_from_aux_table()` to create container

**Changes needed**:
1. Replace file-based saving with container creation
2. Add PONI validation before creating container
3. Show validation errors to user if distance mismatch
4. Lock container after user confirmation

### B. Attenuation Tab (`zone_measurements/attenuation_mixin.py`)

**Current**: Saves attenuation frames as `.npy` files, calculates α in GUI
**New**: Store in session container as analytical measurements

**Changes needed**:
1. Call `writer.add_analytical_measurement()` instead of saving `.npy`
2. Store both I₀ (without) and I (with) measurements
3. Link to all relevant points after measurements complete
4. Keep calculation UI for display, but data comes from container

### C. Session Management (`main_window.py` or session controller)

**Current**: No session container concept
**New**: Create session container at start of each sample measurement

**Changes needed**:
1. Add "New Session" workflow:
   - Prompt for sample_id, operator_id
   - Create session container
   - Copy technical data from active technical container
2. Store current `session_path` in application state
3. All measurement operations write to this session

### D. Point Definition and Measurement

**Current**: Points defined in UI, measurements stored separately
**New**: Points and measurements stored in session container

**Changes needed**:
1. After user defines points on image, call `writer.add_point()` for each
2. During point measurement loop, call `writer.add_measurement()`
3. Link analytical measurements (attenuation) to points

## Container File Management

### File Naming Convention
- Technical: `technical_<container_id>_<distance>cm.h5`
- Session: `session_<container_id>_<sample_id>.h5`
- Archived: `technical_<id>_archived_<timestamp>.h5`

### Directory Structure
```
/data/difra/sessions/2024-01-15/
├── technical_a1b2c3d4_17cm.h5         # Locked technical container
├── session_e5f6g7h8_SAMPLE_001.h5    # Session container
├── session_i9j0k1l2_SAMPLE_002.h5
└── archive/
    └── technical_m3n4o5p6_archived_20240115_120000.h5
```

### Workflow States

1. **Technical Container Creation**
   - Status: Unlocked
   - Action: User fills Aux Table, creates container
   - Result: `technical_*.h5` (unlocked)

2. **Technical Container Locking**
   - Status: Unlocked → Locked
   - Action: User confirms measurements are correct
   - Result: `technical_*.h5` (locked, read-only)

3. **Session Creation**
   - Status: New session
   - Action: User starts new sample measurement
   - Result: `session_*.h5` with copied technical data

4. **Measurement Acquisition**
   - Status: Active session
   - Action: Record attenuation, then sample measurements
   - Result: Session container filled with data

5. **Archive**
   - Status: New distance needed
   - Action: Create new technical container, archive old
   - Result: Old moved to `archive/`

## Helper Functions for GUI

### Get Current Active Technical Container

```python
from hardware.container.v0_1.container_manager import find_active_technical_container

tech_path = find_active_technical_container(
    folder=session_folder,
    distance_cm=17.0,
    exclude_archived=True,
)

if tech_path is None:
    # No technical container for this distance - need to create one
    show_aux_table_dialog()
```

### Check if Technical Container is Locked

```python
from hardware.container.v0_1.container_manager import is_container_locked

if not is_container_locked(tech_path):
    # Warn user: Container not locked, may be modified
    confirm = show_lock_confirmation_dialog()
    if confirm:
        lock_container(tech_path)
```

### Find Active Session for Sample

```python
from hardware.container.v0_1.writer import find_active_session_container

session_path = find_active_session_container(
    folder=session_folder,
    sample_id="SAMPLE_001",
)
```

## Data Reading (For Analysis)

### Read Container Contents

```python
import h5py

with h5py.File(session_path, 'r') as f:
    # Read session metadata
    sample_id = f.attrs['sample_id']
    beam_energy = f.attrs['beam_energy_keV']
    
    # Read PONI for detector
    poni_content = f['/technical/pony/pony_det1'][()].decode()
    
    # Read point locations
    point_group = f['/points/pt_001']
    coords_mm = point_group.attrs['physical_coordinates_mm']
    
    # Read measurement data
    meas_data = f['/measurements/pt_001/meas_000000001/det_det1/raw_signal'][:]
    
    # Read attenuation data
    i0_data = f['/analytical_measurements/ana_000000001/det_det1/raw_signal'][:]
    i_data = f['/analytical_measurements/ana_000000002/det_det1/raw_signal'][:]
    
    # Calculate attenuation
    alpha = -np.log10(i_data / i0_data)
```

## Migration Strategy

### Phase 1: Technical Container Only
1. Replace Aux Table file saving with `generate_from_aux_table()`
2. Keep existing session/measurement workflow unchanged
3. Test: Create technical containers, verify PONI validation

### Phase 2: Session Container + Measurements
1. Add session creation on sample start
2. Store measurements in session container
3. Keep attenuation as separate files for now
4. Test: Full measurement workflow with container storage

### Phase 3: Analytical Measurements
1. Integrate attenuation into session as analytical measurements
2. Link attenuation to points
3. Update analysis code to read from containers
4. Test: Complete workflow including attenuation

### Phase 4: Cleanup
1. Remove old file-based storage code
2. Add container validation before analysis
3. Implement archive management UI
4. Document user workflows

## Testing

Run all container tests:
```bash
pytest src/hardware/difra/tests/test_container_management.py -v    # 24 tests
pytest src/hardware/difra/tests/test_analytical_measurements.py -v # 13 tests
```

## Support

- Container API documentation: `src/hardware/container/v0_1/README.md`
- Workflow documentation: `WORKFLOW.md`
- Schema constants: `src/hardware/container/v0_1/schema.py`
- Writer functions: `src/hardware/container/v0_1/writer.py`
- Technical container: `src/hardware/container/v0_1/technical_container.py`

## Example: Complete Workflow

See `src/hardware/difra/tests/test_analytical_measurements.py::test_attenuation_workflow` for a complete example of:
1. Creating session with technical data
2. Adding multiple points
3. Recording attenuation (without/with sample)
4. Linking attenuation to all points
5. Recording sample measurements

This test demonstrates the full integration pattern.
