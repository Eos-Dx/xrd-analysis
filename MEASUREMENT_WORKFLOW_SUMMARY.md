# DIFRA Measurement Workflow & H5 Container Relationship

## Overview
This document summarizes the complete measurement workflow in DIFRA, from technical calibration through session measurements, and how data flows into HDF5 containers.

---

## Container Architecture

### Two Container Types

#### 1. **Technical Container** (`technical_<id>_<distance>cm.h5`)
- **Purpose**: Stores calibration and geometry data
- **Lifecycle**: Created during technical measurements, recreated when geometry changes
- **Location**: `difra/technical/` folder
- **Contents**:
  - Technical measurements (DARK, EMPTY, BACKGROUND, AGBH, WATER)
  - PONI calibration files per detector
  - Detector configuration
  - Raw detector files (.txt, .dsc) stored as compressed blobs
- **Status**: Must be **locked** before use in sessions
- **Usage**: Source data copied into session containers

#### 2. **Session Container** (`session_<id>.h5`)
- **Purpose**: Final data product for one sample/acquisition session
- **Lifecycle**: Created per sample, uploaded to cloud, archived locally
- **Location**: User-specified session folder
- **Contents**:
  - **Copy of `/technical`** from technical container
  - Sample images
  - Zones (regions of interest)
  - Measurement points
  - Point measurements (detector data at each point)
  - Analytical measurements (attenuation, etc.)
- **Self-contained**: No cross-file references, fully portable

---

## Complete Workflow

### Phase 1: Technical Measurements (Calibration)

**Goal**: Create and validate technical container with calibration data

#### Steps:
1. **Hardware Initialization**
   - Initialize detectors (SAXS, WAXS, etc.)
   - Set detector distances (e.g., SAXS: 100cm, WAXS: 17cm)
   - Verify hardware connectivity

2. **Set Operator**
   - Select or define operator from `operators.json`
   - Operator tracked for all measurements

3. **Capture Technical Measurements**
   - For each detector and type:
     - **DARK**: Beam off, shutter closed
     - **EMPTY**: Beam on, no sample
     - **BACKGROUND**: Beam on, background only
     - **AGBH**: Silver behenate calibration standard
     - **WATER** (optional): Special measurements
   - Each measurement generates:
     - `.txt` file (raw detector data)
     - `.dsc` file (descriptor with metadata)
     - `.npy` file (processed numpy array)
   - Mark primary measurements (one per type per detector)

4. **Configure Distances**
   - Set per-detector distances via "Distances..." button
   - Example: SAXS=100cm, WAXS=17cm
   - Required before H5 generation

5. **Select PONI Files**
   - Provide PONI calibration file per detector
   - PONI contains geometry parameters:
     - Distance
     - Beam center (poni1, poni2)
     - Rotation angles
     - Pixel size
     - Wavelength
   - In DEV mode: fake PONIs auto-generated within ±3% of user distance

6. **Generate Technical H5**
   ```
   Click "Generate H5" →
   - Archives old containers to difra/archive/technical/<id>_<timestamp>/
   - Generates new container in difra/technical/
   - Stores .txt and .dsc as compressed blobs in H5
   - Validates schema version
   ```

7. **Validation & Locking**
   - Automatic validation runs:
     - Checks all required types present
     - Verifies schema version
     - Validates PONI distances (±5% tolerance)
   - User prompted to lock container
   - **Locking**:
     - Marks container immutable
     - Records operator and timestamp
     - Archives raw files (.txt, .dsc) to archive folder
     - Container now ready for session use

**Output**: `technical_<id>_<distance>cm.h5` (locked, validated)

---

### Phase 2: Session Measurements (Sample Analysis)

**Goal**: Measure sample at multiple points, store in self-contained session container

#### Prerequisites:
- ✅ Valid, locked technical container exists
- ✅ Technical container distance matches measurement setup
- ✅ Operator selected

#### Steps:

##### 2.1 Create Session & Load Image

1. **Start New Session**
   ```python
   session_manager.create_session(
       folder=session_folder,
       sample_id="SAMPLE_001",
       distance_cm=100.0,  # Must match technical container
       operator_id="operator_name"
   )
   ```
   - Creates `session_<session_id>.h5`
   - **Copies entire `/technical` group** from technical container
   - Initializes global measurement counter

2. **Load Sample Image**
   ```python
   session_manager.add_sample_image(
       image_data=sample_image,  # 2D numpy array or file path
       image_index=1,
       image_type="sample"
   )
   ```
   - Stores image in `/images/img_001`
   - Supports multiple images per session

##### 2.2 Define Zones (Regions of Interest)

3. **Define Sample Holder Zone** (required, exactly one)
   ```python
   session_manager.add_zone(
       zone_index=1,
       geometry_px={"center": [x, y], "radius": r},  # Pixels
       shape="circle",
       zone_role="sample_holder",
       holder_diameter_mm=50.0,  # Physical size
   )
   ```
   - Stores in `/zones/zone_001`
   - Defines valid measurement area

4. **Define Include/Exclude Zones** (optional)
   ```python
   session_manager.add_zone(
       zone_index=2,
       geometry_px=polygon_points,
       shape="polygon",
       zone_role="include"  # or "exclude"
   )
   ```
   - Include zones: only measure here
   - Exclude zones: skip these areas

##### 2.3 Add Image Mapping (Pixel ↔ Physical Coordinates)

5. **Configure Pixel-to-MM Conversion**
   ```python
   session_manager.add_image_mapping(
       sample_holder_zone_id="/zones/zone_001",
       pixel_to_mm_conversion={
           "scale_x": 0.05,  # mm/pixel
           "scale_y": 0.05,
           "offset_x": 0.0,
           "offset_y": 0.0
       },
       orientation="standard"
   )
   ```
   - Enables translation stage control in mm
   - Derives from holder diameter and pixel geometry

##### 2.4 Define Measurement Points

6. **Add Points**
   ```python
   points = [
       {
           "pixel_coordinates": [100, 150],     # [x_px, y_px]
           "physical_coordinates_mm": [5.0, 7.5],  # [x_mm, y_mm]
           "point_status": "pending"
       },
       # ... more points
   ]
   session_manager.add_points(points)
   ```
   - Stores in `/points/pt_001`, `/points/pt_002`, etc.
   - Status: `pending` → `measured` → `completed`/`failed`

##### 2.5 Perform Attenuation Measurements (Optional)

7. **Measure I₀ (without sample)**
   ```python
   i0_counter = session_manager.add_attenuation_measurement(
       measurement_data={
           "det_001": saxs_data_2d,
           "det_002": waxs_data_2d
       },
       detector_metadata={
           "det_001": {"integration_time_ms": 1000},
           "det_002": {"integration_time_ms": 500}
       },
       pony_alias_map={"SAXS": "det_001", "WAXS": "det_002"},
       mode="without"
   )
   ```

8. **Measure I (with sample)**
   ```python
   i_counter = session_manager.add_attenuation_measurement(
       # same parameters
       mode="with"
   )
   ```

9. **Link Attenuation to All Points**
   ```python
   session_manager.link_attenuation_to_points(num_points=len(points))
   ```
   - Stores in `/analytical_measurements/ana_<counter>`
   - Links both I₀ and I to every point via HDF5 references

##### 2.6 Perform Point Measurements

10. **Measure Each Point**
    ```python
    for point_idx in range(1, num_points + 1):
        # Move translation stage to point coordinates
        stage.move_to_mm(
            x=points[point_idx-1]["physical_coordinates_mm"][0],
            y=points[point_idx-1]["physical_coordinates_mm"][1]
        )
        
        # Trigger detector acquisition
        detector_data = capture_measurement(
            integration_time=exposure_time,
            num_frames=num_frames
        )
        
        # Write to container
        session_manager.add_measurement(
            point_index=point_idx,
            measurement_data={
                "det_001": detector_data["SAXS"],
                "det_002": detector_data["WAXS"]
            },
            detector_metadata={
                "det_001": {"integration_time_ms": 1000},
                "det_002": {"integration_time_ms": 500}
            },
            pony_alias_map={"SAXS": "det_001", "WAXS": "det_002"}
        )
        
        # Status automatically updated to "measured"
    ```
    - Stores in `/measurements/pt_<idx>/meas_<counter>/det_<id>`
    - Global measurement counter increments atomically
    - Raw detector signal stored as compressed datasets

11. **Complete Session**
    ```python
    session_manager.close_session()
    # Upload to cloud, archive locally
    ```

**Output**: `session_<session_id>.h5` (self-contained, ready for upload)

---

## HDF5 Structure Comparison

### Technical Container Structure
```
technical_<id>_17cm.h5
├── [attributes]
│   ├── schema_version: "1.0"
│   ├── container_type: "technical"
│   └── creation_timestamp
│
└── /technical
    ├── /tech_evt_001  [AGBH, distance_cm=17.0]
    │   ├── /det_001  [detector_id="det_001", alias="SAXS"]
    │   │   ├── raw_signal  (2D array)
    │   │   ├── raw_blob_txt  (compressed .txt)
    │   │   └── raw_blob_dsc  (compressed .dsc)
    │   └── /det_002  [detector_id="det_002", alias="WAXS"]
    │       ├── raw_signal
    │       ├── raw_blob_txt
    │       └── raw_blob_dsc
    ├── /tech_evt_002  [DARK]
    ├── /tech_evt_003  [EMPTY]
    ├── /tech_evt_004  [BACKGROUND]
    │
    ├── /pony
    │   ├── /pony_001  [detector_id="det_001"]
    │   │   └── poni_content  (string dataset)
    │   └── /pony_002  [detector_id="det_002"]
    │       └── poni_content
    │
    └── /config
        └── detector_config  (JSON)
```

### Session Container Structure
```
session_<session_id>.h5
├── [attributes]
│   ├── sample_id: "SAMPLE_001"
│   ├── session_id: "<uuid>"
│   ├── operator_id: "operator_name"
│   ├── beam_energy_keV: 17.5
│   └── acquisition_date
│
├── /technical  ← COPIED FROM TECHNICAL CONTAINER
│   └── [same structure as above]
│
├── /images
│   ├── /img_001  [image_type="sample"]
│   │   └── image_data  (2D array)
│   └── /zones
│       ├── /zone_001  [zone_role="sample_holder", shape="circle"]
│       │   └── geometry_px  (JSON)
│       └── /mapping
│           └── pixel_to_mm  (JSON)
│
├── /points
│   ├── /pt_001  [pixel_coordinates, physical_coordinates_mm, point_status]
│   │   └── analytical_measurement_refs → [/analytical_measurements/ana_001, ...]
│   ├── /pt_002
│   └── ...
│
├── /measurements
│   ├── /pt_001
│   │   ├── /meas_000000003  [counter=3, timestamp, status]
│   │   │   ├── /det_001  [integration_time_ms, beam_energy_keV]
│   │   │   │   ├── raw_signal  (2D array)
│   │   │   │   └── raw_files/  (optional)
│   │   │   └── /det_002
│   │   │       └── raw_signal
│   │   └── /meas_000000004
│   ├── /pt_002
│   └── ...
│
└── /analytical_measurements
    ├── /ana_000000001  [counter=1, analysis_type="attenuation"]
    │   ├── /det_001
    │   │   └── raw_signal  (I₀)
    │   └── /det_002
    └── /ana_000000002  [counter=2]
        ├── /det_001
        │   └── raw_signal  (I)
        └── /det_002
```

---

## Key Relationships

### 1. Technical → Session (Copy on Create)
```
When session created:
- Entire /technical group copied from technical container
- Session becomes self-contained
- No runtime dependency on technical container
```

### 2. Points ← Analytical Measurements (HDF5 References)
```
/points/pt_001/analytical_measurement_refs = [
    HDF5_REFERENCE(/analytical_measurements/ana_000000001),  # I₀
    HDF5_REFERENCE(/analytical_measurements/ana_000000002)   # I
]
```

### 3. Measurements → Points (Path-based)
```
/measurements/pt_001/meas_000000003/
    ↓ belongs to
/points/pt_001/
```

### 4. PONI → Detectors (ID-based)
```
/technical/pony/pony_001
    [detector_id="det_001"]
        ↓ applies to
/measurements/pt_001/meas_000000003/det_001/
```

---

## Measurement Counter Logic

**Global, monotonic counter shared across:**
- Regular measurements (`/measurements/pt_*/meas_*`)
- Analytical measurements (`/analytical_measurements/ana_*`)

**Rules:**
- Never resets during session
- Increments atomically for each acquisition
- Preserved even for failed/aborted measurements
- Format: Zero-padded 9-digit (`meas_000000001`)

**Example sequence:**
```
ana_000000001  ← I₀ attenuation (without sample)
ana_000000002  ← I attenuation (with sample)
meas_000000003 ← Point 1 measurement
meas_000000004 ← Point 2 measurement
meas_000000005 ← Point 3 measurement
...
```

---

## Workflow State Transitions

### Technical Workflow
```
START
  ↓
HARDWARE_INIT
  ↓
SET_OPERATOR
  ↓
CAPTURE_MEASUREMENTS → [DARK, EMPTY, BACKGROUND, AGBH captured]
  ↓
CONFIGURE_DISTANCES → [Distances set per detector]
  ↓
SELECT_PONI_FILES
  ↓
GENERATE_H5 → [Container created in temp, moved to storage]
  ↓
VALIDATE → [Schema + PONI distance validation]
  ↓
LOCK → [Container locked, raw files archived]
  ↓
READY_FOR_SESSIONS
```

### Session Workflow
```
START
  ↓
CREATE_SESSION → [Copy /technical, init counter]
  ↓
LOAD_IMAGE → [Sample image stored]
  ↓
DEFINE_ZONES → [Holder + include/exclude zones]
  ↓
ADD_MAPPING → [Pixel↔mm conversion]
  ↓
ADD_POINTS → [Measurement locations defined]
  ↓
[Optional] ATTENUATION → [I₀, I measured and linked]
  ↓
MEASURE_POINTS → [Loop: move stage, capture, write]
  ↓
CLOSE_SESSION → [Upload, archive]
  ↓
END
```

---

## Critical Validation Rules

### Technical Container
1. ✅ All required types present per detector
2. ✅ Exactly one primary measurement per (type, detector)
3. ✅ PONI distances within ±5% of user-specified distances
4. ✅ Schema version matches expected version
5. ✅ Container must be locked before session use

### Session Container
1. ✅ Valid locked technical container exists at specified distance
2. ✅ Exactly one sample_holder zone
3. ✅ All points within valid zones
4. ✅ Measurement counter never decreases
5. ✅ Point status transitions: pending → measured → completed/failed

---

## File Management

### Raw Data Archiving
```
BEFORE H5 GENERATION:
difra/technical/
├── agbh_measurement_SAXS.txt
├── agbh_measurement_SAXS.dsc
├── agbh_measurement_WAXS.txt
└── agbh_measurement_WAXS.dsc

AFTER H5 GENERATION & LOCK:
difra/technical/
└── technical_<id>_100cm.h5  (contains .txt/.dsc as blobs)

difra/archive/technical/<id>_<timestamp>/
├── technical_<id>_100cm.h5  (old container if existed)
├── agbh_measurement_SAXS.txt
├── agbh_measurement_SAXS.dsc
├── agbh_measurement_WAXS.txt
└── agbh_measurement_WAXS.dsc
```

### Session Management
```
session_folder/
├── session_<id>.h5  (active session)
└── [uploaded to cloud after completion]

After upload:
- Local copy retained or archived
- New sample → new session container
```

---

## Summary

1. **Technical Container** = Calibration data source
   - Created during technical measurements
   - Locked before use
   - Copied into each session

2. **Session Container** = Complete measurement dataset
   - Self-contained (includes technical data)
   - One per sample
   - Ready for cloud upload and analysis

3. **Workflow** = Two-phase process
   - Phase 1: Calibration (technical container)
   - Phase 2: Measurements (session container)

4. **Data Flow** = Technical → Session (copy) + Images + Zones + Points + Measurements

5. **Key Innovation** = Self-contained session containers eliminate runtime dependencies
