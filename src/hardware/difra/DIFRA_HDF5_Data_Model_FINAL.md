# DIFRA HDF5 Data Model Specification

Version: 1.0  
Status: Final / Approved for implementation  

---

## 1. Design Principles

- HDF5 is the primary structured data container.
- Final session containers are fully self-contained.
- No cross-file HDF5 object references are used.
- Technical calibration data and acquisition data are logically separated but physically aggregated in final containers.
- Raw detector data are never deleted automatically.
- The model supports crash recovery and partial session continuation.

---

## 2. Container Types

### 2.1 Technical Container (`technical.h5`)

Persistent local container holding calibration and geometry data.

**Lifecycle**
- Created when technical measurements start.
- Recreated when detector geometry or distance changes.
- Used as a source for session containers.
- Never uploaded as a final product.

Contains:
```
/technical
```

---

### 2.2 Session / Sample Container (`session_<id>.h5`)

Final data product for a single sample and acquisition session.

**Lifecycle**
1. Created when a new sample/image is loaded.
2. `/technical` is copied from the active technical container.
3. Images, points, measurements, and analytical measurements are appended.
4. Container is finalized (locked/read-only).
5. Container is sent to cloud (currently fake send in development mode).
6. Container is moved from active measurements folder to session archive.

---

## 3. Root-Level Attributes (Session Container)

Required:
```
sample_id
study_name
session_id
creation_timestamp
acquisition_date
operator_id
site_id
machine_name
beam_energy_keV
```

Optional:
```
patient_id
```

Attributes are mutable until finalization. After finalization/send, the container is immutable at the OS level.

---

## 4. Technical Data (`/technical`)

### 4.1 Guaranteed Technical Measurement Types

- DARK (beam off)
- EMPTY
- BACKGROUND
- AGBH
- WATER

The set may be extended in the future.

---

### 4.2 Technical Measurements Structure

```
/technical
├── tech_evt_001
│   ├── det_primary
│   └── det_secondary
├── tech_evt_002
└── ...
```

Attributes:
```
technical_type
distance_cm
timestamp
detector_id
```

---

### 4.3 Detector Configuration

Stored in `/technical/config`:
- detector IDs
- spatial arrangement matrix
- detector roles
- optional bad-pixel metadata

---

### 4.4 PONI Geometry (`/technical/poni`)

One PONI file per detector.

```
/technical/poni
├── pony_primary
├── pony_secondary
```

Attributes:
```
detector_id
distance_cm
derived_from
operator_confirmed
```

Measurements cannot start without valid PONI geometry.

---

## 5. Images and Geometry

### 5.1 Images

```
/images
└── img_001
```

Attributes:
```
timestamp
image_type
```

Multiple images are allowed.

---

### 5.2 Zones

```
/images/zones
```

Zone roles:
- sample_holder (exactly one)
- include
- exclude

Attributes:
```
zone_role
shape
geometry_px
holder_diameter_mm (sample_holder only)
```

---

### 5.3 Mapping

```
/images/mapping
```

Contains a JSON dataset defining:
- sample_holder zone ID
- pixel-to-mm conversion
- orientation
- mapping version

---

## 6. Points

```
/points
├── pt_001
├── pt_002
└── ...
```

Attributes:
```
pixel_coordinates
physical_coordinates_mm
point_status
analytical_measurement_refs (list of HDF5 references)
```

---

## 7. Measurements (Point-Centric)

Physical detector acquisitions performed at specific spatial points.

```
/measurements
├── pt_001
│   ├── meas_000000001
│   │   ├── det_primary
│   │   └── det_secondary
│   └── meas_000000002
├── pt_002
└── ...
```

### 7.1 Measurement Event Attributes

```
measurement_counter        (global, monotonic)
timestamp_start
timestamp_end (optional)
measurement_status
point_ref
poni_ref
```

---

### 7.2 Detector-Level Data

```
detector_id
integration_time_ms
beam_energy_keV

raw_files/
raw_signal
```

---

## 8. Analytical Measurements

Analytical measurements are physical detector acquisitions performed for analytical or correction purposes (e.g. attenuation). They are not derived or computed results and differ from regular measurements only by storage location and declared analytical purpose.

### 8.1 Storage Structure

Analytical measurements are not grouped by points.

```
/analytical_measurements
├── ana_000000901
├── ana_000000902
└── ...
```

---

### 8.2 Analytical Measurement Attributes

```
measurement_counter        (global, monotonic)
timestamp_start
timestamp_end (optional)
measurement_status
poni_ref
analysis_type              ("attenuation")
```

---

### 8.3 Detector-Level Data

Identical to regular measurements:

```
detector_id
integration_time_ms
beam_energy_keV

raw_files/
raw_signal
```

---

### 8.4 Association with Points

Points explicitly declare which analytical measurements apply to them:

```
/points/pt_001
    attributes:
        analytical_measurement_refs = [
            ref(/analytical_measurements/ana_000000901)
        ]
```

Analytical measurements do not reference points.

---

## 9. Measurement Counter

- Global, monotonic, never reset.
- Incremented for every detector acquisition.
- Preserved even for invalid or aborted measurements.
- Shared between measurements and analytical measurements.

---

## 10. Translation Stage Logic

- Points are defined in pixel coordinates.
- Pixel-to-mm conversion is derived from `/images/mapping`.
- Translation stage operates in millimeters.
- Motion commands are not stored in the container.

---

## 11. Raw Data Retention and Compression

- Detector TXT and DSC files are never deleted automatically.
- Daily raw files are archived into ZIP files.
- HDF5 containers are retained locally after upload.

Compression:
- Raw files: maximum compression
- Parsed numeric data: medium compression

---

## 12. Container Lifecycle Summary

1. Technical container created during calibration.
2. Session container created when a new sample starts.
3. `/technical` copied into session container.
4. Images, points, measurements, and analytical measurements appended.
5. Session container finalized/locked.
6. Session container sent (fake send currently in development mode).
7. Session container archived from measurements folder to session archive.
8. New sample → new session container.
9. New calibration → new technical container.

---

End of specification.
