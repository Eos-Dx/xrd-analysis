# DIFRA Container Adapter Contract v0.2

## 1. Purpose

This document is the implementation contract for collaborators who need to:
- unpack DIFRA containers (`.nxs.h5` or `.zip` bundles),
- understand the schema without reverse-engineering code,
- import data into an external database safely and deterministically.

Target schema version: `0.2`.

## 2. Core Rules (Read This First)

1. Treat `/entry/...` as canonical for all scientific/business data.
2. Treat root (`/`) as container metadata + NeXus root attributes (not measurement trees).
3. For attenuation analytics, do not rely on ordering; read `analysis_role` (`i0` or `i`) explicitly.
4. Preserve human-readable metadata (`sample_id`, `project_id`, `operator_id`, `machine_name`) during archive/import.
5. Keep bidirectional links between points and analytical measurements (`*_ids` and `*_refs`).

## 3. Supported Input Artifacts

### 3.1 Single container file
- Typically named `*.nxs.h5` (regular HDF5 file).
- Contains one container: session or technical.

### 3.2 ZIP bundle
- Contains at least one `.h5` file (including `.nxs.h5`).
- May include raw/operator payload files.
- Selection rule:
1. extract ZIP,
2. discover `*.h5`,
3. select first by key `(len(path.parts), path.as_posix())`.

## 4. Version and Type Detection

Read root attributes:
- `schema_version` (expected `"0.2"`),
- `container_type` (`"session"` or `"technical"`),
- `NX_class` (expected `NXroot`).

Fallback marker for v0.2:
- root `NX_class == NXroot`,
- `/entry` exists,
- `/entry/definition` exists.

## 5. Canonical Layout

All canonical groups are under `/entry`.

### 5.1 Session container (`container_type=session`)
- `/entry/sample`
- `/entry/user`
- `/entry/instrument`
- `/entry/images`
- `/entry/images/zones`
- `/entry/images/mapping`
- `/entry/points`
- `/entry/measurements`
- `/entry/analytical_measurements`
- `/entry/technical`
- `/entry/technical/config`
- `/entry/technical/poni`
- `/entry/difra_runtime`

### 5.2 Technical container (`container_type=technical`)
- `/entry/technical`
- `/entry/technical/config`
- `/entry/technical/config/detectors`
- `/entry/technical/poni`
- `/entry/difra_runtime`

## 6. Root Metadata Contract

Always parse these root attributes:
- `container_id`
- `container_type`
- `schema_version`
- `creation_timestamp`
- `NX_class`
- `producer_software`
- `producer_version`

Session root attributes:
- `sample_id`
- `study_name`
- `project_id` (if absent in legacy, fallback to `study_name`)
- `session_id`
- `acquisition_date`
- `operator_id`
- `site_id`
- `machine_name`
- `beam_energy_keV`
- optional `patient_id`
- optional `human_summary`

Technical root attributes:
- `distance_cm`

## 7. Human-Readable Metadata and Machine Naming

### 7.1 Human-readable metadata retained for archive
Session containers must keep readable IDs in root attrs and `human_summary`, including:
- sample ID,
- project ID,
- study name,
- operator ID,
- machine name,
- site ID,
- session ID.

### 7.2 Machine naming source (important)
`machine_name` is not the host/computer nickname.  
It is resolved from active setup/config identity in this priority:
1. `config.machine_name`
2. `config.setup_name`
3. `config.name` (setup JSON `name`, e.g. `Ulster (Xena)`, `Ulster (Moli)`)
4. `config.default_setup`
5. fallback `DIFRA-01`

So for adapters, `machine_name` should be treated as installation/setup identity.

## 7.3 Runtime Software Log in Container

Session runtime log is stored in:
- `/entry/difra_runtime/session_log`

Format:
- UTF-8 text payload stored as compressed byte array (`gzip`, high compression).
- One line per event with timestamp/level/source/event_type.

Usage:
- If measurement started and app crashed, container log + measurement status provide audit trail.
- Container is self-describing without relying only on external log files.

## 8. File Naming and Archive Naming (Human-Readable)

### 8.1 Container filenames
- Session: `session_<container_id>_<sample_id>_<YYYYMMDD>.nxs.h5`
- Technical: `technical_<container_id>_<distanceToken>cm_<YYYYMMDD>.nxs.h5`
  - `distanceToken` example: `17p00` for `17.00 cm`

### 8.2 Session archive folder naming
Raw session payload archive directory pattern:
- `<sample>_<project>_<operator>_<YYYYMMDD_HHMMSS>`
- Fallback forms drop missing tokens (project/operator).

### 8.3 Session container archival tree naming
Archived session container directory pattern:
- `<sessionIdOrStem>_<operator>_<YYYYMMDD_HHMMSS>/`
- session `.nxs.h5` is moved inside this directory.

### 8.4 Technical archive naming
Two valid patterns exist:
- low-level archive move: `<stem>_archived_<operator>_<YYYYMMDD_HHMMSS>.nxs.h5`
- GUI workflow archive folder: `<container_id>_<operator>_<YYYYMMDD_HHMMSS>/` (contains `.nxs.h5` and archived payload)

Operator token is intentionally present to make archive browsing easier.

## 9. Entity Mapping for DB Import

### 9.1 Points
Path:
- `/entry/points/pt_###`

Key attrs:
- `pixel_coordinates`
- `physical_coordinates_mm`
- `point_status`
- `thickness` (required; use `unknown` when not measured, e.g. current DiFRA flow)
- `analytical_measurement_ids`
- `analytical_measurement_refs`

### 9.2 Regular measurements
Path:
- `/entry/measurements/pt_###/meas_#########`

Attrs:
- `measurement_counter`
- `timestamp_start`
- optional `timestamp_end`
- `measurement_status` (`in_progress`, `completed`, `failed`, `aborted`)
- `point_ref` (point ID string like `pt_001`)
- optional `failure_reason`

Detector groups:
- `det_*` (for example `det_primary`, `det_secondary`, `det_saxs`)

Detector attrs:
- `detector_id`
- `detector_alias`
- optional `integration_time_ms`
- optional `beam_energy_keV`
- `poni_path` (string path to copied PONI dataset)

Detector datasets:
- `processed_signal`
- optional `blob/raw_*` datasets

Crash-recovery rule:
- if `measurement_status == in_progress` and `timestamp_end` is missing, measurement started but did not finish (e.g. software crash).

### 9.3 Analytical measurements
Path:
- `/entry/analytical_measurements/ana_#########`

Attrs:
- `measurement_counter`
- `timestamp_start`
- optional `timestamp_end`
- `measurement_status`
- `analysis_type`
- `analysis_role` (`i0`, `i`, `unspecified`)
- `point_ids`
- `point_refs`

Detector semantics are identical to regular measurements.

### 9.4 Technical events
Path:
- `/entry/technical/tech_evt_######`

Event attrs:
- `type` (`DARK`, `EMPTY`, `BACKGROUND`, `AGBH`, optional `WATER`)
- `timestamp`
- `distance_cm`

Detector attrs:
- `technical_type`
- `timestamp`
- `detector_id`
- `detector_alias`
- `distance_cm`
- `detector_distance_cm`
- optional `poni_path` (typically for `AGBH`)
- optional `source_file`

Detector datasets:
- `processed_signal`
- optional `blob/raw_*`

### 9.5 PONI calibration datasets
Path:
- `/entry/technical/poni/poni_*`

Attrs:
- `detector_id`
- `detector_alias`
- `distance_cm`
- `operator_confirmed`
- `poni_filename`
- optional `derived_from`
- optional `derived_from_event_path`

## 10. Attenuation Semantics (`I0` vs `I`)

For attenuation workflows:
- `analysis_type` should be `attenuation`.
- `analysis_role` must carry role semantics:
  - `i0` = beam/reference without sample,
  - `i` = with sample.

Writers may normalize legacy values:
- role-like types (`attenuation_i0`, `attenuation_i`, `with`, `without`, etc.) are mapped to explicit `analysis_role`.

Adapter requirement:
- consume `analysis_role` as ground truth for role.
- do not infer role purely by order.

## 11. Relationship Contract (Bidirectional)

Point -> Analytical:
- `analytical_measurement_ids` contains `ana_*` IDs,
- `analytical_measurement_refs` contains HDF5 refs.

Analytical -> Point:
- `point_ids` contains `pt_*` IDs,
- `point_refs` contains HDF5 refs.

Importer behavior:
- use IDs as stable external keys,
- use refs to validate in-file consistency.

## 12. Archive Payload Requirements

### 12.1 Session raw payload archive defaults
Expected included patterns:
- `*.txt`
- `*.dsc`
- `*.npy`
- `*.t3pa`
- `*.poni`
- `*_state.json`

### 12.2 Technical raw payload archive defaults
Expected included patterns:
- `*.txt`
- `*.dsc`
- `*.npy`
- `*.poni`

Important: `.poni` files are part of archival payload by default.

## 13. Minimal Import Algorithm

1. Open artifact (`.nxs.h5` directly, or unzip and locate `.h5`).
2. Detect `schema_version` and `container_type`.
3. Read root metadata and preserve IDs/text fields.
4. Use `/entry/...` as canonical structure.
5. Import:
   - points,
   - regular measurements,
   - analytical measurements (`analysis_type` + `analysis_role`),
   - point/analytical links (both directions),
   - technical snapshot or technical events + PONI datasets.
6. Validate consistency:
   - IDs resolve,
   - refs resolve,
   - attenuation rows have meaningful `analysis_role`.

## 14. Recommended Minimal Relational Model

Tables:
- `containers`
- `points`
- `measurements`
- `measurement_detectors`
- `analytical_measurements`
- `analytical_links`
- `technical_events`
- `technical_event_detectors`
- `poni_calibrations`

Core FKs:
- `measurements.point_id -> points.point_id`
- `measurement_detectors.measurement_id -> measurements.measurement_id`
- `analytical_links.analytical_id -> analytical_measurements.analytical_id`
- `analytical_links.point_id -> points.point_id`
- `technical_event_detectors.event_id -> technical_events.event_id`

## 15. Interoperability Notes for Collaborators

If collaborators follow this contract, they can implement an adapter without DIFRA internals:
- schema is explicit,
- IDs and links are deterministic,
- attenuation role semantics are explicit (`i0` vs `i`),
- archive naming is human-readable (includes operator token),
- payload expectations include calibration files (`.poni`).
