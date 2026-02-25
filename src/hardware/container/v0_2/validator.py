"""Session NeXus container validator for v0.2."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py

from . import schema

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a validation message."""

    def __init__(self, severity: str, path: str, message: str):
        self.severity = severity
        self.path = path
        self.message = message

    def __repr__(self) -> str:
        return f"{self.severity} [{self.path}]: {self.message}"


class SessionContainerValidator:
    """Validates session NeXus containers."""

    def __init__(self, session_file: Union[str, Path]):
        self.session_file = Path(session_file)
        self.errors: List[ValidationError] = []

    def _add(self, severity: str, path: str, message: str):
        self.errors.append(ValidationError(severity, path, message))

    def validate(self) -> Tuple[bool, List[ValidationError]]:
        self.errors = []

        try:
            with h5py.File(self.session_file, "r") as f:
                self._validate_root(f)
                self._validate_entry(f)
                self._validate_required_groups(f)
                self._validate_measurement_counter(f)
                self._validate_points(f)
                self._validate_measurements(f)
                self._validate_analytical_measurements(f)
                self._validate_analytical_links(f)
                self._validate_measurement_counter_monotonic(f)
                self._validate_lock_invariants(f)
        except Exception as error:
            self._add("ERROR", "/", f"Failed to open container: {error}")

        has_errors = any(item.severity == "ERROR" for item in self.errors)
        return not has_errors, self.errors

    def _validate_root(self, f: h5py.File):
        required = [
            schema.ATTR_CONTAINER_ID,
            schema.ATTR_CONTAINER_TYPE,
            schema.ATTR_SCHEMA_VERSION,
            schema.ATTR_CREATION_TIMESTAMP,
            schema.ATTR_NX_CLASS,
            schema.ATTR_SAMPLE_ID,
            schema.ATTR_STUDY_NAME,
            schema.ATTR_SESSION_ID,
            schema.ATTR_ACQUISITION_DATE,
            schema.ATTR_OPERATOR_ID,
            schema.ATTR_SITE_ID,
            schema.ATTR_MACHINE_NAME,
            schema.ATTR_BEAM_ENERGY_KEV,
        ]
        for attr in required:
            if attr not in f.attrs:
                self._add("ERROR", "/", f"Missing required root attribute: {attr}")

        if f.attrs.get(schema.ATTR_NX_CLASS) != schema.NX_CLASS_ROOT:
            self._add("ERROR", "/", "Root NX_class must be NXroot")

        if f.attrs.get(schema.ATTR_CONTAINER_TYPE) != schema.CONTAINER_TYPE_SESSION:
            self._add("ERROR", "/", "container_type must be session")

        if str(f.attrs.get(schema.ATTR_SCHEMA_VERSION, "")) != schema.SCHEMA_VERSION:
            self._add(
                "ERROR",
                "/",
                f"schema_version must be {schema.SCHEMA_VERSION}",
            )
        for producer_attr in (
            schema.ATTR_PRODUCER_SOFTWARE,
            schema.ATTR_PRODUCER_VERSION,
        ):
            if producer_attr not in f.attrs:
                self._add("WARNING", "/", f"Missing producer metadata: {producer_attr}")

    def _validate_entry(self, f: h5py.File):
        if schema.GROUP_ENTRY not in f:
            self._add("ERROR", schema.GROUP_ENTRY, "Missing entry group")
            return

        entry = f[schema.GROUP_ENTRY]
        if entry.attrs.get(schema.ATTR_NX_CLASS) != schema.NX_CLASS_ENTRY:
            self._add("ERROR", schema.GROUP_ENTRY, "NX_class must be NXentry")

        definition_path = f"{schema.GROUP_ENTRY}/{schema.ATTR_ENTRY_DEFINITION}"
        if definition_path not in f:
            self._add("ERROR", definition_path, "Missing /entry/definition")
        else:
            value = f[definition_path][()]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if value != schema.APPDEF_SESSION:
                self._add("ERROR", definition_path, f"Must be {schema.APPDEF_SESSION}, got {value}")

    def _validate_required_groups(self, f: h5py.File):
        required_groups = [
            schema.GROUP_SAMPLE,
            schema.GROUP_USER,
            schema.GROUP_INSTRUMENT,
            schema.GROUP_IMAGES,
            schema.GROUP_IMAGES_ZONES,
            schema.GROUP_IMAGES_MAPPING,
            schema.GROUP_POINTS,
            schema.GROUP_MEASUREMENTS,
            schema.GROUP_ANALYTICAL_MEASUREMENTS,
            schema.GROUP_CALIBRATION_SNAPSHOT,
            schema.GROUP_TECHNICAL_CONFIG,
            schema.GROUP_TECHNICAL_PONI,
            schema.GROUP_RUNTIME,
        ]
        for path in required_groups:
            if path not in f:
                self._add("ERROR", path, "Missing required group")

        # Calibration snapshot completeness
        if schema.GROUP_CALIBRATION_SNAPSHOT in f:
            snapshot = f[schema.GROUP_CALIBRATION_SNAPSHOT]
            for attr in ("source_file", "source_container_id", "copied_timestamp"):
                if attr not in snapshot.attrs:
                    self._add("ERROR", schema.GROUP_CALIBRATION_SNAPSHOT, f"Missing '{attr}'")
            has_events = any(
                key.startswith("tech_evt_") for key in snapshot.keys()
            )
            if not has_events:
                self._add(
                    "ERROR",
                    schema.GROUP_CALIBRATION_SNAPSHOT,
                    "Missing technical events (tech_evt_*) in copied technical group",
                )

    def _validate_measurement_counter(self, f: h5py.File):
        if schema.GROUP_RUNTIME not in f:
            return
        runtime = f[schema.GROUP_RUNTIME]
        if "measurement_counter" not in runtime.attrs:
            self._add("ERROR", schema.GROUP_RUNTIME, "Missing measurement_counter in runtime attrs")

    def _validate_points(self, f: h5py.File):
        if schema.GROUP_POINTS not in f:
            return
        points_group = f[schema.GROUP_POINTS]
        for point_id in points_group.keys():
            point_path = f"{schema.GROUP_POINTS}/{point_id}"
            point = points_group[point_id]
            for attr in [
                schema.ATTR_PIXEL_COORDINATES,
                schema.ATTR_PHYSICAL_COORDINATES_MM,
                schema.ATTR_POINT_STATUS,
            ]:
                if attr not in point.attrs:
                    self._add("WARNING", point_path, f"Missing attribute: {attr}")
            if schema.ATTR_THICKNESS not in point.attrs:
                self._add("ERROR", point_path, f"Missing attribute: {schema.ATTR_THICKNESS}")
            else:
                thickness = point.attrs.get(schema.ATTR_THICKNESS)
                if isinstance(thickness, bytes):
                    thickness = thickness.decode("utf-8", errors="replace")
                thickness = str(thickness).strip()
                if not thickness:
                    self._add("ERROR", point_path, f"Empty attribute: {schema.ATTR_THICKNESS}")
            if schema.ATTR_ANALYTICAL_MEASUREMENT_IDS not in point.attrs:
                self._add(
                    "WARNING",
                    point_path,
                    f"Missing attribute: {schema.ATTR_ANALYTICAL_MEASUREMENT_IDS}",
                )

            point_status = point.attrs.get(schema.ATTR_POINT_STATUS, "")
            if isinstance(point_status, bytes):
                point_status = point_status.decode("utf-8", errors="replace")
            point_status = str(point_status).strip().lower()
            if point_status == str(schema.POINT_STATUS_SKIPPED).strip().lower():
                if schema.ATTR_SKIP_REASON not in point.attrs:
                    self._add(
                        "ERROR",
                        point_path,
                        f"Missing attribute for skipped point: {schema.ATTR_SKIP_REASON}",
                    )
                else:
                    skip_reason = point.attrs.get(schema.ATTR_SKIP_REASON, "")
                    if isinstance(skip_reason, bytes):
                        skip_reason = skip_reason.decode("utf-8", errors="replace")
                    if not str(skip_reason).strip():
                        self._add(
                            "ERROR",
                            point_path,
                            f"Empty attribute for skipped point: {schema.ATTR_SKIP_REASON}",
                        )

    def _validate_measurements(self, f: h5py.File):
        if schema.GROUP_MEASUREMENTS not in f:
            return

        for point_id, point_group in f[schema.GROUP_MEASUREMENTS].items():
            point_path = f"{schema.GROUP_MEASUREMENTS}/{point_id}"
            for meas_id, meas_group in point_group.items():
                meas_path = f"{point_path}/{meas_id}"
                for attr in [
                    schema.ATTR_MEASUREMENT_COUNTER,
                    schema.ATTR_TIMESTAMP_START,
                    schema.ATTR_MEASUREMENT_STATUS,
                ]:
                    if attr not in meas_group.attrs:
                        self._add("WARNING", meas_path, f"Missing attribute: {attr}")

                detector_groups = [name for name in meas_group.keys() if name.startswith("det_")]
                if not detector_groups:
                    self._add("ERROR", meas_path, "No detector groups found")
                for detector_name in detector_groups:
                    detector_path = f"{meas_path}/{detector_name}"
                    detector_group = meas_group[detector_name]
                    if schema.DATASET_PROCESSED_SIGNAL not in detector_group:
                        self._add(
                            "ERROR",
                            detector_path,
                            f"Missing {schema.DATASET_PROCESSED_SIGNAL} dataset",
                        )

    def _validate_analytical_measurements(self, f: h5py.File):
        if schema.GROUP_ANALYTICAL_MEASUREMENTS not in f:
            return

        for ana_id, ana_group in f[schema.GROUP_ANALYTICAL_MEASUREMENTS].items():
            ana_path = f"{schema.GROUP_ANALYTICAL_MEASUREMENTS}/{ana_id}"
            for attr in [
                schema.ATTR_MEASUREMENT_COUNTER,
                schema.ATTR_TIMESTAMP_START,
                schema.ATTR_MEASUREMENT_STATUS,
                schema.ATTR_ANALYSIS_TYPE,
            ]:
                if attr not in ana_group.attrs:
                    self._add("WARNING", ana_path, f"Missing attribute: {attr}")
            if schema.ATTR_ANALYSIS_ROLE not in ana_group.attrs:
                self._add("WARNING", ana_path, f"Missing attribute: {schema.ATTR_ANALYSIS_ROLE}")
            else:
                analysis_type = ana_group.attrs.get(schema.ATTR_ANALYSIS_TYPE, "")
                analysis_role = ana_group.attrs.get(schema.ATTR_ANALYSIS_ROLE, "")
                if isinstance(analysis_type, bytes):
                    analysis_type = analysis_type.decode("utf-8", errors="replace")
                if isinstance(analysis_role, bytes):
                    analysis_role = analysis_role.decode("utf-8", errors="replace")
                if (
                    analysis_type == schema.ANALYSIS_TYPE_ATTENUATION
                    and analysis_role == schema.ANALYSIS_ROLE_UNSPECIFIED
                ):
                    self._add(
                        "WARNING",
                        ana_path,
                        "attenuation analytical measurement should specify analysis_role=i0|i",
                    )
            if schema.ATTR_POINT_IDS not in ana_group.attrs:
                self._add("WARNING", ana_path, f"Missing attribute: {schema.ATTR_POINT_IDS}")

            detector_groups = [name for name in ana_group.keys() if name.startswith("det_")]
            if not detector_groups:
                self._add("ERROR", ana_path, "No detector groups found")
            for detector_name in detector_groups:
                detector_path = f"{ana_path}/{detector_name}"
                detector_group = ana_group[detector_name]
                if schema.DATASET_PROCESSED_SIGNAL not in detector_group:
                    self._add(
                        "ERROR",
                        detector_path,
                        f"Missing {schema.DATASET_PROCESSED_SIGNAL} dataset",
                    )

    def _validate_analytical_links(self, f: h5py.File):
        if schema.GROUP_POINTS not in f or schema.GROUP_ANALYTICAL_MEASUREMENTS not in f:
            return

        valid_ids = set(f[schema.GROUP_ANALYTICAL_MEASUREMENTS].keys())
        for point_id, point_group in f[schema.GROUP_POINTS].items():
            point_path = f"{schema.GROUP_POINTS}/{point_id}"
            values = point_group.attrs.get(schema.ATTR_ANALYTICAL_MEASUREMENT_IDS)
            if values is None:
                continue

            if isinstance(values, (str, bytes)):
                values = [values]
            else:
                values = list(values)

            for value in values:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                value = str(value)
                if value and value not in valid_ids:
                    self._add(
                        "ERROR",
                        point_path,
                        f"Unknown analytical measurement id referenced: {value}",
                    )

            refs_attr = point_group.attrs.get(schema.ATTR_ANALYTICAL_MEASUREMENT_REFS)
            if refs_attr is not None:
                refs = [refs_attr] if isinstance(refs_attr, h5py.Reference) else list(refs_attr)
                for ref in refs:
                    try:
                        target = f[ref]
                    except Exception:
                        self._add("ERROR", point_path, "Invalid analytical measurement reference")
                        continue
                    if not target.name.startswith(f"{schema.GROUP_ANALYTICAL_MEASUREMENTS}/"):
                        self._add(
                            "ERROR",
                            point_path,
                            f"Reference target outside analytical measurements: {target.name}",
                        )

        for ana_id, ana_group in f[schema.GROUP_ANALYTICAL_MEASUREMENTS].items():
            ana_path = f"{schema.GROUP_ANALYTICAL_MEASUREMENTS}/{ana_id}"
            point_refs = ana_group.attrs.get(schema.ATTR_POINT_REFS)
            if point_refs is None:
                continue
            refs = [point_refs] if isinstance(point_refs, h5py.Reference) else list(point_refs)
            for ref in refs:
                try:
                    target = f[ref]
                except Exception:
                    self._add("ERROR", ana_path, "Invalid point reference")
                    continue
                if not target.name.startswith(f"{schema.GROUP_POINTS}/"):
                    self._add(
                        "ERROR",
                        ana_path,
                        f"Reference target outside points: {target.name}",
                    )

    def _validate_measurement_counter_monotonic(self, f: h5py.File):
        counters = []
        if schema.GROUP_MEASUREMENTS in f:
            for _point_id, point_group in f[schema.GROUP_MEASUREMENTS].items():
                for _meas_id, meas_group in point_group.items():
                    if schema.ATTR_MEASUREMENT_COUNTER in meas_group.attrs:
                        counters.append(int(meas_group.attrs[schema.ATTR_MEASUREMENT_COUNTER]))
        if schema.GROUP_ANALYTICAL_MEASUREMENTS in f:
            for _ana_id, ana_group in f[schema.GROUP_ANALYTICAL_MEASUREMENTS].items():
                if schema.ATTR_MEASUREMENT_COUNTER in ana_group.attrs:
                    counters.append(int(ana_group.attrs[schema.ATTR_MEASUREMENT_COUNTER]))

        if not counters:
            return

        sorted_counters = sorted(counters)
        unique_counters = sorted(set(counters))
        if unique_counters != sorted_counters:
            self._add("ERROR", schema.GROUP_RUNTIME, "measurement_counter values are not unique")
        if unique_counters != list(range(min(unique_counters), max(unique_counters) + 1)):
            self._add("WARNING", schema.GROUP_RUNTIME, "measurement_counter sequence has gaps")

        if schema.GROUP_RUNTIME in f:
            runtime_counter = int(f[schema.GROUP_RUNTIME].attrs.get("measurement_counter", 0))
            if runtime_counter < max(unique_counters):
                self._add(
                    "ERROR",
                    schema.GROUP_RUNTIME,
                    f"runtime measurement_counter {runtime_counter} is smaller than max used {max(unique_counters)}",
                )

    def _validate_lock_invariants(self, f: h5py.File):
        runtime = f.get(schema.GROUP_RUNTIME)
        if runtime is None:
            return
        locked = bool(runtime.attrs.get(schema.ATTR_LOCKED, f.attrs.get("locked", False)))
        if not locked:
            return
        if schema.ATTR_LOCKED_TIMESTAMP not in runtime.attrs and "locked_timestamp" not in f.attrs:
            self._add("ERROR", schema.GROUP_RUNTIME, "Locked container missing lock timestamp")
        if schema.ATTR_LOCKED_BY not in runtime.attrs and "locked_by" not in f.attrs:
            self._add("WARNING", schema.GROUP_RUNTIME, "Locked container missing lock owner")
