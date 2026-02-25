"""DIFRA NeXus Session Container Writer (v0.2)."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from . import schema, utils


def _string_list_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _read_string_list_attr(attrs, name: str) -> List[str]:
    value = attrs.get(name)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8", errors="replace")]
    if isinstance(value, np.ndarray):
        result = []
        for item in value.tolist():
            if isinstance(item, bytes):
                result.append(item.decode("utf-8", errors="replace"))
            else:
                result.append(str(item))
        return result
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _append_unique_string_attr(file_path: Union[str, Path], obj_path: str, attr_name: str, item: str) -> None:
    with utils.open_h5_append(file_path) as f:
        obj = f[obj_path]
        values = _read_string_list_attr(obj.attrs, attr_name)
        if item not in values:
            values.append(item)
        obj.attrs[attr_name] = np.array(values, dtype=_string_list_dtype())


def _set_nx_class(file_path: Union[str, Path], path: str, nx_class: str) -> None:
    utils.set_attrs(file_path=file_path, path=path, attrs={schema.ATTR_NX_CLASS: nx_class})


def _decode_attr(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_analysis_role(analysis_type: str, analysis_role: Optional[str]) -> str:
    if analysis_role:
        role = str(analysis_role).strip().lower()
        if role in {"i0", "without", "without_sample"}:
            return schema.ANALYSIS_ROLE_I0
        if role in {"i", "with", "with_sample"}:
            return schema.ANALYSIS_ROLE_I
        return role

    lowered = str(analysis_type or "").strip().lower()
    if lowered in {"attenuation_i0", "i0", "attenuation_without", "attenuation_without_sample"}:
        return schema.ANALYSIS_ROLE_I0
    if lowered in {"attenuation_i", "i", "attenuation_with", "attenuation_with_sample"}:
        return schema.ANALYSIS_ROLE_I
    return schema.ANALYSIS_ROLE_UNSPECIFIED


def _build_human_summary(
    *,
    sample_id: str,
    project_id: str,
    study_name: str,
    operator_id: str,
    machine_name: str,
    site_id: str,
    session_id: str,
    acquisition_date: str,
    creation_timestamp: str,
) -> str:
    lines = [
        f"Sample ID: {sample_id}",
        f"Project ID: {project_id}",
        f"Study Name: {study_name}",
        f"Operator ID: {operator_id}",
        f"Machine: {machine_name}",
        f"Site ID: {site_id}",
        f"Session ID: {session_id}",
        f"Acquisition Date: {acquisition_date}",
        f"Created At: {creation_timestamp}",
    ]
    return "\n".join(lines)


def refresh_human_summary(file_path: Union[str, Path]) -> str:
    """Rebuild and persist user-readable session summary."""
    with utils.open_h5_append(file_path) as f:
        sample_group = f.get(schema.GROUP_SAMPLE)
        user_group = f.get(schema.GROUP_USER)

        sample_id = _decode_attr(
            f.attrs.get(schema.ATTR_SAMPLE_ID)
            or (sample_group.attrs.get(schema.ATTR_SAMPLE_ID) if sample_group else None),
            "unknown",
        )
        study_name = _decode_attr(
            f.attrs.get(schema.ATTR_STUDY_NAME)
            or (sample_group.attrs.get(schema.ATTR_STUDY_NAME) if sample_group else None),
            "UNSPECIFIED",
        )
        project_id = _decode_attr(
            f.attrs.get(schema.ATTR_PROJECT_ID)
            or (sample_group.attrs.get(schema.ATTR_PROJECT_ID) if sample_group else None),
            study_name,
        )
        operator_id = _decode_attr(
            f.attrs.get(schema.ATTR_OPERATOR_ID)
            or (user_group.attrs.get(schema.ATTR_OPERATOR_ID) if user_group else None),
            "unknown",
        )
        machine_name = _decode_attr(
            f.attrs.get(schema.ATTR_MACHINE_NAME)
            or (user_group.attrs.get(schema.ATTR_MACHINE_NAME) if user_group else None),
            "unknown",
        )
        site_id = _decode_attr(
            f.attrs.get(schema.ATTR_SITE_ID)
            or (user_group.attrs.get(schema.ATTR_SITE_ID) if user_group else None),
            "unknown",
        )
        session_id = _decode_attr(f.attrs.get(schema.ATTR_SESSION_ID), "unknown")
        acquisition_date = _decode_attr(f.attrs.get(schema.ATTR_ACQUISITION_DATE), "")
        creation_timestamp = _decode_attr(f.attrs.get(schema.ATTR_CREATION_TIMESTAMP), "")

        summary = _build_human_summary(
            sample_id=sample_id,
            project_id=project_id,
            study_name=study_name,
            operator_id=operator_id,
            machine_name=machine_name,
            site_id=site_id,
            session_id=session_id,
            acquisition_date=acquisition_date,
            creation_timestamp=creation_timestamp,
        )

        f.attrs[schema.ATTR_PROJECT_ID] = project_id
        f.attrs[schema.ATTR_HUMAN_SUMMARY] = summary
        summary_dataset = f"{schema.GROUP_ENTRY}/{schema.ATTR_HUMAN_SUMMARY}"
        if summary_dataset in f:
            del f[summary_dataset]
        f.create_dataset(
            summary_dataset,
            data=summary,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    return summary


def append_runtime_log(
    file_path: Union[str, Path],
    message: str,
    level: str = "INFO",
    event_type: str = "event",
    source: str = "difra",
    timestamp: Optional[str] = None,
    details: Optional[Dict] = None,
) -> None:
    """Append a human-readable log line to /entry/difra_runtime/session_log."""
    if timestamp is None:
        timestamp = schema.now_timestamp()

    details = details or {}
    details_json = json.dumps(details, ensure_ascii=False, separators=(",", ":"))
    line = f"[{timestamp}] [{level.upper()}] [{source}] {event_type}: {message}"
    if details:
        line = f"{line} | {details_json}"

    log_path = f"{schema.GROUP_RUNTIME}/{schema.DATASET_SESSION_LOG}"
    with utils.open_h5_append(file_path) as f:
        existing = ""
        if log_path in f:
            raw = f[log_path][()]
            if isinstance(raw, np.ndarray):
                existing = raw.tobytes().decode("utf-8", errors="replace")
            elif isinstance(raw, bytes):
                existing = raw.decode("utf-8", errors="replace")
            else:
                existing = str(raw)

        text = f"{existing}\n{line}" if existing else line
        payload = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)

    utils.write_dataset(
        file_path=file_path,
        dataset_path=log_path,
        data=payload,
        attrs={
            schema.ATTR_NX_CLASS: schema.NX_CLASS_NOTE,
            "encoding": "utf-8",
            "format": "text/plain",
            "line_count": int(text.count("\n") + 1),
        },
        compression="gzip",
        compression_opts=schema.COMPRESSION_BLOB_MAX,
        overwrite=True,
    )


def create_session_container(
    folder: Union[str, Path],
    sample_id: str,
    operator_id: str,
    site_id: str,
    machine_name: str,
    beam_energy_keV: float,
    acquisition_date: str,
    patient_id: Optional[str] = None,
    study_name: str = "UNSPECIFIED",
    project_id: Optional[str] = None,
    container_id: Optional[str] = None,
    producer_software: str = "difra",
    producer_version: str = "unknown",
) -> Tuple[str, str]:
    """Create a new NeXus session container."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if container_id is None:
        container_id = schema.generate_container_id()
    elif not schema.validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")

    filename = schema.format_session_container_filename(container_id, sample_id)
    file_path = str(folder / filename)
    resolved_project_id = project_id or study_name

    root_attrs = {
        schema.ATTR_SAMPLE_ID: sample_id,
        schema.ATTR_STUDY_NAME: study_name,
        schema.ATTR_PROJECT_ID: resolved_project_id,
        schema.ATTR_SESSION_ID: container_id,
        schema.ATTR_CREATION_TIMESTAMP: schema.now_timestamp(),
        schema.ATTR_ACQUISITION_DATE: acquisition_date,
        schema.ATTR_OPERATOR_ID: operator_id,
        schema.ATTR_SITE_ID: site_id,
        schema.ATTR_MACHINE_NAME: machine_name,
        schema.ATTR_BEAM_ENERGY_KEV: beam_energy_keV,
        schema.ATTR_PRODUCER_SOFTWARE: producer_software,
        schema.ATTR_PRODUCER_VERSION: producer_version,
    }

    if patient_id is not None:
        root_attrs[schema.ATTR_PATIENT_ID] = patient_id

    utils.create_empty_container(
        file_path=file_path,
        container_id=container_id,
        container_type=schema.CONTAINER_TYPE_SESSION,
        root_attrs=root_attrs,
    )

    # NXentry metadata
    utils.write_dataset(
        file_path=file_path,
        dataset_path=f"{schema.GROUP_ENTRY}/{schema.ATTR_ENTRY_DEFINITION}",
        data=schema.APPDEF_SESSION,
        compression=None,
        overwrite=True,
    )
    utils.write_dataset(
        file_path=file_path,
        dataset_path=f"{schema.GROUP_ENTRY}/{schema.ATTR_START_TIME}",
        data=schema.now_timestamp(),
        compression=None,
        overwrite=True,
    )
    utils.set_attrs(
        file_path=file_path,
        path=schema.GROUP_ENTRY,
        attrs={schema.ATTR_ENTRY_DEFAULT: "images", schema.ATTR_NX_CLASS: schema.NX_CLASS_ENTRY},
    )

    # Required groups
    for group_path, nx_class in [
        (schema.GROUP_SAMPLE, schema.NX_CLASS_SAMPLE),
        (schema.GROUP_USER, schema.NX_CLASS_USER),
        (schema.GROUP_INSTRUMENT, schema.NX_CLASS_INSTRUMENT),
        (schema.GROUP_IMAGES, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_IMAGES_ZONES, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_IMAGES_MAPPING, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_POINTS, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_MEASUREMENTS, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_ANALYTICAL_MEASUREMENTS, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_CALIBRATION_SNAPSHOT, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_TECHNICAL_CONFIG, schema.NX_CLASS_INSTRUMENT),
        (schema.GROUP_TECHNICAL_PONI, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_RUNTIME, schema.NX_CLASS_COLLECTION),
    ]:
        utils.create_group_if_missing(file_path, group_path)
        _set_nx_class(file_path, group_path, nx_class)

    # Mirror important attributes on semantic groups.
    utils.set_attrs(
        file_path=file_path,
        path=schema.GROUP_SAMPLE,
        attrs={
            schema.ATTR_SAMPLE_ID: sample_id,
            schema.ATTR_STUDY_NAME: study_name,
            schema.ATTR_PROJECT_ID: resolved_project_id,
            **({schema.ATTR_PATIENT_ID: patient_id} if patient_id is not None else {}),
        },
    )
    utils.set_attrs(
        file_path=file_path,
        path=schema.GROUP_USER,
        attrs={
            schema.ATTR_OPERATOR_ID: operator_id,
            schema.ATTR_SITE_ID: site_id,
            schema.ATTR_MACHINE_NAME: machine_name,
        },
    )
    utils.set_attrs(
        file_path=file_path,
        path=schema.GROUP_INSTRUMENT,
        attrs={schema.ATTR_BEAM_ENERGY_KEV: beam_energy_keV},
    )

    # Initialize global measurement counter in runtime and root for compatibility.
    with utils.open_h5_append(file_path) as f:
        f.attrs["measurement_counter"] = 0
        runtime = f[schema.GROUP_RUNTIME]
        runtime.attrs["measurement_counter"] = 0
        runtime.attrs[schema.ATTR_PRODUCER_SOFTWARE] = producer_software
        runtime.attrs[schema.ATTR_PRODUCER_VERSION] = producer_version

    refresh_human_summary(file_path)
    append_runtime_log(
        file_path=file_path,
        source=producer_software,
        event_type="session_created",
        message="Session container created",
        details={
            "sample_id": sample_id,
            "project_id": resolved_project_id,
            "operator_id": operator_id,
            "machine_name": machine_name,
        },
    )

    return container_id, file_path


def copy_technical_to_session(
    technical_file: Union[str, Path],
    session_file: Union[str, Path],
    auto_lock: bool = False,
    user_confirm_lock: Optional[callable] = None,
) -> None:
    """Copy technical NeXus data into session calibration snapshot."""
    from . import container_manager

    technical_file = Path(technical_file)

    if not technical_file.exists():
        raise FileNotFoundError(f"Technical container not found: {technical_file}")

    is_locked = container_manager.is_container_locked(technical_file)

    if not is_locked:
        should_lock = False
        if auto_lock:
            should_lock = True
        elif user_confirm_lock is not None:
            should_lock = user_confirm_lock(technical_file)

        if should_lock:
            container_manager.lock_container(technical_file)

    with h5py.File(technical_file, "r") as src, h5py.File(session_file, "a") as dst:
        snapshot_path = schema.GROUP_CALIBRATION_SNAPSHOT
        if snapshot_path in dst:
            del dst[snapshot_path]

        source_technical_path = None
        for candidate in (schema.GROUP_TECHNICAL, "/technical"):
            if candidate in src:
                source_technical_path = candidate
                break
        if source_technical_path is None:
            raise KeyError(
                f"Technical container is missing required group: {schema.GROUP_TECHNICAL}"
            )

        src.copy(source_technical_path, dst, name=snapshot_path)
        snapshot = dst[snapshot_path]
        if schema.ATTR_NX_CLASS not in snapshot.attrs:
            snapshot.attrs[schema.ATTR_NX_CLASS] = schema.NX_CLASS_COLLECTION
        snapshot.attrs["source_file"] = str(technical_file)
        snapshot.attrs["copied_timestamp"] = schema.now_timestamp()
        snapshot.attrs["source_container_id"] = src.attrs.get(schema.ATTR_CONTAINER_ID, "")


def add_detector_data_with_blobs(
    file_path: Union[str, Path],
    detector_path: str,
    processed_signal: np.ndarray,
    raw_files: Dict[str, bytes],
    poni_ref_path: Optional[str] = None,
) -> None:
    """Add detector data and raw blobs."""
    utils.create_group_if_missing(file_path, detector_path)
    _set_nx_class(file_path, detector_path, schema.NX_CLASS_DETECTOR)

    processed_path = f"{detector_path}/{schema.DATASET_PROCESSED_SIGNAL}"
    utils.write_dataset(
        file_path=file_path,
        dataset_path=processed_path,
        data=processed_signal,
        compression="gzip",
        compression_opts=schema.COMPRESSION_PROCESSED,
        overwrite=True,
    )

    blob_group = f"{detector_path}/{schema.DATASET_BLOB}"
    utils.create_group_if_missing(file_path, blob_group)
    _set_nx_class(file_path, blob_group, schema.NX_CLASS_COLLECTION)

    for blob_key, content in (raw_files or {}).items():
        if not blob_key.startswith("raw_"):
            import os

            _, ext = os.path.splitext(blob_key)
            blob_key = f"raw_{ext[1:] if ext else 'unknown'}"

        blob_path = f"{blob_group}/{blob_key}"
        if isinstance(content, bytes):
            blob_data = np.frombuffer(content, dtype=np.uint8)
        elif isinstance(content, np.ndarray):
            blob_data = content
        else:
            raise TypeError(f"Raw file content must be bytes or numpy array, got {type(content)}")

        utils.write_dataset(
            file_path=file_path,
            dataset_path=blob_path,
            data=blob_data,
            compression="gzip",
            compression_opts=schema.COMPRESSION_BLOB_MAX,
            overwrite=True,
        )

    if poni_ref_path:
        utils.set_attrs(
            file_path=file_path,
            path=detector_path,
            attrs={"poni_path": poni_ref_path},
        )


def add_image(
    file_path: Union[str, Path],
    image_index: int,
    image_data: Union[np.ndarray, str],
    image_type: str = schema.IMAGE_TYPE_SAMPLE,
    timestamp: Optional[str] = None,
) -> str:
    if timestamp is None:
        timestamp = schema.now_timestamp()

    image_id = schema.format_image_id(image_index)
    image_path = f"{schema.GROUP_IMAGES}/{image_id}"

    utils.create_group_if_missing(file_path, image_path)
    _set_nx_class(file_path, image_path, schema.NX_CLASS_DATA)

    if isinstance(image_data, str):
        image_data = np.load(image_data)

    utils.write_dataset(
        file_path=file_path,
        dataset_path=f"{image_path}/data",
        data=image_data,
        compression="gzip",
        compression_opts=schema.COMPRESSION_IMAGE,
        overwrite=True,
    )

    utils.set_attrs(
        file_path=file_path,
        path=image_path,
        attrs={schema.ATTR_IMAGE_TYPE: image_type, schema.ATTR_TIMESTAMP: timestamp},
    )

    return image_path


def add_zone(
    file_path: Union[str, Path],
    zone_index: int,
    zone_role: str,
    geometry_px: Union[List, np.ndarray, str],
    shape: str = "polygon",
    holder_diameter_mm: Optional[float] = None,
) -> str:
    if not schema.validate_zone_role(zone_role):
        raise ValueError(f"Invalid zone role: {zone_role}")

    zone_id = schema.format_zone_id(zone_index)
    zone_path = f"{schema.GROUP_IMAGES_ZONES}/{zone_id}"

    utils.create_group_if_missing(file_path, zone_path)
    _set_nx_class(file_path, zone_path, schema.NX_CLASS_COLLECTION)

    if isinstance(geometry_px, str):
        utils.write_dataset(
            file_path=file_path,
            dataset_path=f"{zone_path}/geometry_px",
            data=geometry_px,
            compression=None,
            overwrite=True,
        )
    else:
        utils.write_dataset(
            file_path=file_path,
            dataset_path=f"{zone_path}/geometry_px",
            data=np.array(geometry_px),
            compression="gzip",
            overwrite=True,
        )

    attrs = {schema.ATTR_ZONE_ROLE: zone_role, schema.ATTR_ZONE_SHAPE: shape}
    if holder_diameter_mm is not None:
        attrs[schema.ATTR_HOLDER_DIAMETER_MM] = holder_diameter_mm

    utils.set_attrs(file_path, zone_path, attrs)

    return zone_path


def add_image_mapping(
    file_path: Union[str, Path],
    sample_holder_zone_id: str,
    pixel_to_mm_conversion: Dict,
    orientation: str = "standard",
    mapping_version: str = "0.2",
) -> str:
    mapping_data = {
        "sample_holder_zone_id": sample_holder_zone_id,
        "pixel_to_mm_conversion": pixel_to_mm_conversion,
        "orientation": orientation,
        "mapping_version": mapping_version,
    }

    mapping_json = json.dumps(mapping_data, indent=2)
    mapping_path = f"{schema.GROUP_IMAGES_MAPPING}/mapping"

    utils.write_dataset(
        file_path=file_path,
        dataset_path=mapping_path,
        data=mapping_json,
        compression=None,
        overwrite=True,
    )
    _set_nx_class(file_path, schema.GROUP_IMAGES_MAPPING, schema.NX_CLASS_NOTE)
    return mapping_path


def add_point(
    file_path: Union[str, Path],
    point_index: int,
    pixel_coordinates: List[float],
    physical_coordinates_mm: List[float],
    point_status: str = schema.POINT_STATUS_PENDING,
    thickness: Optional[str] = schema.THICKNESS_UNKNOWN,
) -> str:
    point_id = schema.format_point_id(point_index)
    point_path = f"{schema.GROUP_POINTS}/{point_id}"

    utils.create_group_if_missing(file_path, point_path)
    _set_nx_class(file_path, point_path, schema.NX_CLASS_COLLECTION)

    thickness_value = (
        str(thickness).strip()
        if thickness is not None and str(thickness).strip()
        else schema.THICKNESS_UNKNOWN
    )

    utils.set_attrs(
        file_path=file_path,
        path=point_path,
        attrs={
            schema.ATTR_PIXEL_COORDINATES: np.array(pixel_coordinates),
            schema.ATTR_PHYSICAL_COORDINATES_MM: np.array(physical_coordinates_mm),
            schema.ATTR_POINT_STATUS: point_status,
            schema.ATTR_THICKNESS: thickness_value,
        },
    )

    with utils.open_h5_append(file_path) as f:
        f[point_path].attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_IDS] = np.array(
            [], dtype=_string_list_dtype()
        )

    return point_path


def update_point_status(
    file_path: Union[str, Path],
    point_index: int,
    point_status: str,
    skip_reason: Optional[str] = None,
) -> None:
    point_id = schema.format_point_id(point_index)
    point_path = f"{schema.GROUP_POINTS}/{point_id}"

    attrs = {schema.ATTR_POINT_STATUS: point_status}
    if point_status == schema.POINT_STATUS_SKIPPED:
        reason_text = str(skip_reason or "").strip() or "unspecified"
        attrs[schema.ATTR_SKIP_REASON] = reason_text
    utils.set_attrs(file_path=file_path, path=point_path, attrs=attrs)

    # Keep skip_reason only for skipped points.
    if point_status != schema.POINT_STATUS_SKIPPED:
        try:
            with utils.open_h5_append(file_path) as f:
                point = f[point_path]
                if schema.ATTR_SKIP_REASON in point.attrs:
                    del point.attrs[schema.ATTR_SKIP_REASON]
        except Exception:
            pass


def get_next_measurement_counter(file_path: Union[str, Path]) -> int:
    with utils.open_h5_append(file_path) as f:
        runtime = f[schema.GROUP_RUNTIME]
        counter = int(runtime.attrs.get("measurement_counter", f.attrs.get("measurement_counter", 0)))
        next_counter = counter + 1
        runtime.attrs["measurement_counter"] = next_counter
        f.attrs["measurement_counter"] = next_counter
    return next_counter


def begin_measurement(
    file_path: Union[str, Path],
    point_index: int,
    timestamp_start: Optional[str] = None,
    measurement_status: str = schema.STATUS_IN_PROGRESS,
) -> str:
    """Create a measurement group with start timestamp before detector payload is written.

    This is used to preserve in-container recovery info in case the app crashes mid-capture.
    """
    if timestamp_start is None:
        timestamp_start = schema.now_timestamp()

    meas_counter = get_next_measurement_counter(file_path)
    point_id = schema.format_point_id(point_index)
    meas_id = schema.format_measurement_id(meas_counter)
    meas_path = f"{schema.GROUP_MEASUREMENTS}/{point_id}/{meas_id}"

    utils.create_group_if_missing(file_path, meas_path)
    _set_nx_class(file_path, meas_path, schema.NX_CLASS_DATA)
    utils.set_attrs(
        file_path=file_path,
        path=meas_path,
        attrs={
            schema.ATTR_MEASUREMENT_COUNTER: meas_counter,
            schema.ATTR_TIMESTAMP_START: timestamp_start,
            schema.ATTR_MEASUREMENT_STATUS: measurement_status,
            schema.ATTR_POINT_REF: point_id,
        },
    )

    return meas_path


def _write_measurement_detector_payload(
    file_path: Union[str, Path],
    measurement_path: str,
    measurement_data: Dict[str, np.ndarray],
    detector_metadata: Dict[str, Dict],
    poni_alias_map: Dict[str, str],
    raw_files: Optional[Dict[str, Dict[str, bytes]]] = None,
) -> None:
    raw_files = raw_files or {}

    for detector_id, processed_signal in measurement_data.items():
        alias = next((a for a, d in poni_alias_map.items() if d == detector_id), detector_id)
        role = schema.format_detector_role(alias)
        detector_path = f"{measurement_path}/{role}"

        metadata = detector_metadata.get(detector_id) if detector_metadata else None
        detector_raw_files = raw_files.get(detector_id, {})

        det_attrs = {
            schema.ATTR_DETECTOR_ID: detector_id,
            schema.ATTR_DETECTOR_ALIAS: alias,
        }
        if metadata:
            det_attrs[schema.ATTR_INTEGRATION_TIME_MS] = metadata.get("integration_time_ms", 0)
            if "beam_energy_keV" in metadata:
                det_attrs[schema.ATTR_BEAM_ENERGY_KEV] = metadata["beam_energy_keV"]

        poni_path = f"{schema.GROUP_CALIBRATION_SNAPSHOT}/poni/poni_{role[4:]}"
        add_detector_data_with_blobs(
            file_path=file_path,
            detector_path=detector_path,
            processed_signal=processed_signal,
            raw_files=detector_raw_files,
            poni_ref_path=poni_path,
        )
        utils.set_attrs(file_path, detector_path, det_attrs)


def finalize_measurement(
    file_path: Union[str, Path],
    measurement_path: str,
    measurement_data: Dict[str, np.ndarray],
    detector_metadata: Dict[str, Dict],
    poni_alias_map: Dict[str, str],
    raw_files: Optional[Dict[str, Dict[str, bytes]]] = None,
    timestamp_end: Optional[str] = None,
    measurement_status: str = schema.STATUS_COMPLETED,
    failure_reason: Optional[str] = None,
) -> str:
    """Finalize a pre-created measurement group by writing payload and terminal status."""
    if timestamp_end is None and measurement_status != schema.STATUS_IN_PROGRESS:
        timestamp_end = schema.now_timestamp()

    attrs = {schema.ATTR_MEASUREMENT_STATUS: measurement_status}
    if timestamp_end is not None:
        attrs[schema.ATTR_TIMESTAMP_END] = timestamp_end
    if failure_reason:
        attrs[schema.ATTR_FAILURE_REASON] = str(failure_reason)
    utils.set_attrs(file_path=file_path, path=measurement_path, attrs=attrs)

    if measurement_data:
        _write_measurement_detector_payload(
            file_path=file_path,
            measurement_path=measurement_path,
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            raw_files=raw_files,
        )

    return measurement_path


def fail_measurement(
    file_path: Union[str, Path],
    measurement_path: str,
    failure_reason: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    measurement_status: str = schema.STATUS_FAILED,
) -> str:
    """Mark measurement as failed/aborted without detector payload."""
    return finalize_measurement(
        file_path=file_path,
        measurement_path=measurement_path,
        measurement_data={},
        detector_metadata={},
        poni_alias_map={},
        raw_files=None,
        timestamp_end=timestamp_end,
        measurement_status=measurement_status,
        failure_reason=failure_reason,
    )


def add_measurement(
    file_path: Union[str, Path],
    point_index: int,
    measurement_data: Dict[str, np.ndarray],
    detector_metadata: Dict[str, Dict],
    poni_alias_map: Dict[str, str],
    raw_files: Optional[Dict[str, Dict[str, bytes]]] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    measurement_status: str = schema.STATUS_COMPLETED,
) -> str:
    meas_path = begin_measurement(
        file_path=file_path,
        point_index=point_index,
        timestamp_start=timestamp_start,
        measurement_status=schema.STATUS_IN_PROGRESS,
    )
    return finalize_measurement(
        file_path=file_path,
        measurement_path=meas_path,
        measurement_data=measurement_data,
        detector_metadata=detector_metadata,
        poni_alias_map=poni_alias_map,
        raw_files=raw_files,
        timestamp_end=timestamp_end,
        measurement_status=measurement_status,
    )


def add_analytical_measurement(
    file_path: Union[str, Path],
    measurement_data: Dict[str, np.ndarray],
    detector_metadata: Dict[str, Dict],
    poni_alias_map: Dict[str, str],
    analysis_type: str,
    analysis_role: Optional[str] = None,
    raw_files: Optional[Dict[str, Dict[str, bytes]]] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    measurement_status: str = schema.STATUS_COMPLETED,
) -> str:
    if timestamp_start is None:
        timestamp_start = schema.now_timestamp()

    meas_counter = get_next_measurement_counter(file_path)

    ana_id = schema.format_analytical_measurement_id(meas_counter)
    ana_path = f"{schema.GROUP_ANALYTICAL_MEASUREMENTS}/{ana_id}"
    normalized_role = _normalize_analysis_role(analysis_type, analysis_role)

    utils.create_group_if_missing(file_path, ana_path)
    _set_nx_class(file_path, ana_path, schema.NX_CLASS_DATA)

    attrs = {
        schema.ATTR_MEASUREMENT_COUNTER: meas_counter,
        schema.ATTR_TIMESTAMP_START: timestamp_start,
        schema.ATTR_MEASUREMENT_STATUS: measurement_status,
        schema.ATTR_ANALYSIS_TYPE: analysis_type,
        schema.ATTR_ANALYSIS_ROLE: normalized_role,
    }
    if timestamp_end is not None:
        attrs[schema.ATTR_TIMESTAMP_END] = timestamp_end

    utils.set_attrs(file_path, ana_path, attrs)
    utils.set_reference_list_attr(
        file_path=file_path,
        obj_path=ana_path,
        attr_name=schema.ATTR_POINT_REFS,
        target_paths=[],
    )
    with utils.open_h5_append(file_path) as f:
        f[ana_path].attrs[schema.ATTR_POINT_IDS] = np.array([], dtype=_string_list_dtype())

    raw_files = raw_files or {}

    for detector_id, processed_signal in measurement_data.items():
        alias = next((a for a, d in poni_alias_map.items() if d == detector_id), detector_id)
        role = schema.format_detector_role(alias)
        detector_path = f"{ana_path}/{role}"

        metadata = detector_metadata.get(detector_id) if detector_metadata else None
        detector_raw_files = raw_files.get(detector_id, {})

        det_attrs = {
            schema.ATTR_DETECTOR_ID: detector_id,
            schema.ATTR_DETECTOR_ALIAS: alias,
        }
        if metadata:
            det_attrs[schema.ATTR_INTEGRATION_TIME_MS] = metadata.get("integration_time_ms", 0)
            if "beam_energy_keV" in metadata:
                det_attrs[schema.ATTR_BEAM_ENERGY_KEV] = metadata["beam_energy_keV"]

        poni_path = f"{schema.GROUP_CALIBRATION_SNAPSHOT}/poni/poni_{role[4:]}"
        add_detector_data_with_blobs(
            file_path=file_path,
            detector_path=detector_path,
            processed_signal=processed_signal,
            raw_files=detector_raw_files,
            poni_ref_path=poni_path,
        )
        utils.set_attrs(file_path, detector_path, det_attrs)

    return ana_path


def link_analytical_measurement_to_point(
    file_path: Union[str, Path],
    point_index: int,
    analytical_measurement_index: int,
) -> None:
    point_id = schema.format_point_id(point_index)
    point_path = f"{schema.GROUP_POINTS}/{point_id}"
    ana_id = schema.format_analytical_measurement_id(analytical_measurement_index)
    ana_path = f"{schema.GROUP_ANALYTICAL_MEASUREMENTS}/{ana_id}"

    _append_unique_string_attr(
        file_path=file_path,
        obj_path=point_path,
        attr_name=schema.ATTR_ANALYTICAL_MEASUREMENT_IDS,
        item=ana_id,
    )
    utils.append_reference_to_list_attr(
        file_path=file_path,
        obj_path=point_path,
        attr_name=schema.ATTR_ANALYTICAL_MEASUREMENT_REFS,
        target_path=ana_path,
    )
    utils.append_reference_to_list_attr(
        file_path=file_path,
        obj_path=ana_path,
        attr_name=schema.ATTR_POINT_REFS,
        target_path=point_path,
    )
    _append_unique_string_attr(
        file_path=file_path,
        obj_path=ana_path,
        attr_name=schema.ATTR_POINT_IDS,
        item=point_id,
    )


def find_active_session_container(
    folder: Union[str, Path], sample_id: Optional[str] = None
) -> Optional[str]:
    folder = Path(folder)
    if not folder.exists():
        return None

    candidates = list(folder.glob("session_*.nxs.h5"))

    if sample_id:
        candidates = [path for path in candidates if sample_id in path.name]

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])
