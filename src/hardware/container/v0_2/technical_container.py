"""DIFRA NeXus Technical Container Writer (v0.2)."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from . import schema, utils


def create_technical_container(
    folder: Union[str, Path],
    distance_cm: float,
    container_id: Optional[str] = None,
    producer_software: str = "difra",
    producer_version: str = "unknown",
) -> Tuple[str, str]:
    """Create an empty NeXus technical container."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if container_id is None:
        container_id = schema.generate_container_id()
    elif not schema.validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")

    filename = schema.format_technical_container_filename(container_id, distance_cm)
    file_path = str(folder / filename)

    root_attrs = {
        schema.ATTR_CREATION_TIMESTAMP: schema.now_timestamp(),
        schema.ATTR_DISTANCE_CM: distance_cm,
        schema.ATTR_PRODUCER_SOFTWARE: producer_software,
        schema.ATTR_PRODUCER_VERSION: producer_version,
    }

    utils.create_empty_container(
        file_path=file_path,
        container_id=container_id,
        container_type=schema.CONTAINER_TYPE_TECHNICAL,
        root_attrs=root_attrs,
    )

    utils.write_dataset(
        file_path=file_path,
        dataset_path=f"{schema.GROUP_ENTRY}/{schema.ATTR_ENTRY_DEFINITION}",
        data=schema.APPDEF_TECHNICAL,
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

    group_classes = [
        (schema.GROUP_TECHNICAL, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_TECHNICAL_CONFIG, schema.NX_CLASS_INSTRUMENT),
        (schema.GROUP_INSTRUMENT_DETECTORS, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_TECHNICAL_PONI, schema.NX_CLASS_COLLECTION),
        (schema.GROUP_RUNTIME, schema.NX_CLASS_COLLECTION),
    ]
    for group_path, nx_class in group_classes:
        utils.create_group_if_missing(file_path, group_path)
        utils.set_attrs(file_path, group_path, {schema.ATTR_NX_CLASS: nx_class})

    utils.set_attrs(
        file_path=file_path,
        path=schema.GROUP_RUNTIME,
        attrs={
            schema.ATTR_PRODUCER_SOFTWARE: producer_software,
            schema.ATTR_PRODUCER_VERSION: producer_version,
        },
    )

    return container_id, file_path


def write_detector_config(
    file_path: Union[str, Path],
    detectors_config: List[Dict],
    active_detector_ids: List[str],
) -> None:
    """Write detector config and NeXus detector groups."""
    config = {
        "detectors": [],
        "active_detector_ids": active_detector_ids,
        "roles": {},
        "spatial_arrangement": "placeholder",
    }

    for det_cfg in detectors_config:
        if det_cfg.get("id") not in active_detector_ids:
            continue

        alias = det_cfg.get("alias")
        role = schema.format_detector_role(alias)
        det_info = {
            "id": det_cfg.get("id"),
            "alias": alias,
            "role": role,
            "type": det_cfg.get("type"),
            "size": det_cfg.get("size"),
            "pixel_size_um": det_cfg.get("pixel_size_um"),
            "faulty_pixels_file": det_cfg.get("faulty_pixels"),
        }
        config["detectors"].append(det_info)
        config["roles"][alias] = role

        detector_group = f"{schema.GROUP_INSTRUMENT_DETECTORS}/{alias}"
        utils.create_group_if_missing(file_path, detector_group)
        utils.set_attrs(
            file_path,
            detector_group,
            {
                schema.ATTR_NX_CLASS: schema.NX_CLASS_DETECTOR,
                schema.ATTR_DETECTOR_ID: det_cfg.get("id", alias),
                schema.ATTR_DETECTOR_ALIAS: alias,
                "pixel_size_um": np.array(det_cfg.get("pixel_size_um", [])),
                "shape": json.dumps(det_cfg.get("size", {})),
            },
        )

    config_json = json.dumps(config, indent=2)
    utils.write_dataset(
        file_path=file_path,
        dataset_path=f"{schema.GROUP_TECHNICAL_CONFIG}/detector_config",
        data=config_json,
        compression=None,
        overwrite=True,
    )


def write_poni_datasets(
    file_path: Union[str, Path],
    poni_data: Dict[str, Tuple[str, str]],
    distances_cm: Union[float, Dict[str, float]],
    detector_id_by_alias: Optional[Dict[str, str]] = None,
    operator_confirmed: bool = True,
) -> None:
    detector_id_by_alias = detector_id_by_alias or {}
    for alias, (poni_content, poni_filename) in poni_data.items():
        role = schema.format_detector_role(alias)
        poni_path = f"{schema.GROUP_TECHNICAL_PONI}/poni_{role[4:]}"

        if isinstance(distances_cm, dict):
            distance_cm = distances_cm.get(alias, list(distances_cm.values())[0])
        else:
            distance_cm = distances_cm

        attrs = {
            schema.ATTR_NX_CLASS: schema.NX_CLASS_NOTE,
            schema.ATTR_DETECTOR_ID: detector_id_by_alias.get(alias, alias),
            schema.ATTR_DETECTOR_ALIAS: alias,
            schema.ATTR_DISTANCE_CM: distance_cm,
            schema.ATTR_PONI_OPERATOR_CONFIRMED: operator_confirmed,
            "poni_filename": poni_filename,
        }

        utils.write_dataset(
            file_path=file_path,
            dataset_path=poni_path,
            data=poni_content,
            attrs=attrs,
            compression=None,
            overwrite=True,
        )


def add_technical_event(
    file_path: Union[str, Path],
    event_index: int,
    technical_type: str,
    measurements: Dict[str, Dict],
    timestamp: str,
    distances_cm: Union[float, Dict[str, float]],
) -> str:
    if not schema.validate_technical_type(technical_type):
        raise ValueError(f"Invalid technical type: {technical_type}")

    event_id = schema.format_technical_event_id(event_index)
    event_path = f"{schema.GROUP_TECHNICAL}/{event_id}"
    utils.create_group_if_missing(file_path, event_path)

    if isinstance(distances_cm, dict):
        event_distance_cm = list(distances_cm.values())[0]
    else:
        event_distance_cm = distances_cm

    utils.set_attrs(
        file_path,
        event_path,
        {
            schema.ATTR_NX_CLASS: schema.NX_CLASS_PROCESS,
            "type": technical_type,
            schema.ATTR_TIMESTAMP: timestamp,
            schema.ATTR_DISTANCE_CM: event_distance_cm,
        },
    )

    for alias, meas_data in measurements.items():
        role = schema.format_detector_role(alias)
        detector_path = f"{event_path}/{role}"
        utils.create_group_if_missing(file_path, detector_path)

        detector_attrs = {
            schema.ATTR_NX_CLASS: schema.NX_CLASS_DETECTOR,
            schema.ATTR_TECHNICAL_TYPE: technical_type,
            schema.ATTR_TIMESTAMP: meas_data.get("timestamp", timestamp),
            schema.ATTR_DETECTOR_ID: meas_data.get("detector_id", alias),
            schema.ATTR_DETECTOR_ALIAS: alias,
            **(
                {"poni_path": f"{schema.GROUP_TECHNICAL_PONI}/poni_{role[4:]}"}
                if technical_type == schema.TECHNICAL_TYPE_AGBH
                else {}
            ),
            **({"source_file": meas_data.get("source_file")} if meas_data.get("source_file") else {}),
        }

        integration_time_ms = meas_data.get(schema.ATTR_INTEGRATION_TIME_MS)
        if integration_time_ms is not None:
            try:
                detector_attrs[schema.ATTR_INTEGRATION_TIME_MS] = float(integration_time_ms)
            except Exception:
                pass

        n_frames = meas_data.get(schema.ATTR_N_FRAMES)
        if n_frames is not None:
            try:
                detector_attrs[schema.ATTR_N_FRAMES] = int(n_frames)
            except Exception:
                pass

        thickness = meas_data.get(schema.ATTR_THICKNESS)
        if thickness is not None:
            try:
                detector_attrs[schema.ATTR_THICKNESS] = float(thickness)
            except Exception:
                detector_attrs[schema.ATTR_THICKNESS] = str(thickness)

        utils.set_attrs(file_path, detector_path, detector_attrs)

        if isinstance(distances_cm, dict):
            detector_distance_cm = distances_cm.get(alias, event_distance_cm)
        else:
            detector_distance_cm = distances_cm
        utils.set_attrs(
            file_path,
            detector_path,
            {
                schema.ATTR_DISTANCE_CM: detector_distance_cm,
                schema.ATTR_DETECTOR_DISTANCE_CM: detector_distance_cm,
            },
        )

        utils.write_dataset(
            file_path=file_path,
            dataset_path=f"{detector_path}/{schema.DATASET_PROCESSED_SIGNAL}",
            data=meas_data["data"],
            compression="gzip",
            compression_opts=schema.COMPRESSION_PROCESSED,
            overwrite=True,
        )

        source_file = meas_data.get("source_file")
        if source_file and os.path.exists(source_file):
            source_path = Path(source_file)
            base_name = source_path.stem
            source_dir = source_path.parent

            associated_files = []
            for ext in [".txt", ".dsc"]:
                candidate = source_dir / f"{base_name}{ext}"
                if candidate.exists():
                    associated_files.append(candidate)

            if associated_files:
                blob_group_path = f"{detector_path}/{schema.DATASET_BLOB}"
                utils.create_group_if_missing(file_path, blob_group_path)
                utils.set_attrs(
                    file_path,
                    blob_group_path,
                    {schema.ATTR_NX_CLASS: schema.NX_CLASS_COLLECTION},
                )

            for file_to_store in associated_files:
                with open(file_to_store, "rb") as file_handle:
                    raw_blob = file_handle.read()
                ext = file_to_store.suffix.lower()
                file_format = ext[1:] if ext else "unknown"
                blob_dataset_path = f"{blob_group_path}/raw_{file_format}"
                utils.write_dataset(
                    file_path=file_path,
                    dataset_path=blob_dataset_path,
                    data=np.frombuffer(raw_blob, dtype=np.uint8),
                    attrs={
                        "source_filename": file_to_store.name,
                        "file_format": file_format,
                        "blob_size_bytes": len(raw_blob),
                    },
                    compression="gzip",
                    compression_opts=9,
                    overwrite=True,
                )

    return event_path


def link_poni_to_event(
    file_path: Union[str, Path],
    poni_alias: str,
    event_index: int,
) -> None:
    role = schema.format_detector_role(poni_alias)
    poni_path = f"{schema.GROUP_TECHNICAL_PONI}/poni_{role[4:]}"

    event_id = schema.format_technical_event_id(event_index)
    event_path = f"{schema.GROUP_TECHNICAL}/{event_id}"

    utils.set_attrs(
        file_path=file_path,
        path=poni_path,
        attrs={schema.ATTR_PONI_DERIVED_FROM: event_id, "derived_from_event_path": event_path},
    )


def _parse_capture_metadata_from_filename(file_path: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    stem = Path(str(file_path)).stem

    integration_match = re.search(
        r"(?:^|_)(\d+(?:\.\d+)?)s(?:_|$)",
        stem,
        flags=re.IGNORECASE,
    )
    if integration_match:
        try:
            metadata[schema.ATTR_INTEGRATION_TIME_MS] = (
                float(integration_match.group(1)) * 1000.0
            )
        except Exception:
            pass

    frames_match = re.search(r"(?:^|_)(\d+)frames(?:_|$)", stem, flags=re.IGNORECASE)
    if frames_match:
        try:
            metadata[schema.ATTR_N_FRAMES] = int(frames_match.group(1))
        except Exception:
            pass

    thickness_match = re.search(
        r"(?:^|_)(\d+(?:\.\d+)?)mm(?:_|$)",
        stem,
        flags=re.IGNORECASE,
    )
    if thickness_match:
        try:
            metadata[schema.ATTR_THICKNESS] = float(thickness_match.group(1))
        except Exception:
            pass

    return metadata


def _normalize_aux_measurement_entry(
    entry: Union[str, Dict[str, Any]],
    *,
    default_thickness_mm: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    if isinstance(entry, str):
        file_path = entry
        metadata: Dict[str, Any] = {}
    elif isinstance(entry, dict):
        file_path = (
            entry.get("file_path")
            or entry.get("path")
            or entry.get("source_file")
            or ""
        )
        metadata = {}
        embedded = entry.get("metadata")
        if isinstance(embedded, dict):
            metadata.update(embedded)
        for key in (
            schema.ATTR_INTEGRATION_TIME_MS,
            "integration_time_s",
            schema.ATTR_N_FRAMES,
            "frames",
            schema.ATTR_THICKNESS,
            "thickness_mm",
        ):
            if key in entry and entry[key] is not None:
                metadata[key] = entry[key]
    else:
        raise ValueError(f"Unsupported measurement entry type: {type(entry)!r}")

    file_path = str(file_path)
    parsed = _parse_capture_metadata_from_filename(file_path)
    for key, value in parsed.items():
        metadata.setdefault(key, value)

    if "integration_time_s" in metadata and schema.ATTR_INTEGRATION_TIME_MS not in metadata:
        try:
            metadata[schema.ATTR_INTEGRATION_TIME_MS] = (
                float(metadata.pop("integration_time_s")) * 1000.0
            )
        except Exception:
            metadata.pop("integration_time_s", None)

    if "frames" in metadata and schema.ATTR_N_FRAMES not in metadata:
        try:
            metadata[schema.ATTR_N_FRAMES] = int(metadata.pop("frames"))
        except Exception:
            metadata.pop("frames", None)

    if "thickness_mm" in metadata and schema.ATTR_THICKNESS not in metadata:
        try:
            metadata[schema.ATTR_THICKNESS] = float(metadata.pop("thickness_mm"))
        except Exception:
            metadata.pop("thickness_mm", None)

    if default_thickness_mm is not None and schema.ATTR_THICKNESS not in metadata:
        metadata[schema.ATTR_THICKNESS] = float(default_thickness_mm)

    return file_path, metadata


def generate_from_aux_table(
    folder: Union[str, Path],
    aux_measurements: Dict[str, Dict[str, Union[str, Dict[str, Any]]]],
    poni_data: Dict[str, Tuple[str, str]],
    detector_config: List[Dict],
    active_detector_ids: List[str],
    distances_cm: Union[float, Dict[str, float]],
    poni_distances_cm: Optional[Union[float, Dict[str, float]]] = None,
    technical_thickness_mm: Optional[float] = None,
    container_id: Optional[str] = None,
    validate_poni: bool = True,
    poni_tolerance_percent: float = 5.0,
    producer_software: str = "difra",
    producer_version: str = "unknown",
) -> Tuple[str, str]:
    detector_id_by_alias = {
        cfg.get("alias"): cfg.get("id", cfg.get("alias"))
        for cfg in detector_config
        if cfg.get("alias")
    }

    if validate_poni and poni_data:
        if isinstance(distances_cm, (int, float)):
            distances_dict = {alias: float(distances_cm) for alias in poni_data.keys()}
        else:
            distances_dict = distances_cm

        for alias, (poni_content, _poni_filename) in poni_data.items():
            detector_distance = distances_dict.get(alias)
            if detector_distance is None:
                continue
            schema.validate_poni_distance(
                poni_content,
                detector_distance,
                tolerance_percent=poni_tolerance_percent,
            )

    if isinstance(distances_cm, dict):
        root_distance_cm = list(distances_cm.values())[0]
    else:
        root_distance_cm = distances_cm

    container_id, file_path = create_technical_container(
        folder=folder,
        distance_cm=root_distance_cm,
        container_id=container_id,
        producer_software=producer_software,
        producer_version=producer_version,
    )

    if poni_distances_cm is not None:
        with utils.open_h5_append(file_path) as f:
            if isinstance(poni_distances_cm, dict):
                f.attrs["poni_distances_cm_json"] = json.dumps(poni_distances_cm)
            else:
                f.attrs["poni_distance_cm"] = poni_distances_cm

            if isinstance(distances_cm, dict) and isinstance(poni_distances_cm, dict):
                all_verified = all(
                    abs(distances_cm.get(alias, 0) - poni_distances_cm.get(alias, 0)) < 0.1
                    for alias in distances_cm.keys()
                )
                f.attrs["distance_verified"] = all_verified
            elif not isinstance(distances_cm, dict) and not isinstance(poni_distances_cm, dict):
                f.attrs["distance_verified"] = abs(distances_cm - poni_distances_cm) < 0.1

    write_detector_config(file_path, detector_config, active_detector_ids)

    write_poni_datasets(
        file_path,
        poni_data,
        distances_cm,
        detector_id_by_alias=detector_id_by_alias,
    )

    event_index = 1
    agbh_event_indices = {}

    for tech_type in schema.ALL_TECHNICAL_TYPES:
        if tech_type not in aux_measurements:
            continue

        alias_files = aux_measurements[tech_type]
        measurements = {}

        for alias, measurement_entry in alias_files.items():
            file_path_str, capture_metadata = _normalize_aux_measurement_entry(
                measurement_entry,
                default_thickness_mm=technical_thickness_mm,
            )
            if not file_path_str.endswith(".npy"):
                raise ValueError(
                    f"Container v0.2 requires .npy files for detector '{alias}'. Got: {file_path_str}"
                )

            data = np.load(file_path_str)
            detector_id = detector_id_by_alias.get(alias, alias)
            measurements[alias] = {
                "data": data,
                "detector_id": detector_id,
                "timestamp": schema.now_timestamp(),
                "source_file": file_path_str,
                schema.ATTR_INTEGRATION_TIME_MS: capture_metadata.get(
                    schema.ATTR_INTEGRATION_TIME_MS
                ),
                schema.ATTR_N_FRAMES: capture_metadata.get(schema.ATTR_N_FRAMES),
                schema.ATTR_THICKNESS: capture_metadata.get(schema.ATTR_THICKNESS),
            }

        if measurements:
            add_technical_event(
                file_path=file_path,
                event_index=event_index,
                technical_type=tech_type,
                measurements=measurements,
                timestamp=schema.now_timestamp(),
                distances_cm=distances_cm,
            )

            if tech_type == "AGBH":
                for alias in measurements.keys():
                    agbh_event_indices[alias] = event_index

            event_index += 1

    if agbh_event_indices:
        for alias, evt_idx in agbh_event_indices.items():
            link_poni_to_event(file_path, alias, evt_idx)

    return container_id, file_path


def find_active_technical_container(
    folder: Union[str, Path],
    distance_cm: Optional[float] = None,
) -> Optional[str]:
    folder = Path(folder)
    if not folder.exists():
        return None

    candidates = list(folder.glob("technical_*.nxs.h5"))

    if distance_cm is not None:
        filtered = []
        for candidate in candidates:
            try:
                with utils.open_h5_readonly(candidate) as f:
                    if abs(float(f.attrs.get(schema.ATTR_DISTANCE_CM, 0.0)) - float(distance_cm)) < 0.5:
                        filtered.append(candidate)
            except Exception:
                continue
        candidates = filtered

    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])
