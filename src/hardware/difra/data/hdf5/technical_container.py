"""DIFRA Technical Container Writer

Creates and updates technical_<id>.h5 containers that store:
- Detector configuration
- PONY/PONI calibration data
- Technical measurement events (DARK, EMPTY, BACKGROUND, AGBH, WATER)

This module provides the primary API for generating technical containers
from the DIFRA Technical Measurements UI.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from hardware.difra.data.hdf5 import io, schema_v1


def create_technical_container(
    folder: Union[str, Path],
    distance_cm: float,
    container_id: Optional[str] = None
) -> Tuple[str, str]:
    """Create a new empty technical container.
    
    Args:
        folder: Directory where container will be created
        distance_cm: Sample-detector distance in cm
        container_id: Optional 16-char hex ID (generated if not provided)
    
    Returns:
        Tuple of (container_id, file_path)
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    
    if container_id is None:
        container_id = schema_v1.generate_container_id()
    elif not schema_v1.validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")
    
    filename = schema_v1.format_technical_container_filename(container_id, distance_cm)
    file_path = str(folder / filename)
    
    root_attrs = {
        schema_v1.ATTR_CREATION_TIMESTAMP: time.strftime("%Y-%m-%d %H:%M:%S"),
        schema_v1.ATTR_DISTANCE_CM: distance_cm,
    }
    
    io.create_empty_container(
        file_path=file_path,
        container_id=container_id,
        container_type=schema_v1.CONTAINER_TYPE_TECHNICAL,
        root_attrs=root_attrs
    )
    
    # Create top-level groups
    io.create_group_if_missing(file_path, schema_v1.GROUP_TECHNICAL)
    io.create_group_if_missing(file_path, schema_v1.GROUP_TECHNICAL_CONFIG)
    io.create_group_if_missing(file_path, schema_v1.GROUP_TECHNICAL_PONY)
    
    return container_id, file_path


def write_detector_config(
    file_path: Union[str, Path],
    detectors_config: List[Dict],
    active_detector_ids: List[str]
) -> None:
    """Write detector configuration to /technical/config.
    
    Args:
        file_path: Technical container path
        detectors_config: List of detector config dicts from DIFRA config
        active_detector_ids: List of active detector IDs
    """
    # Build configuration structure
    config = {
        "detectors": [],
        "active_detector_ids": active_detector_ids,
        "roles": {},
        "spatial_arrangement": "placeholder"  # TODO: define matrix structure
    }
    
    for det_cfg in detectors_config:
        if det_cfg.get("id") not in active_detector_ids:
            continue
        
        alias = det_cfg.get("alias")
        role = schema_v1.format_detector_role(alias)
        
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
    
    # Store as JSON string dataset
    config_json = json.dumps(config, indent=2)
    io.write_dataset(
        file_path=file_path,
        dataset_path=f"{schema_v1.GROUP_TECHNICAL_CONFIG}/detector_config",
        data=config_json,
        compression=None,
        overwrite=True
    )


def write_pony_datasets(
    file_path: Union[str, Path],
    pony_data: Dict[str, Tuple[str, str]],
    distance_cm: float,
    operator_confirmed: bool = True
) -> None:
    """Write PONY calibration data to /technical/pony.
    
    Args:
        file_path: Technical container path
        pony_data: Dict mapping alias to (pony_content, pony_filename)
        distance_cm: Sample-detector distance
        operator_confirmed: Whether PONY is operator-confirmed
    """
    for alias, (pony_content, pony_filename) in pony_data.items():
        role = schema_v1.format_detector_role(alias)
        pony_path = f"{schema_v1.GROUP_TECHNICAL_PONY}/pony_{role[4:]}"  # Remove "det_" prefix
        
        attrs = {
            schema_v1.ATTR_DETECTOR_ID: alias,
            schema_v1.ATTR_DISTANCE_CM: distance_cm,
            schema_v1.ATTR_PONY_OPERATOR_CONFIRMED: operator_confirmed,
            "pony_filename": pony_filename,
        }
        
        io.write_dataset(
            file_path=file_path,
            dataset_path=pony_path,
            data=pony_content,
            attrs=attrs,
            compression=None,
            overwrite=True
        )


def add_technical_event(
    file_path: Union[str, Path],
    event_index: int,
    technical_type: str,
    measurements: Dict[str, Dict],
    timestamp: str,
    distance_cm: float
) -> str:
    """Add a technical measurement event to /technical/tech_evt_###.
    
    Args:
        file_path: Technical container path
        event_index: Event index (1-based)
        technical_type: Type (DARK, EMPTY, BACKGROUND, AGBH, WATER)
        measurements: Dict mapping alias to measurement data dict with keys:
                     - 'data': np.ndarray (2D detector image)
                     - 'detector_id': str
                     - 'timestamp': str
        timestamp: Event timestamp
        distance_cm: Sample-detector distance
    
    Returns:
        Event group path
    """
    if not schema_v1.validate_technical_type(technical_type):
        raise ValueError(f"Invalid technical type: {technical_type}")
    
    event_id = schema_v1.format_technical_event_id(event_index)
    event_path = f"{schema_v1.GROUP_TECHNICAL}/{event_id}"
    
    # Create event group
    io.create_group_if_missing(file_path, event_path)
    
    # Set event-level attributes
    event_attrs = {
        "type": technical_type,
        "timestamp_utc": timestamp,
        schema_v1.ATTR_DISTANCE_CM: distance_cm,
    }
    io.set_attrs(file_path, event_path, event_attrs)
    
    # Write per-detector measurements
    for alias, meas_data in measurements.items():
        role = schema_v1.format_detector_role(alias)
        detector_path = f"{event_path}/{role}"
        
        io.create_group_if_missing(file_path, detector_path)
        
        # Write raw_signal dataset
        raw_signal_path = f"{detector_path}/{schema_v1.DATASET_RAW_SIGNAL}"
        io.write_dataset(
            file_path=file_path,
            dataset_path=raw_signal_path,
            data=meas_data["data"],
            compression="gzip",
            compression_opts=9,  # Maximum compression for raw data
            overwrite=True
        )
        
        # Set detector group attributes
        attrs = {
            schema_v1.ATTR_TECHNICAL_TYPE: technical_type,
            schema_v1.ATTR_DISTANCE_CM: distance_cm,
            schema_v1.ATTR_TIMESTAMP: meas_data.get("timestamp", timestamp),
            schema_v1.ATTR_DETECTOR_ID: meas_data.get("detector_id", alias),
        }
        io.set_attrs(file_path, detector_path, attrs)
    
    return event_path


def link_pony_to_event(
    file_path: Union[str, Path],
    pony_alias: str,
    event_index: int
) -> None:
    """Link a PONY dataset to the technical event it was derived from.
    
    Args:
        file_path: Technical container path
        pony_alias: Detector alias (e.g. "PRIMARY")
        event_index: Technical event index
    """
    role = schema_v1.format_detector_role(pony_alias)
    pony_path = f"{schema_v1.GROUP_TECHNICAL_PONY}/pony_{role[4:]}"
    
    event_id = schema_v1.format_technical_event_id(event_index)
    event_path = f"{schema_v1.GROUP_TECHNICAL}/{event_id}"
    
    io.set_reference_attr(
        file_path=file_path,
        obj_path=pony_path,
        attr_name=schema_v1.ATTR_PONY_DERIVED_FROM,
        target_path=event_path
    )


def generate_from_aux_table(
    folder: Union[str, Path],
    aux_measurements: Dict[str, Dict[str, str]],
    pony_data: Dict[str, Tuple[str, str]],
    detector_config: List[Dict],
    active_detector_ids: List[str],
    distance_cm: float,
    container_id: Optional[str] = None
) -> Tuple[str, str]:
    """Generate technical container from DIFRA Aux table selections.
    
    This is the primary API for the Technical Measurements UI.
    
    Args:
        folder: Directory where container will be created
        aux_measurements: Dict structure:
            {
                "DARK": {"PRIMARY": "/path/to/dark_primary.npy", "SECONDARY": "/path/..."},
                "EMPTY": {...},
                "BACKGROUND": {...},
                "AGBH": {...},
            }
        pony_data: Dict mapping alias to (pony_content, pony_filename)
        detector_config: List of detector config dicts from DIFRA config
        active_detector_ids: List of active detector IDs
        distance_cm: Sample-detector distance in cm
        container_id: Optional container ID (generated if not provided)
    
    Returns:
        Tuple of (container_id, file_path)
    """
    # Create container
    container_id, file_path = create_technical_container(folder, distance_cm, container_id)
    
    # Write detector configuration
    write_detector_config(file_path, detector_config, active_detector_ids)
    
    # Write PONY datasets
    write_pony_datasets(file_path, pony_data, distance_cm)
    
    # Add technical events
    event_index = 1
    for tech_type in schema_v1.ALL_TECHNICAL_TYPES:
        if tech_type not in aux_measurements:
            continue
        
        alias_files = aux_measurements[tech_type]
        measurements = {}
        
        for alias, file_path_str in alias_files.items():
            try:
                data = np.load(file_path_str)
                measurements[alias] = {
                    "data": data,
                    "detector_id": alias,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            except Exception as e:
                raise RuntimeError(f"Failed to load measurement file {file_path_str}: {e}")
        
        if measurements:
            add_technical_event(
                file_path=file_path,
                event_index=event_index,
                technical_type=tech_type,
                measurements=measurements,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                distance_cm=distance_cm
            )
            event_index += 1
    
    return container_id, file_path


def find_active_technical_container(
    folder: Union[str, Path],
    distance_cm: Optional[float] = None
) -> Optional[str]:
    """Find the most recent technical container in a folder.
    
    Args:
        folder: Directory to search
        distance_cm: Optional distance filter
    
    Returns:
        Path to most recent technical container, or None if not found
    """
    folder = Path(folder)
    if not folder.exists():
        return None
    
    # Find all technical_*.h5 files
    pattern = "technical_*.h5"
    candidates = list(folder.glob(pattern))
    
    if not candidates:
        return None
    
    # Filter by distance if specified
    if distance_cm is not None:
        filtered = []
        for candidate in candidates:
            try:
                with io.open_h5_append(candidate) as f:
                    if f.attrs.get(schema_v1.ATTR_DISTANCE_CM) == distance_cm:
                        filtered.append(candidate)
            except Exception:
                continue
        candidates = filtered
    
    if not candidates:
        return None
    
    # Return most recent by modification time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])
