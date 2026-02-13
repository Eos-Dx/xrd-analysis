"""DIFRA Technical Container Writer

Creates and updates technical_<id>.h5 containers that store:
- Detector configuration
- PONI calibration data
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

from . import utils, schema


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
        container_id = schema.generate_container_id()
    elif not schema.validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")
    
    filename = schema.format_technical_container_filename(container_id, distance_cm)
    file_path = str(folder / filename)
    
    root_attrs = {
        schema.ATTR_CREATION_TIMESTAMP: time.strftime("%Y-%m-%d %H:%M:%S"),
        schema.ATTR_DISTANCE_CM: distance_cm,
    }
    
    utils.create_empty_container(
        file_path=file_path,
        container_id=container_id,
        container_type=schema.CONTAINER_TYPE_TECHNICAL,
        root_attrs=root_attrs
    )
    
    # Create top-level groups
    utils.create_group_if_missing(file_path, schema.GROUP_TECHNICAL)
    utils.create_group_if_missing(file_path, schema.GROUP_TECHNICAL_CONFIG)
    utils.create_group_if_missing(file_path, schema.GROUP_TECHNICAL_PONI)
    
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
    
    # Store as JSON string dataset
    config_json = json.dumps(config, indent=2)
    utils.write_dataset(
        file_path=file_path,
        dataset_path=f"{schema.GROUP_TECHNICAL_CONFIG}/detector_config",
        data=config_json,
        compression=None,
        overwrite=True
    )


def write_poni_datasets(
    file_path: Union[str, Path],
    poni_data: Dict[str, Tuple[str, str]],
    distances_cm: Union[float, Dict[str, float]],
    detector_id_by_alias: Optional[Dict[str, str]] = None,
    operator_confirmed: bool = True
) -> None:
    """Write PONI calibration data to /technical/poni.
    
    Args:
        file_path: Technical container path
        poni_data: Dict mapping alias to (poni_content, poni_filename)
        distances_cm: Sample-detector distance (float for single, dict for per-detector)
        detector_id_by_alias: Optional mapping alias -> hardware detector ID
        operator_confirmed: Whether PONI is operator-confirmed
    """
    detector_id_by_alias = detector_id_by_alias or {}
    for alias, (poni_content, poni_filename) in poni_data.items():
        role = schema.format_detector_role(alias)
        poni_path = f"{schema.GROUP_TECHNICAL_PONI}/poni_{role[4:]}"  # Remove "det_" prefix
        
        # Get distance for this detector
        if isinstance(distances_cm, dict):
            distance_cm = distances_cm.get(alias, list(distances_cm.values())[0])
        else:
            distance_cm = distances_cm
        
        attrs = {
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
            overwrite=True
        )


def add_technical_event(
    file_path: Union[str, Path],
    event_index: int,
    technical_type: str,
    measurements: Dict[str, Dict],
    timestamp: str,
    distances_cm: Union[float, Dict[str, float]]
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
                     - 'source_file': str (optional) - path to original raw file
        timestamp: Event timestamp
        distances_cm: Sample-detector distance (float for single, dict for per-detector)
    
    Returns:
        Event group path
    """
    if not schema.validate_technical_type(technical_type):
        raise ValueError(f"Invalid technical type: {technical_type}")
    
    event_id = schema.format_technical_event_id(event_index)
    event_path = f"{schema.GROUP_TECHNICAL}/{event_id}"
    
    # Create event group
    utils.create_group_if_missing(file_path, event_path)
    
    # Get primary distance for event-level attribute (use first detector if dict)
    if isinstance(distances_cm, dict):
        event_distance_cm = list(distances_cm.values())[0]  # Primary detector distance
    else:
        event_distance_cm = distances_cm
    
    # Set event-level attributes
    event_attrs = {
        "type": technical_type,
        "timestamp_utc": timestamp,
        schema.ATTR_DISTANCE_CM: event_distance_cm,
    }
    utils.set_attrs(file_path, event_path, event_attrs)
    
    # Write per-detector measurements
    for alias, meas_data in measurements.items():
        role = schema.format_detector_role(alias)
        detector_path = f"{event_path}/{role}"
        
        utils.create_group_if_missing(file_path, detector_path)
        
        # Write processed_signal dataset (only signal stored)
        processed_signal_path = f"{detector_path}/{schema.DATASET_PROCESSED_SIGNAL}"
        utils.write_dataset(
            file_path=file_path,
            dataset_path=processed_signal_path,
            data=meas_data["data"],
            compression="gzip",
            compression_opts=schema.COMPRESSION_PROCESSED,
            overwrite=True
        )
        
        # Write raw data blobs if source file is provided
        # Store only .txt (ASCII data) and .dsc (descriptor) as raw blobs
        # (.npy is processed data and already stored in processed_signal dataset)
        source_file = meas_data.get("source_file")
        if source_file and os.path.exists(source_file):
            from pathlib import Path
            import logging
            logger = logging.getLogger(__name__)
            
            source_path = Path(source_file)
            base_name = source_path.stem  # Remove extension
            source_dir = source_path.parent
            
            # Find associated RAW files: .txt (ASCII data) and .dsc (descriptor)
            # Skip .npy as it's processed data already in raw_signal dataset
            associated_files = []
            for ext in [".txt", ".dsc"]:
                candidate = source_dir / f"{base_name}{ext}"
                if candidate.exists():
                    associated_files.append(candidate)
            
            # Store each raw file in blob/ subfolder
            # Create blob group if we have files to store
            if associated_files:
                blob_group_path = f"{detector_path}/blob"
                utils.create_group_if_missing(file_path, blob_group_path)
            
            for file_to_store in associated_files:
                try:
                    with open(file_to_store, 'rb') as f:
                        raw_blob = f.read()
                    
                    # Determine file format
                    ext = file_to_store.suffix.lower()
                    file_format = ext[1:] if ext else "unknown"  # Remove leading dot
                    
                    # Create blob dataset inside blob/ group
                    blob_dataset_path = f"{blob_group_path}/raw_{file_format}"
                    utils.write_dataset(
                        file_path=file_path,
                        dataset_path=blob_dataset_path,
                        data=np.frombuffer(raw_blob, dtype=np.uint8),
                        attrs={
                            "source_filename": file_to_store.name,
                            "file_format": file_format,
                            "blob_size_bytes": len(raw_blob)
                        },
                        compression="gzip",
                        compression_opts=9,
                        overwrite=True
                    )
                    logger.info(f"Stored {file_format} blob in blob/: {file_to_store.name} ({len(raw_blob)} bytes)")
                except Exception as e:
                    # Non-fatal: log but continue
                    logger.warning(f"Failed to store {file_format} blob from {file_to_store}: {e}")
        
        # Get distance for this specific detector
        if isinstance(distances_cm, dict):
            detector_distance_cm = distances_cm.get(alias, event_distance_cm)
        else:
            detector_distance_cm = distances_cm
        
        # Set detector group attributes
        attrs = {
            schema.ATTR_TECHNICAL_TYPE: technical_type,
            schema.ATTR_DISTANCE_CM: detector_distance_cm,  # Per-detector distance
            schema.ATTR_DETECTOR_DISTANCE_CM: detector_distance_cm,  # Explicit per-detector attr
            schema.ATTR_TIMESTAMP: meas_data.get("timestamp", timestamp),
            schema.ATTR_DETECTOR_ID: meas_data.get("detector_id", alias),  # Alias for technical
            schema.ATTR_DETECTOR_ALIAS: alias,
        }
        if source_file:
            attrs["source_file"] = os.path.basename(source_file)
        
        utils.set_attrs(file_path, detector_path, attrs)
    
    return event_path


def link_poni_to_event(
    file_path: Union[str, Path],
    poni_alias: str,
    event_index: int
) -> None:
    """Link a PONI dataset to the technical event it was derived from.
    
    Args:
        file_path: Technical container path
        poni_alias: Detector alias (e.g. "PRIMARY")
        event_index: Technical event index
    """
    role = schema.format_detector_role(poni_alias)
    poni_path = f"{schema.GROUP_TECHNICAL_PONI}/poni_{role[4:]}"
    
    event_id = schema.format_technical_event_id(event_index)
    event_path = f"{schema.GROUP_TECHNICAL}/{event_id}"
    
    utils.set_reference_attr(
        file_path=file_path,
        obj_path=poni_path,
        attr_name=schema.ATTR_PONI_DERIVED_FROM,
        target_path=event_path
    )


def generate_from_aux_table(
    folder: Union[str, Path],
    aux_measurements: Dict[str, Dict[str, str]],
    poni_data: Dict[str, Tuple[str, str]],
    detector_config: List[Dict],
    active_detector_ids: List[str],
    distances_cm: Union[float, Dict[str, float]],
    poni_distances_cm: Optional[Union[float, Dict[str, float]]] = None,
    container_id: Optional[str] = None,
    validate_poni: bool = True,
    poni_tolerance_percent: float = 5.0,
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
        poni_data: Dict mapping alias to (poni_content, poni_filename)
        detector_config: List of detector config dicts from DIFRA config
        active_detector_ids: List of active detector IDs
        distances_cm: User-defined sample-detector distance(s) in cm (float for single, dict for per-detector)
        poni_distances_cm: Distance(s) from PONI file(s) in cm (optional)
        container_id: Optional container ID (generated if not provided)
        validate_poni: If True, validate PONI distances (default: True)
        poni_tolerance_percent: Maximum allowed deviation % (default: 5.0)
    
    Returns:
        Tuple of (container_id, file_path)
        
    Raises:
        ValueError: If PONI distance validation fails
    """
    import logging
    logger = logging.getLogger(__name__)
    detector_id_by_alias = {
        cfg.get("alias"): cfg.get("id", cfg.get("alias"))
        for cfg in detector_config
        if cfg.get("alias")
    }
    
    # STEP 1: Validate PONI distances against user distance(s)
    if validate_poni and poni_data:
        # Convert single distance to dict for uniform processing
        if isinstance(distances_cm, (int, float)):
            distances_dict = {alias: float(distances_cm) for alias in poni_data.keys()}
            logger.info(f"Validating PONI distances against user distance: {distances_cm:.2f} cm...")
        else:
            distances_dict = distances_cm
            logger.info(f"Validating PONI distances against per-detector distances...")
        
        for alias, (poni_content, poni_filename) in poni_data.items():
            # Get distance for this detector
            detector_distance = distances_dict.get(alias)
            if detector_distance is None:
                logger.warning(f"  ⚠ {alias}: No distance specified, skipping validation")
                continue
            
            try:
                schema.validate_poni_distance(
                    poni_content, 
                    detector_distance, 
                    tolerance_percent=poni_tolerance_percent
                )
                logger.info(f"  ✓ {alias}: {poni_filename} - distance {detector_distance:.2f} cm OK")
            except ValueError as e:
                logger.error(f"  ✗ {alias}: {poni_filename} - {e}")
                raise ValueError(
                    f"PONI validation failed for {alias} ({poni_filename}):\n{e}"
                )
    
    # Get root distance (use first detector distance if dict)
    if isinstance(distances_cm, dict):
        root_distance_cm = list(distances_cm.values())[0]
    else:
        root_distance_cm = distances_cm
    
    # Create container
    container_id, file_path = create_technical_container(folder, root_distance_cm, container_id)
    
    # Store poni_distances_cm in root attributes if provided
    if poni_distances_cm is not None:
        with utils.open_h5_append(file_path) as f:
            if isinstance(poni_distances_cm, dict):
                # Store as JSON for per-detector distances
                import json
                f.attrs["poni_distances_cm_json"] = json.dumps(poni_distances_cm)
            else:
                f.attrs["poni_distance_cm"] = poni_distances_cm
            
            # Verify distances match
            if isinstance(distances_cm, dict) and isinstance(poni_distances_cm, dict):
                all_verified = all(
                    abs(distances_cm.get(alias, 0) - poni_distances_cm.get(alias, 0)) < 0.1
                    for alias in distances_cm.keys()
                )
                f.attrs["distance_verified"] = all_verified
            elif not isinstance(distances_cm, dict) and not isinstance(poni_distances_cm, dict):
                f.attrs["distance_verified"] = abs(distances_cm - poni_distances_cm) < 0.1
    
    # Write detector configuration
    write_detector_config(file_path, detector_config, active_detector_ids)
    
    # Write PONI datasets
    write_poni_datasets(
        file_path,
        poni_data,
        distances_cm,
        detector_id_by_alias=detector_id_by_alias,
    )
    
    # Add technical events
    event_index = 1
    agbh_event_indices = {}  # Track AGBH event indices by alias for PONI linking
    
    for tech_type in schema.ALL_TECHNICAL_TYPES:
        if tech_type not in aux_measurements:
            continue
        
        alias_files = aux_measurements[tech_type]
        measurements = {}
        
        for alias, file_path_str in alias_files.items():
            try:
                # Container v0.1 strictly requires .npy files
                # Detectors must convert raw output using convert_to_container_format()
                if not file_path_str.endswith('.npy'):
                    raise ValueError(
                        f"Container v0.1 requires .npy files for detector '{alias}'.\n"
                        f"Got: {file_path_str}\n\n"
                        f"Solution: Use detector.convert_to_container_format(raw_file, '0.1') "
                        f"to convert raw detector output before passing to container writer.\n"
                        f"Example for Advacam detectors:\n"
                        f"  detector.convert_to_container_format('measurement.txt', '0.1') -> 'measurement.npy'"
                    )
                
                data = np.load(file_path_str)
                detector_id = detector_id_by_alias.get(alias, alias)
                measurements[alias] = {
                    "data": data,
                    "detector_id": detector_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_file": file_path_str,  # Pass source file for blob storage
                }
            except Exception as e:
                raise RuntimeError(f"Failed to load measurement file {file_path_str}: {e}")
        
        if measurements:
            event_path = add_technical_event(
                file_path=file_path,
                event_index=event_index,
                technical_type=tech_type,
                measurements=measurements,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                distances_cm=distances_cm
            )
            
            # Track AGBH events for PONI linking
            if tech_type == "AGBH":
                for alias in measurements.keys():
                    agbh_event_indices[alias] = event_index
            
            event_index += 1
    
    # Add PONI references to AGBH events
    if agbh_event_indices:
        for alias, evt_idx in agbh_event_indices.items():
            role = schema.format_detector_role(alias)
            poni_path = f"{schema.GROUP_TECHNICAL_PONI}/poni_{role[4:]}"
            
            event_id = schema.format_technical_event_id(evt_idx)
            event_path = f"{schema.GROUP_TECHNICAL}/{event_id}"
            detector_path = f"{event_path}/{role}"
            
            # Add PONI reference to event group
            try:
                utils.set_reference_attr(
                    file_path=file_path,
                    obj_path=event_path,
                    attr_name=f"poni_{role[4:]}_ref",
                    target_path=poni_path
                )
            except Exception:
                pass  # PONI may not exist for this alias
            
            # Add PONI reference to detector subgroup
            try:
                utils.set_reference_attr(
                    file_path=file_path,
                    obj_path=detector_path,
                    attr_name="poni_ref",
                    target_path=poni_path
                )
            except Exception:
                pass  # PONI may not exist for this alias
    
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
                with utils.open_h5_append(candidate) as f:
                    if f.attrs.get(schema.ATTR_DISTANCE_CM) == distance_cm:
                        filtered.append(candidate)
            except Exception:
                continue
        candidates = filtered
    
    if not candidates:
        return None
    
    # Return most recent by modification time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])
