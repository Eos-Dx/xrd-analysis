"""DIFRA Session Container Writer

Creates and updates session_<id>.h5 containers that store:
- Complete sample acquisition session data (sample_id, operator, beam energy, etc.)
- Copy of /technical group from technical container (PONI, detector config, technical measurements)
- Sample images and zone definitions
- Point locations and metadata
- Measurement data organized point-centrically
- Analytical measurements for corrections

Session containers are self-contained "freight wagons" that can be transported,
analyzed, and archived independently of their origin machine.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from . import utils, schema


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
    container_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Create a new empty session container with required root attributes.

    Args:
        folder: Directory where container will be created
        sample_id: Unique sample identifier
        study_name: Study name/identifier
        operator_id: ID/name of operator
        site_id: Site/location identifier
        machine_name: Name/ID of acquisition machine
        beam_energy_keV: Beam energy in keV
        acquisition_date: Acquisition date (ISO format or human-readable)
        patient_id: Optional patient identifier
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

    filename = schema.format_session_container_filename(container_id, sample_id)
    file_path = str(folder / filename)

    root_attrs = {
        schema.ATTR_SAMPLE_ID: sample_id,
        schema.ATTR_STUDY_NAME: study_name,
        schema.ATTR_SESSION_ID: container_id,
        schema.ATTR_CREATION_TIMESTAMP: time.strftime("%Y-%m-%d %H:%M:%S"),
        schema.ATTR_ACQUISITION_DATE: acquisition_date,
        schema.ATTR_OPERATOR_ID: operator_id,
        schema.ATTR_SITE_ID: site_id,
        schema.ATTR_MACHINE_NAME: machine_name,
        schema.ATTR_BEAM_ENERGY_KEV: beam_energy_keV,
    }

    if patient_id is not None:
        root_attrs[schema.ATTR_PATIENT_ID] = patient_id

    utils.create_empty_container(
        file_path=file_path,
        container_id=container_id,
        container_type=schema.CONTAINER_TYPE_SESSION,
        root_attrs=root_attrs,
    )

    # Create top-level groups
    utils.create_group_if_missing(file_path, schema.GROUP_IMAGES)
    utils.create_group_if_missing(file_path, schema.GROUP_IMAGES_ZONES)
    utils.create_group_if_missing(file_path, schema.GROUP_IMAGES_MAPPING)
    utils.create_group_if_missing(file_path, schema.GROUP_POINTS)
    utils.create_group_if_missing(file_path, schema.GROUP_MEASUREMENTS)
    utils.create_group_if_missing(file_path, schema.GROUP_ANALYTICAL_MEASUREMENTS)

    # Initialize measurement counter at 0
    with utils.open_h5_append(file_path) as f:
        f.attrs["measurement_counter"] = 0

    return container_id, file_path


def copy_technical_to_session(
    technical_file: Union[str, Path],
    session_file: Union[str, Path],
    auto_lock: bool = False,
    user_confirm_lock: Optional[callable] = None,
) -> None:
    """Copy /technical group from technical container to session container.
    
    If technical container is not locked, will prompt for locking.
    Locked containers can be reused by multiple sessions.

    Args:
        technical_file: Path to technical container
        session_file: Path to session container
        auto_lock: If True, automatically lock unlocked containers
        user_confirm_lock: Optional callback function for lock confirmation
                          Should return True to lock, False to skip
                          Signature: (tech_file: Path) -> bool
                          
    Raises:
        RuntimeError: If container is not locked and user declines locking
    """
    from . import container_manager
    import logging
    logger = logging.getLogger(__name__)
    
    technical_file = Path(technical_file)
    
    if not technical_file.exists():
        raise FileNotFoundError(f"Technical container not found: {technical_file}")
    
    # Check if technical container is locked
    is_locked = container_manager.is_container_locked(technical_file)
    
    if not is_locked:
        logger.warning(
            f"Technical container is not locked: {technical_file}\n"
            f"Locked containers can be reused by multiple session containers."
        )
        
        should_lock = False
        
        if auto_lock:
            should_lock = True
            logger.info("Auto-locking enabled - locking container")
        elif user_confirm_lock is not None:
            # Ask user via callback
            should_lock = user_confirm_lock(technical_file)
        
        if should_lock:
            logger.info(f"Locking technical container: {technical_file}")
            container_manager.lock_container(technical_file)
        else:
            logger.warning(
                f"Proceeding with unlocked technical container.\n"
                f"This container may be modified, which could affect this session."
            )
    
    # Copy technical data
    utils.copy_group(
        src_file=technical_file,
        src_group=schema.GROUP_TECHNICAL,
        dst_file=session_file,
        dst_group=schema.GROUP_TECHNICAL,
    )

    # Track technical source used by this session snapshot.
    utils.set_attrs(
        file_path=session_file,
        path=schema.GROUP_TECHNICAL,
        attrs={
            "source_file": str(technical_file),
            "copied_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def add_detector_data_with_blobs(
    file_path: Union[str, Path],
    detector_path: str,
    processed_signal: np.ndarray,
    raw_files: Dict[str, bytes],
    poni_ref_path: Optional[str] = None,
) -> None:
    """Add detector data with raw file blobs and mandatory processed signal.
    
    Args:
        file_path: Container path
        detector_path: Full path to detector group (e.g., "/measurements/pt_001/meas_NNN/det_saxs")
        processed_signal: Mandatory processed numpy array (compression=4)
        raw_files: Mandatory dict of {"raw_txt": bytes, "raw_dsc": bytes, ...} for raw data blobs
                   Keys should use format: raw_<extension> (e.g., raw_txt, raw_dsc, raw_t3pa)
                   This matches technical container blob naming convention.
        poni_ref_path: Optional path to PONI file for reference
        
    Notes:
        - processed_signal is MANDATORY (stored at detector_path/processed_signal)
        - raw_files is MANDATORY (stored as blobs under detector_path/blob with max compression 9)
        - Blob naming convention: raw_txt, raw_dsc, etc. matches technical containers
    """
    # Create detector group
    utils.create_group_if_missing(file_path, detector_path)
    
    # 1. Write mandatory processed_signal (compression=4)
    processed_path = f"{detector_path}/{schema.DATASET_PROCESSED_SIGNAL}"
    utils.write_dataset(
        file_path=file_path,
        dataset_path=processed_path,
        data=processed_signal,
        compression="gzip",
        compression_opts=schema.COMPRESSION_PROCESSED,
        overwrite=True,
    )
    
    # 2. Write raw file blobs (compression=9)
    # Blobs are stored with keys: raw_txt, raw_dsc, etc.
    # This matches the naming convention used in technical containers
    blob_group = f"{detector_path}/{schema.DATASET_BLOB}"
    utils.create_group_if_missing(file_path, blob_group)
    
    for blob_key, content in raw_files.items():
        # blob_key should already be in format: raw_txt, raw_dsc, etc.
        # If caller provides full filenames, extract format
        if not blob_key.startswith('raw_'):
            # Extract extension from filename (e.g., file.txt -> txt)
            import os
            _, ext = os.path.splitext(blob_key)
            file_format = ext[1:] if ext else "unknown"
            blob_key = f"raw_{file_format}"
        
        blob_path = f"{blob_group}/{blob_key}"
        
        # Convert bytes to numpy array for HDF5 storage
        if isinstance(content, bytes):
            blob_data = np.frombuffer(content, dtype=np.uint8)
        elif isinstance(content, np.ndarray):
            # If already numpy, store as-is
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
    
    # 3. Add PONI reference if provided
    if poni_ref_path:
        try:
            utils.set_reference_attr(
                file_path=file_path,
                obj_path=detector_path,
                attr_name=schema.ATTR_PONI_REF,
                target_path=poni_ref_path,
            )
        except KeyError:
            # PONI may not exist
            pass


def add_image(
    file_path: Union[str, Path],
    image_index: int,
    image_data: Union[np.ndarray, str],
    image_type: str = schema.IMAGE_TYPE_SAMPLE,
    timestamp: Optional[str] = None,
) -> str:
    """Add a sample image to /images.

    Args:
        file_path: Session container path
        image_index: Image index (1-based)
        image_data: Image array (2D grayscale or 3D RGB) or path to image file
        image_type: Type of image (e.g., "sample", "reference")
        timestamp: Optional timestamp (generated if not provided)

    Returns:
        Image group path
        
    Notes:
        - Images stored as-is (no grayscale conversion)
        - Supports grayscale (H×W) and RGB (H×W×3) images
    """
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    image_id = schema.format_image_id(image_index)
    image_path = f"{schema.GROUP_IMAGES}/{image_id}"

    # Create image group
    utils.create_group_if_missing(file_path, image_path)

    # Write image data (preserve as-is, no grayscale conversion)
    if isinstance(image_data, str):
        # If it's a filename, read it
        image_data = np.load(image_data)

    utils.write_dataset(
        file_path=file_path,
        dataset_path=f"{image_path}/data",
        data=image_data,
        compression="gzip",
        compression_opts=schema.COMPRESSION_IMAGE,  # Use schema constant
    )

    # Set attributes
    attrs = {
        schema.ATTR_IMAGE_TYPE: image_type,
        schema.ATTR_TIMESTAMP: timestamp,
    }
    utils.set_attrs(file_path, image_path, attrs)

    return image_path


def add_zone(
    file_path: Union[str, Path],
    zone_index: int,
    zone_role: str,
    geometry_px: Union[List, np.ndarray, str],
    shape: str = "polygon",
    holder_diameter_mm: Optional[float] = None,
) -> str:
    """Add a zone definition to /images/zones.

    Args:
        file_path: Session container path
        zone_index: Zone index (1-based)
        zone_role: Role of zone (sample_holder, include, exclude)
        geometry_px: Pixel coordinates defining zone (array or JSON string)
        shape: Shape type (polygon, circle, etc.)
        holder_diameter_mm: Diameter in mm (only for sample_holder zones)

    Returns:
        Zone group path
    """
    if not schema.validate_zone_role(zone_role):
        raise ValueError(f"Invalid zone role: {zone_role}")

    zone_id = schema.format_zone_id(zone_index)
    zone_path = f"{schema.GROUP_IMAGES_ZONES}/{zone_id}"

    # Create zone group
    utils.create_group_if_missing(file_path, zone_path)

    # Write geometry
    if isinstance(geometry_px, str):
        # Assume it's JSON
        utils.write_dataset(
            file_path=file_path,
            dataset_path=f"{zone_path}/geometry_px",
            data=geometry_px,
            compression=None,
        )
    else:
        # NumPy array or list
        utils.write_dataset(
            file_path=file_path,
            dataset_path=f"{zone_path}/geometry_px",
            data=np.array(geometry_px),
            compression="gzip",
        )

    # Set attributes
    attrs = {
        schema.ATTR_ZONE_ROLE: zone_role,
        schema.ATTR_ZONE_SHAPE: shape,
    }
    if holder_diameter_mm is not None:
        attrs[schema.ATTR_HOLDER_DIAMETER_MM] = holder_diameter_mm

    utils.set_attrs(file_path, zone_path, attrs)

    return zone_path


def add_image_mapping(
    file_path: Union[str, Path],
    sample_holder_zone_id: str,
    pixel_to_mm_conversion: Dict,
    orientation: str = "standard",
    mapping_version: str = "0.1",
) -> str:
    """Add pixel-to-mm mapping metadata to /images/mapping.

    Args:
        file_path: Session container path
        sample_holder_zone_id: ID of sample_holder zone
        pixel_to_mm_conversion: Dict with conversion factors
        orientation: Orientation label
        mapping_version: Mapping version

    Returns:
        Mapping dataset path
    """
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

    return mapping_path


def add_point(
    file_path: Union[str, Path],
    point_index: int,
    pixel_coordinates: List[float],
    physical_coordinates_mm: List[float],
    point_status: str = schema.POINT_STATUS_PENDING,
) -> str:
    """Add a measurement point to /points.

    Args:
        file_path: Session container path
        point_index: Point index (1-based)
        pixel_coordinates: [x_px, y_px]
        physical_coordinates_mm: [x_mm, y_mm]
        point_status: Status of point (pending, measured, skipped)

    Returns:
        Point group path
    """
    point_id = schema.format_point_id(point_index)
    point_path = f"{schema.GROUP_POINTS}/{point_id}"

    # Create point group
    utils.create_group_if_missing(file_path, point_path)

    # Set attributes
    attrs = {
        schema.ATTR_PIXEL_COORDINATES: np.array(pixel_coordinates),
        schema.ATTR_PHYSICAL_COORDINATES_MM: np.array(physical_coordinates_mm),
        schema.ATTR_POINT_STATUS: point_status,
    }
    utils.set_attrs(file_path, point_path, attrs)

    return point_path


def update_point_status(
    file_path: Union[str, Path],
    point_index: int,
    point_status: str,
) -> None:
    """Update the status of a point.

    Args:
        file_path: Session container path
        point_index: Point index (1-based)
        point_status: New status value
    """
    point_id = schema.format_point_id(point_index)
    point_path = f"{schema.GROUP_POINTS}/{point_id}"

    utils.set_attrs(
        file_path=file_path,
        path=point_path,
        attrs={schema.ATTR_POINT_STATUS: point_status},
    )


def get_next_measurement_counter(file_path: Union[str, Path]) -> int:
    """Get the next measurement counter value and increment it.

    Args:
        file_path: Session container path

    Returns:
        Next measurement counter value
    """
    with utils.open_h5_append(file_path) as f:
        counter = int(f.attrs.get("measurement_counter", 0))
        f.attrs["measurement_counter"] = counter + 1
    return counter + 1


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
    """Add a measurement to /measurements/pt_###/meas_#########.

    Args:
        file_path: Session container path
        point_index: Point index (1-based)
        measurement_data: Dict mapping detector_id to processed numpy array (MANDATORY)
        detector_metadata: Dict mapping detector_id to metadata dict with keys:
                          - 'integration_time_ms': float
                          - 'beam_energy_keV': float (optional)
        poni_alias_map: Dict mapping detector_alias to detector_id
        raw_files: Dict mapping detector_id to {"raw_txt": bytes, "raw_dsc": bytes} (MANDATORY)
        timestamp_start: Start timestamp (generated if not provided)
        timestamp_end: End timestamp (optional)
        measurement_status: Status of measurement (completed, failed, aborted)

    Returns:
        Measurement group path
        
    Notes:
        - measurement_data contains mandatory processed signals (compression=4)
        - raw_files contains mandatory raw detector file blobs (compression=9)
    """
    if timestamp_start is None:
        timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S")

    # Get next measurement counter
    meas_counter = get_next_measurement_counter(file_path)

    point_id = schema.format_point_id(point_index)
    meas_id = schema.format_measurement_id(meas_counter)
    meas_path = f"{schema.GROUP_MEASUREMENTS}/{point_id}/{meas_id}"

    # Create measurement group
    utils.create_group_if_missing(file_path, meas_path)

    # Set measurement-level attributes
    attrs = {
        schema.ATTR_MEASUREMENT_COUNTER: meas_counter,
        schema.ATTR_TIMESTAMP_START: timestamp_start,
        schema.ATTR_MEASUREMENT_STATUS: measurement_status,
    }
    if timestamp_end is not None:
        attrs[schema.ATTR_TIMESTAMP_END] = timestamp_end

    # Add point reference
    point_path = f"{schema.GROUP_POINTS}/{point_id}"
    try:
        utils.set_reference_attr(
            file_path=file_path,
            obj_path=meas_path,
            attr_name=schema.ATTR_POINT_REF,
            target_path=point_path,
        )
    except KeyError:
        # Point may not exist yet, continue without reference
        pass

    utils.set_attrs(file_path, meas_path, attrs)

    # Write per-detector data
    raw_files = raw_files or {}

    for detector_id, processed_signal in measurement_data.items():
        # Determine detector role from alias
        alias = None
        for a, d in poni_alias_map.items():
            if d == detector_id:
                alias = a
                break
        if alias is None:
            alias = detector_id

        role = schema.format_detector_role(alias)
        detector_path = f"{meas_path}/{role}"
        
        # Get metadata for this detector
        metadata = detector_metadata.get(detector_id) if detector_metadata else None
        
        # Get raw files for this detector (mandatory)
        detector_raw_files = raw_files.get(detector_id, {})
        
        # Set detector-level attributes
        det_attrs = {
            schema.ATTR_DETECTOR_ID: detector_id,  # Real hardware ID
            schema.ATTR_DETECTOR_ALIAS: alias,  # Alias (e.g., "PRIMARY")
        }
        if metadata:
            det_attrs[schema.ATTR_INTEGRATION_TIME_MS] = metadata.get(
                "integration_time_ms", 0
            )
            if "beam_energy_keV" in metadata:
                det_attrs[schema.ATTR_BEAM_ENERGY_KEV] = metadata["beam_energy_keV"]
        
        # Write detector data with mandatory blobs
        poni_path = f"{schema.GROUP_TECHNICAL_PONI}/poni_{role[4:]}"
        add_detector_data_with_blobs(
            file_path=file_path,
            detector_path=detector_path,
            processed_signal=processed_signal,
            raw_files=detector_raw_files,
            poni_ref_path=poni_path,
        )
        
        # Set detector attributes
        utils.set_attrs(file_path, detector_path, det_attrs)

    return meas_path


def add_analytical_measurement(
    file_path: Union[str, Path],
    measurement_data: Dict[str, np.ndarray],
    detector_metadata: Dict[str, Dict],
    poni_alias_map: Dict[str, str],
    analysis_type: str,
    raw_files: Optional[Dict[str, Dict[str, bytes]]] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    measurement_status: str = schema.STATUS_COMPLETED,
) -> str:
    """Add an analytical measurement to /analytical_measurements.

    Args:
        file_path: Session container path
        measurement_data: Dict mapping detector_id to processed numpy array (MANDATORY)
        detector_metadata: Dict mapping detector_id to metadata dict
        poni_alias_map: Dict mapping detector_alias to detector_id
        analysis_type: Type of analysis (e.g., "attenuation")
        raw_files: Dict mapping detector_id to {"raw_txt": bytes, "raw_dsc": bytes} (MANDATORY)
        timestamp_start: Start timestamp (generated if not provided)
        timestamp_end: End timestamp (optional)
        measurement_status: Status of measurement

    Returns:
        Analytical measurement group path
    """
    if timestamp_start is None:
        timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S")

    # Get next measurement counter
    meas_counter = get_next_measurement_counter(file_path)

    ana_id = schema.format_analytical_measurement_id(meas_counter)
    ana_path = f"{schema.GROUP_ANALYTICAL_MEASUREMENTS}/{ana_id}"

    # Create analytical measurement group
    utils.create_group_if_missing(file_path, ana_path)

    # Set measurement-level attributes
    attrs = {
        schema.ATTR_MEASUREMENT_COUNTER: meas_counter,
        schema.ATTR_TIMESTAMP_START: timestamp_start,
        schema.ATTR_MEASUREMENT_STATUS: measurement_status,
        schema.ATTR_ANALYSIS_TYPE: analysis_type,
    }
    if timestamp_end is not None:
        attrs[schema.ATTR_TIMESTAMP_END] = timestamp_end

    utils.set_attrs(file_path, ana_path, attrs)

    # Write per-detector data
    raw_files = raw_files or {}

    for detector_id, processed_signal in measurement_data.items():
        # Determine detector role
        alias = None
        for a, d in poni_alias_map.items():
            if d == detector_id:
                alias = a
                break
        if alias is None:
            alias = detector_id

        role = schema.format_detector_role(alias)
        detector_path = f"{ana_path}/{role}"
        
        # Get metadata for this detector
        metadata = detector_metadata.get(detector_id) if detector_metadata else None
        
        # Get raw files for this detector (mandatory)
        detector_raw_files = raw_files.get(detector_id, {})
        
        # Set detector-level attributes
        det_attrs = {
            schema.ATTR_DETECTOR_ID: detector_id,  # Real hardware ID
            schema.ATTR_DETECTOR_ALIAS: alias,  # Alias (e.g., "PRIMARY")
        }
        if metadata:
            det_attrs[schema.ATTR_INTEGRATION_TIME_MS] = metadata.get(
                "integration_time_ms", 0
            )
            if "beam_energy_keV" in metadata:
                det_attrs[schema.ATTR_BEAM_ENERGY_KEV] = metadata["beam_energy_keV"]
        
        # Write detector data with mandatory blobs
        poni_path = f"{schema.GROUP_TECHNICAL_PONI}/poni_{role[4:]}"
        add_detector_data_with_blobs(
            file_path=file_path,
            detector_path=detector_path,
            processed_signal=processed_signal,
            raw_files=detector_raw_files,
            poni_ref_path=poni_path,
        )
        
        # Set detector attributes
        utils.set_attrs(file_path, detector_path, det_attrs)

    return ana_path


def link_analytical_measurement_to_point(
    file_path: Union[str, Path],
    point_index: int,
    analytical_measurement_index: int,
) -> None:
    """Link an analytical measurement to a point via bidirectional references.
    
    Creates two references:
    - Point → Analytical measurement (in point's analytical_measurement_refs)
    - Analytical measurement → Point (in analytical measurement's point_refs)

    Args:
        file_path: Session container path
        point_index: Point index (1-based)
        analytical_measurement_index: Analytical measurement counter
    """
    point_id = schema.format_point_id(point_index)
    point_path = f"{schema.GROUP_POINTS}/{point_id}"

    ana_id = schema.format_analytical_measurement_id(analytical_measurement_index)
    ana_path = f"{schema.GROUP_ANALYTICAL_MEASUREMENTS}/{ana_id}"

    # Reference 1: Point → Analytical measurement
    utils.append_reference_to_list_attr(
        file_path=file_path,
        obj_path=point_path,
        attr_name=schema.ATTR_ANALYTICAL_MEASUREMENT_REFS,
        target_path=ana_path,
    )
    
    # Reference 2: Analytical measurement → Point (bidirectional)
    utils.append_reference_to_list_attr(
        file_path=file_path,
        obj_path=ana_path,
        attr_name=schema.ATTR_POINT_REFS,
        target_path=point_path,
    )


def find_active_session_container(
    folder: Union[str, Path], sample_id: Optional[str] = None
) -> Optional[str]:
    """Find the most recent session container in a folder.

    Args:
        folder: Directory to search
        sample_id: Optional sample_id filter

    Returns:
        Path to most recent session container, or None if not found
    """
    folder = Path(folder)
    if not folder.exists():
        return None

    # Find all session_*.h5 files
    pattern = "session_*.h5"
    candidates = list(folder.glob(pattern))

    if not candidates:
        return None

    # Filter by sample_id if specified
    if sample_id is not None:
        filtered = []
        for candidate in candidates:
            try:
                with utils.open_h5_append(candidate) as f:
                    if f.attrs.get(schema.ATTR_SAMPLE_ID) == sample_id:
                        filtered.append(candidate)
            except Exception:
                continue
        candidates = filtered

    if not candidates:
        return None

    # Return most recent by modification time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])
