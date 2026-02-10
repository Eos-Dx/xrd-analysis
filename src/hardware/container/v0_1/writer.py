"""DIFRA Session Container Writer

Creates and updates session_<id>.h5 containers that store:
- Complete sample acquisition session data (sample_id, operator, beam energy, etc.)
- Copy of /technical group from technical container (PONY, detector config, technical measurements)
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
    container_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Create a new empty session container with required root attributes.

    Args:
        folder: Directory where container will be created
        sample_id: Unique sample identifier
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
    technical_file: Union[str, Path], session_file: Union[str, Path]
) -> None:
    """Copy /technical group from technical container to session container.

    Args:
        technical_file: Path to technical container
        session_file: Path to session container
    """
    utils.copy_group(
        src_file=technical_file,
        src_group=schema.GROUP_TECHNICAL,
        dst_file=session_file,
        dst_group=schema.GROUP_TECHNICAL,
    )


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
        image_data: 2D image array or path to image file
        image_type: Type of image (e.g., "sample", "reference")
        timestamp: Optional timestamp (generated if not provided)

    Returns:
        Image group path
    """
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    image_id = schema.format_image_id(image_index)
    image_path = f"{schema.GROUP_IMAGES}/{image_id}"

    # Create image group
    utils.create_group_if_missing(file_path, image_path)

    # Write image data
    if isinstance(image_data, str):
        # If it's a filename, read it
        image_data = np.load(image_data)

    utils.write_dataset(
        file_path=file_path,
        dataset_path=f"{image_path}/data",
        data=image_data,
        compression="gzip",
        compression_opts=4,  # Medium compression for image data
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
    mapping_version: str = "1.0",
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
    pony_alias_map: Dict[str, str],
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    measurement_status: str = schema.STATUS_COMPLETED,
) -> str:
    """Add a measurement to /measurements/pt_###/meas_#########.

    Args:
        file_path: Session container path
        point_index: Point index (1-based)
        measurement_data: Dict mapping detector_id to numpy array (2D detector image)
        detector_metadata: Dict mapping detector_id to metadata dict with keys:
                          - 'integration_time_ms': float
                          - 'beam_energy_keV': float (optional)
                          - 'detector_id': str
        pony_alias_map: Dict mapping detector_alias to detector_id
        timestamp_start: Start timestamp (generated if not provided)
        timestamp_end: End timestamp (optional)
        measurement_status: Status of measurement (completed, failed, aborted)

    Returns:
        Measurement group path
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
    for detector_id, raw_signal in measurement_data.items():
        # Determine detector role from alias
        alias = None
        for a, d in pony_alias_map.items():
            if d == detector_id:
                alias = a
                break
        if alias is None:
            alias = detector_id

        role = schema.format_detector_role(alias)
        detector_path = f"{meas_path}/{role}"

        # Create detector group
        utils.create_group_if_missing(file_path, detector_path)

        # Write raw_signal dataset
        raw_signal_path = f"{detector_path}/{schema.DATASET_RAW_SIGNAL}"
        utils.write_dataset(
            file_path=file_path,
            dataset_path=raw_signal_path,
            data=raw_signal,
            compression="gzip",
            compression_opts=4,  # Medium compression for measurement data
            overwrite=True,
        )

        # Set detector-level attributes
        det_attrs = {
            schema.ATTR_DETECTOR_ID: detector_id,
        }

        if detector_id in detector_metadata:
            meta = detector_metadata[detector_id]
            det_attrs[schema.ATTR_INTEGRATION_TIME_MS] = meta.get(
                "integration_time_ms", 0
            )
            if "beam_energy_keV" in meta:
                det_attrs[schema.ATTR_BEAM_ENERGY_KEV] = meta["beam_energy_keV"]

        utils.set_attrs(file_path, detector_path, det_attrs)

        # Add PONY reference if available
        try:
            pony_path = f"{schema.GROUP_TECHNICAL_PONY}/pony_{role[4:]}"
            utils.set_reference_attr(
                file_path=file_path,
                obj_path=detector_path,
                attr_name=schema.ATTR_PONY_REF,
                target_path=pony_path,
            )
        except KeyError:
            # PONY may not exist for this detector
            pass

    return meas_path


def add_analytical_measurement(
    file_path: Union[str, Path],
    measurement_data: Dict[str, np.ndarray],
    detector_metadata: Dict[str, Dict],
    pony_alias_map: Dict[str, str],
    analysis_type: str,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    measurement_status: str = schema.STATUS_COMPLETED,
) -> str:
    """Add an analytical measurement to /analytical_measurements.

    Args:
        file_path: Session container path
        measurement_data: Dict mapping detector_id to numpy array (2D detector image)
        detector_metadata: Dict mapping detector_id to metadata dict
        pony_alias_map: Dict mapping detector_alias to detector_id
        analysis_type: Type of analysis (e.g., "attenuation")
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
    for detector_id, raw_signal in measurement_data.items():
        # Determine detector role
        alias = None
        for a, d in pony_alias_map.items():
            if d == detector_id:
                alias = a
                break
        if alias is None:
            alias = detector_id

        role = schema.format_detector_role(alias)
        detector_path = f"{ana_path}/{role}"

        # Create detector group
        utils.create_group_if_missing(file_path, detector_path)

        # Write raw_signal dataset
        raw_signal_path = f"{detector_path}/{schema.DATASET_RAW_SIGNAL}"
        utils.write_dataset(
            file_path=file_path,
            dataset_path=raw_signal_path,
            data=raw_signal,
            compression="gzip",
            compression_opts=4,
            overwrite=True,
        )

        # Set detector-level attributes
        det_attrs = {
            schema.ATTR_DETECTOR_ID: detector_id,
        }

        if detector_id in detector_metadata:
            meta = detector_metadata[detector_id]
            det_attrs[schema.ATTR_INTEGRATION_TIME_MS] = meta.get(
                "integration_time_ms", 0
            )
            if "beam_energy_keV" in meta:
                det_attrs[schema.ATTR_BEAM_ENERGY_KEV] = meta["beam_energy_keV"]

        utils.set_attrs(file_path, detector_path, det_attrs)

        # Add PONY reference if available
        try:
            pony_path = f"{schema.GROUP_TECHNICAL_PONY}/pony_{role[4:]}"
            utils.set_reference_attr(
                file_path=file_path,
                obj_path=detector_path,
                attr_name=schema.ATTR_PONY_REF,
                target_path=pony_path,
            )
        except KeyError:
            # PONY may not exist for this detector
            pass

    return ana_path


def link_analytical_measurement_to_point(
    file_path: Union[str, Path],
    point_index: int,
    analytical_measurement_index: int,
) -> None:
    """Link an analytical measurement to a point via reference list.

    Args:
        file_path: Session container path
        point_index: Point index (1-based)
        analytical_measurement_index: Analytical measurement counter
    """
    point_id = schema.format_point_id(point_index)
    point_path = f"{schema.GROUP_POINTS}/{point_id}"

    ana_id = schema.format_analytical_measurement_id(analytical_measurement_index)
    ana_path = f"{schema.GROUP_ANALYTICAL_MEASUREMENTS}/{ana_id}"

    utils.append_reference_to_list_attr(
        file_path=file_path,
        obj_path=point_path,
        attr_name=schema.ATTR_ANALYTICAL_MEASUREMENT_REFS,
        target_path=ana_path,
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
