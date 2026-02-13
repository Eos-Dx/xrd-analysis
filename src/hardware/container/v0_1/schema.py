"""DIFRA HDF5 Data Model v0.1 - Schema Constants and Utilities

This module defines constants, formatters, and validation rules matching
the approved DIFRA_HDF5_Data_Model_FINAL.md specification.

Container v0.1 Format Requirements:
- Measurement data MUST be in .npy format (numpy binary)
- Raw detector files (.txt, .dsc for Advacam) stored as blobs
- Detectors MUST convert raw output using convert_to_container_format() before
  passing to container writers

Example detector workflow:
    # 1. Detector captures raw data
    detector.capture_point(frames=10, seconds=2.0, filename_base="dark_001")
    # Creates: dark_001.txt, dark_001.dsc
    
    # 2. Convert to container format
    npy_file = detector.convert_to_container_format("dark_001.txt", "0.1")
    # Creates: dark_001.npy
    
    # 3. Pass .npy to container writer
    technical_container.add_technical_event(
        measurements={"PRIMARY": {"data": np.load(npy_file), "source_file": npy_file}}
    )
    # Container stores .npy data as raw_signal dataset, archives .txt/.dsc as blobs
"""

import re
import uuid
from typing import Literal

# ================== Version and Model ======================
SCHEMA_VERSION = "0.1"
DATA_MODEL_VERSION = "0.1"

# ================== Container Types ========================
CONTAINER_TYPE_TECHNICAL = "technical"
CONTAINER_TYPE_SESSION = "session"

# ================== Top-Level Group Names ==================
GROUP_TECHNICAL = "/technical"
GROUP_IMAGES = "/images"
GROUP_IMAGES_ZONES = "/images/zones"
GROUP_IMAGES_MAPPING = "/images/mapping"
GROUP_POINTS = "/points"
GROUP_MEASUREMENTS = "/measurements"
GROUP_ANALYTICAL_MEASUREMENTS = "/analytical_measurements"

# Sub-groups within /technical
GROUP_TECHNICAL_CONFIG = "/technical/config"
GROUP_TECHNICAL_PONI = "/technical/poni"

# ================== ID Formatting ==========================
def format_point_id(index: int) -> str:
    """Format point ID as pt_###"""
    return f"pt_{index:03d}"

def format_measurement_id(counter: int) -> str:
    """Format measurement ID as meas_#########"""
    return f"meas_{counter:09d}"

def format_analytical_measurement_id(counter: int) -> str:
    """Format analytical measurement ID as ana_#########"""
    return f"ana_{counter:09d}"

def format_technical_event_id(index: int) -> str:
    """Format technical event ID as tech_evt_###"""
    return f"tech_evt_{index:03d}"

def format_image_id(index: int) -> str:
    """Format image ID as img_###"""
    return f"img_{index:03d}"

def format_zone_id(index: int) -> str:
    """Format zone ID as zone_###"""
    return f"zone_{index:03d}"

def format_detector_role(alias: str) -> str:
    """Map detector alias to role name used in HDF5 groups.
    
    Examples:
        PRIMARY -> det_primary
        SECONDARY -> det_secondary
        SAXS -> det_saxs
    """
    if alias.upper() == "PRIMARY":
        return "det_primary"
    elif alias.upper() == "SECONDARY":
        return "det_secondary"
    else:
        return f"det_{alias.lower()}"

def parse_detector_role(role: str) -> str:
    """Parse detector role name back to alias.
    
    Examples:
        det_primary -> PRIMARY
        det_secondary -> SECONDARY
        det_saxs -> SAXS
    """
    if not role.startswith("det_"):
        raise ValueError(f"Invalid detector role: {role}")
    
    suffix = role[4:]  # Remove "det_" prefix
    if suffix == "primary":
        return "PRIMARY"
    elif suffix == "secondary":
        return "SECONDARY"
    else:
        return suffix.upper()

# ================== Guaranteed Technical Types =============
TECHNICAL_TYPE_DARK = "DARK"
TECHNICAL_TYPE_EMPTY = "EMPTY"
TECHNICAL_TYPE_BACKGROUND = "BACKGROUND"
TECHNICAL_TYPE_AGBH = "AGBH"
TECHNICAL_TYPE_WATER = "WATER"

REQUIRED_TECHNICAL_TYPES = [
    TECHNICAL_TYPE_DARK,
    TECHNICAL_TYPE_EMPTY,
    TECHNICAL_TYPE_BACKGROUND,
    TECHNICAL_TYPE_AGBH,
]

ALL_TECHNICAL_TYPES = REQUIRED_TECHNICAL_TYPES + [TECHNICAL_TYPE_WATER]

# ================== Zone Roles ============================
ZONE_ROLE_SAMPLE_HOLDER = "sample_holder"
ZONE_ROLE_INCLUDE = "include"
ZONE_ROLE_EXCLUDE = "exclude"

# ================== Root-Level Attributes ==================
ATTR_CONTAINER_ID = "container_id"
ATTR_CONTAINER_TYPE = "container_type"
ATTR_SCHEMA_VERSION = "schema_version"
ATTR_CREATION_TIMESTAMP = "creation_timestamp"

# Technical container specific
ATTR_DISTANCE_CM = "distance_cm"  # Root-level distance (primary detector or single distance)
ATTR_DETECTOR_DISTANCE_CM = "detector_distance_cm"  # Per-detector distance (stored in detector group attrs)

# Session container specific (required)
ATTR_SAMPLE_ID = "sample_id"
ATTR_STUDY_NAME = "study_name"
ATTR_SESSION_ID = "session_id"
ATTR_ACQUISITION_DATE = "acquisition_date"
ATTR_OPERATOR_ID = "operator_id"
ATTR_SITE_ID = "site_id"
ATTR_MACHINE_NAME = "machine_name"
ATTR_BEAM_ENERGY_KEV = "beam_energy_keV"

# Session container specific (optional)
ATTR_PATIENT_ID = "patient_id"

# ================== Technical Measurement Attrs ============
ATTR_TECHNICAL_TYPE = "technical_type"
ATTR_TIMESTAMP = "timestamp"
ATTR_DETECTOR_ID = "detector_id"  # Real hardware ID (e.g., "advacam_001")
ATTR_DETECTOR_ALIAS = "detector_alias"  # Alias/role (e.g., "PRIMARY", "SECONDARY")

# ================== PONI Attributes ========================
ATTR_PONI_DERIVED_FROM = "derived_from"  # HDF5 ref to technical event
ATTR_PONI_OPERATOR_CONFIRMED = "operator_confirmed"

# ================== Measurement Attributes =================
ATTR_MEASUREMENT_COUNTER = "measurement_counter"
ATTR_TIMESTAMP_START = "timestamp_start"
ATTR_TIMESTAMP_END = "timestamp_end"
ATTR_MEASUREMENT_STATUS = "measurement_status"
ATTR_POINT_REF = "point_ref"
# Canonical reference attribute for PONI links
ATTR_PONI_REF = "poni_ref"

# Measurement status values
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_ABORTED = "aborted"

# ================== Analytical Measurement Attrs ===========
ATTR_ANALYSIS_TYPE = "analysis_type"
ATTR_POINT_REFS = "point_refs"  # HDF5 ref array to points using this analytical measurement

# Analysis types
ANALYSIS_TYPE_ATTENUATION = "attenuation"

# ================== Detector-Level Attributes ==============
ATTR_INTEGRATION_TIME_MS = "integration_time_ms"

# ================== Point Attributes =======================
ATTR_PIXEL_COORDINATES = "pixel_coordinates"  # [x, y]
ATTR_PHYSICAL_COORDINATES_MM = "physical_coordinates_mm"  # [x_mm, y_mm]
ATTR_POINT_STATUS = "point_status"
ATTR_ANALYTICAL_MEASUREMENT_REFS = "analytical_measurement_refs"  # ref array

# Point status values
POINT_STATUS_PENDING = "pending"
POINT_STATUS_MEASURED = "measured"
POINT_STATUS_SKIPPED = "skipped"

# ================== Image Attributes =======================
ATTR_IMAGE_TYPE = "image_type"

# Image types
IMAGE_TYPE_SAMPLE = "sample"
IMAGE_TYPE_REFERENCE = "reference"

# ================== Zone Attributes ========================
ATTR_ZONE_ROLE = "zone_role"
ATTR_ZONE_SHAPE = "shape"
ATTR_GEOMETRY_PX = "geometry_px"  # JSON or array depending on shape
ATTR_HOLDER_DIAMETER_MM = "holder_diameter_mm"

# ================== Dataset Names ==========================
DATASET_RAW_SIGNAL = "raw_signal"  # Alias for processed_signal (backward compat)
DATASET_PROCESSED_SIGNAL = "processed_signal"  # Mandatory numpy array
DATASET_BLOB = "blob"  # Group containing raw file blobs
DATASET_METADATA = "metadata"  # Optional JSON metadata

# ================== Compression Levels =====================
COMPRESSION_BLOB_MAX = 9  # Maximum compression for raw file blobs
COMPRESSION_PROCESSED = 4  # Medium compression for processed numpy arrays
COMPRESSION_IMAGE = 4  # Medium compression for images

# ================== Helper Functions =======================
def generate_container_id() -> str:
    """Generate unique container ID using UUID4 (16-char hex)."""
    return uuid.uuid4().hex[:16]

def validate_container_id(container_id: str) -> bool:
    """Validate container ID format (16 hex chars)."""
    return bool(re.match(r"^[0-9a-f]{16}$", container_id))


def parse_poni_distance(poni_content: str) -> float:
    """Parse distance from PONI file content.
    
    Args:
        poni_content: Raw PONI file text content
        
    Returns:
        Distance in meters
        
    Raises:
        ValueError: If Distance field not found
        
    Example PONI content:
        Distance: 0.17
        PixelSize1: 7.5e-05
        PixelSize2: 7.5e-05
        ...
    """
    for line in poni_content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('Distance:'):
            distance_str = stripped.split(':', 1)[1].strip()
            try:
                return float(distance_str)
            except ValueError as e:
                raise ValueError(
                    f"Cannot parse distance value '{distance_str}' from PONI: {e}"
                )
    
    raise ValueError(
        "Distance field not found in PONI content. "
        "Expected line starting with 'Distance:'"
    )


def validate_poni_distance(
    poni_content: str,
    user_distance_cm: float,
    tolerance_percent: float = 5.0
) -> None:
    """Validate PONI distance matches user-specified distance.
    
    Args:
        poni_content: Raw PONI file text content
        user_distance_cm: User-specified distance in cm
        tolerance_percent: Maximum allowed deviation percentage (default 5%)
        
    Raises:
        ValueError: If deviation exceeds tolerance or distance cannot be parsed
        
    Example:
        >>> poni_content = "Distance: 0.17\nPixelSize1: 7.5e-05"
        >>> validate_poni_distance(poni_content, user_distance_cm=17.0)  # OK
        >>> validate_poni_distance(poni_content, user_distance_cm=20.0)  # Raises ValueError
    """
    # Parse PONI distance (in meters)
    poni_distance_m = parse_poni_distance(poni_content)
    poni_distance_cm = poni_distance_m * 100.0
    
    # Calculate deviation
    deviation_cm = abs(poni_distance_cm - user_distance_cm)
    deviation_percent = (deviation_cm / user_distance_cm) * 100.0
    
    if deviation_percent > tolerance_percent:
        raise ValueError(
            f"PONI distance validation failed:\n"
            f"  PONI file: {poni_distance_cm:.2f} cm\n"
            f"  User specified: {user_distance_cm:.2f} cm\n"
            f"  Deviation: {deviation_percent:.1f}% (max allowed: {tolerance_percent:.1f}%)"
        )

def format_technical_container_filename(container_id: str, distance_cm: float = None) -> str:
    """Format technical container filename: technical_<id>.h5
    
    Optionally include distance in the filename for human readability.
    """
    if not validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")
    
    if distance_cm is not None:
        return f"technical_{container_id}_{distance_cm:.0f}cm.h5"
    else:
        return f"technical_{container_id}.h5"

def format_session_container_filename(container_id: str, sample_id: str = None) -> str:
    """Format session container filename: session_<id>_<sample>.h5.

    Keeps naming aligned with technical containers by avoiding date tokens.
    """
    if not validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")

    if sample_id:
        # Sanitize sample_id for filename
        safe_sample_id = re.sub(r'[^a-zA-Z0-9_-]', '_', sample_id)
        return f"session_{container_id}_{safe_sample_id}.h5"
    else:
        return f"session_{container_id}.h5"

def validate_technical_type(tech_type: str) -> bool:
    """Validate technical type is in allowed set."""
    return tech_type in ALL_TECHNICAL_TYPES

def validate_zone_role(role: str) -> bool:
    """Validate zone role value."""
    return role in [ZONE_ROLE_SAMPLE_HOLDER, ZONE_ROLE_INCLUDE, ZONE_ROLE_EXCLUDE]

def validate_detector_data(detector_group) -> tuple:
    """Validate detector data has mandatory processed_signal.
    
    Args:
        detector_group: HDF5 group object for detector
        
    Returns:
        (is_valid: bool, error_message: str or None)
    """
    if DATASET_PROCESSED_SIGNAL not in detector_group:
        return False, f"Missing mandatory {DATASET_PROCESSED_SIGNAL} dataset"
    return True, None
