"""DIFRA NeXus/HDF5 Data Model v0.2 - Schema Constants and Utilities."""

import re
import time
import uuid
from typing import Literal

# ================== Version and Model ======================
SCHEMA_VERSION = "0.2"
DATA_MODEL_VERSION = "0.2"

# ================== Container Types ========================
CONTAINER_TYPE_TECHNICAL = "technical"
CONTAINER_TYPE_SESSION = "session"

# ================== NeXus App Definitions ==================
APPDEF_TECHNICAL = "NXdifra_technical"
APPDEF_SESSION = "NXdifra_session"

# ================== NX Classes / Attributes ================
ATTR_NX_CLASS = "NX_class"
NX_CLASS_ROOT = "NXroot"
NX_CLASS_ENTRY = "NXentry"
NX_CLASS_INSTRUMENT = "NXinstrument"
NX_CLASS_DETECTOR = "NXdetector"
NX_CLASS_COLLECTION = "NXcollection"
NX_CLASS_PROCESS = "NXprocess"
NX_CLASS_NOTE = "NXnote"
NX_CLASS_SAMPLE = "NXsample"
NX_CLASS_USER = "NXuser"
NX_CLASS_DATA = "NXdata"

# ================== Top-Level Group Names ==================
GROUP_ENTRY = "/entry"

# v0.2 canonical layout lives under /entry to support multiple entries in future.
GROUP_TECHNICAL = f"{GROUP_ENTRY}/technical"
GROUP_TECHNICAL_CONFIG = f"{GROUP_TECHNICAL}/config"
GROUP_TECHNICAL_PONI = f"{GROUP_TECHNICAL}/poni"
GROUP_INSTRUMENT_DETECTORS = f"{GROUP_TECHNICAL_CONFIG}/detectors"

GROUP_IMAGES = f"{GROUP_ENTRY}/images"
GROUP_IMAGES_ZONES = f"{GROUP_IMAGES}/zones"
GROUP_IMAGES_MAPPING = f"{GROUP_IMAGES}/mapping"
GROUP_POINTS = f"{GROUP_ENTRY}/points"
GROUP_MEASUREMENTS = f"{GROUP_ENTRY}/measurements"
GROUP_ANALYTICAL_MEASUREMENTS = f"{GROUP_ENTRY}/analytical_measurements"
GROUP_RUNTIME = f"{GROUP_ENTRY}/difra_runtime"

GROUP_SAMPLE = f"{GROUP_ENTRY}/sample"
GROUP_USER = f"{GROUP_ENTRY}/user"
GROUP_INSTRUMENT = f"{GROUP_ENTRY}/instrument"

# NeXus-oriented aliases retained for readability in validators/writers.
GROUP_CALIBRATION = GROUP_TECHNICAL
GROUP_CALIBRATION_EVENTS = GROUP_TECHNICAL
GROUP_PONI = GROUP_TECHNICAL_PONI
GROUP_CALIBRATION_SNAPSHOT = GROUP_TECHNICAL

# ================== ID Formatting ==========================
def format_point_id(index: int) -> str:
    return f"pt_{index:03d}"


def format_measurement_id(counter: int) -> str:
    return f"meas_{counter:09d}"


def format_analytical_measurement_id(counter: int) -> str:
    return f"ana_{counter:09d}"


def format_technical_event_id(index: int) -> str:
    return f"tech_evt_{index:06d}"


def format_image_id(index: int) -> str:
    return f"img_{index:03d}"


def format_zone_id(index: int) -> str:
    return f"zone_{index:03d}"


def format_detector_role(alias: str) -> str:
    if alias.upper() == "PRIMARY":
        return "det_primary"
    if alias.upper() == "SECONDARY":
        return "det_secondary"
    return f"det_{alias.lower()}"


def parse_detector_role(role: str) -> str:
    if not role.startswith("det_"):
        raise ValueError(f"Invalid detector role: {role}")

    suffix = role[4:]
    if suffix == "primary":
        return "PRIMARY"
    if suffix == "secondary":
        return "SECONDARY"
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
ATTR_DISTANCE_CM = "distance_cm"
ATTR_DETECTOR_DISTANCE_CM = "detector_distance_cm"

# Session container specific (required)
ATTR_SAMPLE_ID = "sample_id"
ATTR_STUDY_NAME = "study_name"
ATTR_PROJECT_ID = "project_id"
ATTR_SESSION_ID = "session_id"
ATTR_ACQUISITION_DATE = "acquisition_date"
ATTR_OPERATOR_ID = "operator_id"
ATTR_SITE_ID = "site_id"
ATTR_MACHINE_NAME = "machine_name"
ATTR_BEAM_ENERGY_KEV = "beam_energy_keV"

# Session container specific (optional)
ATTR_PATIENT_ID = "patient_id"
ATTR_HUMAN_SUMMARY = "human_summary"
ATTR_PRODUCER_SOFTWARE = "producer_software"
ATTR_PRODUCER_VERSION = "producer_version"

# ================== NeXus Entry Attributes ================
ATTR_ENTRY_DEFAULT = "default"
ATTR_ENTRY_DEFINITION = "definition"
ATTR_START_TIME = "start_time"

# ================== Technical Measurement Attrs ============
ATTR_TECHNICAL_TYPE = "technical_type"
ATTR_TIMESTAMP = "timestamp"
ATTR_DETECTOR_ID = "detector_id"
ATTR_DETECTOR_ALIAS = "detector_alias"

# ================== PONI Attributes ========================
ATTR_PONI_DERIVED_FROM = "derived_from"
ATTR_PONI_OPERATOR_CONFIRMED = "operator_confirmed"

# ================== Measurement Attributes =================
ATTR_MEASUREMENT_COUNTER = "measurement_counter"
ATTR_TIMESTAMP_START = "timestamp_start"
ATTR_TIMESTAMP_END = "timestamp_end"
ATTR_MEASUREMENT_STATUS = "measurement_status"
ATTR_FAILURE_REASON = "failure_reason"
ATTR_POINT_REF = "point_ref"
ATTR_PONI_REF = "poni_ref"

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_ABORTED = "aborted"

# ================== Analytical Measurement Attrs ===========
ATTR_ANALYSIS_TYPE = "analysis_type"
ATTR_ANALYSIS_ROLE = "analysis_role"
ATTR_POINT_REFS = "point_refs"
ATTR_POINT_IDS = "point_ids"
ATTR_ANALYTICAL_MEASUREMENT_IDS = "analytical_measurement_ids"
ANALYSIS_TYPE_ATTENUATION = "attenuation"
ANALYSIS_ROLE_UNSPECIFIED = "unspecified"
ANALYSIS_ROLE_I0 = "i0"
ANALYSIS_ROLE_I = "i"

# ================== Detector-Level Attributes ==============
ATTR_INTEGRATION_TIME_MS = "integration_time_ms"
ATTR_N_FRAMES = "n_frames"

# ================== Point Attributes =======================
ATTR_PIXEL_COORDINATES = "pixel_coordinates"
ATTR_PHYSICAL_COORDINATES_MM = "physical_coordinates_mm"
ATTR_POINT_STATUS = "point_status"
ATTR_SKIP_REASON = "skip_reason"
ATTR_ANALYTICAL_MEASUREMENT_REFS = "analytical_measurement_refs"
ATTR_THICKNESS = "thickness"

POINT_STATUS_PENDING = "pending"
POINT_STATUS_MEASURED = "measured"
POINT_STATUS_SKIPPED = "skipped"
THICKNESS_UNKNOWN = "unknown"

# ================== Image Attributes =======================
ATTR_IMAGE_TYPE = "image_type"
IMAGE_TYPE_SAMPLE = "sample"
IMAGE_TYPE_REFERENCE = "reference"

# ================== Zone Attributes ========================
ATTR_ZONE_ROLE = "zone_role"
ATTR_ZONE_SHAPE = "shape"
ATTR_GEOMETRY_PX = "geometry_px"
ATTR_HOLDER_DIAMETER_MM = "holder_diameter_mm"

# ================== Runtime Attributes =====================
ATTR_LOCKED = "locked"
ATTR_LOCKED_BY = "locked_by"
ATTR_LOCKED_TIMESTAMP = "lock_timestamp"

# ================== Dataset Names ==========================
DATASET_RAW_SIGNAL = "raw_signal"
DATASET_PROCESSED_SIGNAL = "processed_signal"
DATASET_BLOB = "blob"
DATASET_METADATA = "metadata"
DATASET_SESSION_LOG = "session_log"

# ================== Compression Levels =====================
COMPRESSION_BLOB_MAX = 9
COMPRESSION_PROCESSED = 4
COMPRESSION_IMAGE = 4


# ================== Helper Functions =======================
def now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def today_token() -> str:
    return time.strftime("%Y%m%d")


def generate_container_id() -> str:
    return uuid.uuid4().hex[:16]


def validate_container_id(container_id: str) -> bool:
    return bool(re.match(r"^[0-9a-f]{16}$", container_id))


def parse_poni_distance(poni_content: str) -> float:
    for line in poni_content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Distance:"):
            distance_str = stripped.split(":", 1)[1].strip()
            try:
                return float(distance_str)
            except ValueError as error:
                raise ValueError(
                    f"Cannot parse distance value '{distance_str}' from PONI: {error}"
                ) from error

    raise ValueError(
        "Distance field not found in PONI content. Expected line starting with 'Distance:'"
    )


def validate_poni_distance(
    poni_content: str,
    user_distance_cm: float,
    tolerance_percent: float = 5.0,
) -> None:
    poni_distance_m = parse_poni_distance(poni_content)
    poni_distance_cm = poni_distance_m * 100.0

    deviation_cm = abs(poni_distance_cm - user_distance_cm)
    deviation_percent = (deviation_cm / user_distance_cm) * 100.0

    if deviation_percent > tolerance_percent:
        raise ValueError(
            f"PONI distance validation failed:\n"
            f"  PONI file: {poni_distance_cm:.2f} cm\n"
            f"  User specified: {user_distance_cm:.2f} cm\n"
            f"  Deviation: {deviation_percent:.1f}% (max allowed: {tolerance_percent:.1f}%)"
        )


def format_technical_container_filename(
    container_id: str,
    distance_cm: float = None,
    date_token: str = None,
) -> str:
    """Format technical filename: technical_<id>_<distance>cm_<date>.nxs.h5."""
    if not validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")

    date_part = date_token or today_token()
    if distance_cm is None:
        return f"technical_{container_id}_{date_part}.nxs.h5"

    distance_token = f"{float(distance_cm):.2f}".replace(".", "p").replace("-", "m")
    return f"technical_{container_id}_{distance_token}cm_{date_part}.nxs.h5"


def format_session_container_filename(
    container_id: str,
    sample_id: str = None,
    date_token: str = None,
) -> str:
    """Format session filename: session_<id>_<sample>_<date>.nxs.h5."""
    if not validate_container_id(container_id):
        raise ValueError(f"Invalid container ID: {container_id}")

    date_part = date_token or today_token()
    if sample_id:
        safe_sample_id = re.sub(r"[^a-zA-Z0-9_-]", "_", sample_id)
        return f"session_{container_id}_{safe_sample_id}_{date_part}.nxs.h5"
    return f"session_{container_id}_{date_part}.nxs.h5"


def validate_technical_type(tech_type: str) -> bool:
    return tech_type in ALL_TECHNICAL_TYPES


def validate_zone_role(role: str) -> bool:
    return role in [ZONE_ROLE_SAMPLE_HOLDER, ZONE_ROLE_INCLUDE, ZONE_ROLE_EXCLUDE]


def validate_detector_data(detector_group) -> tuple:
    if DATASET_PROCESSED_SIGNAL not in detector_group:
        return False, f"Missing mandatory {DATASET_PROCESSED_SIGNAL} dataset"
    return True, None


ContainerType = Literal["technical", "session"]
