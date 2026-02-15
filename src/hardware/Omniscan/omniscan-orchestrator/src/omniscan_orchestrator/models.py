"""
Pydantic Models for Omniscan Orchestrator REST API

Request and response schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# Authentication models
class LoginRequest(BaseModel):
    username: str = Field(..., description="User identifier")
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    error: Optional[str] = None


class LogoutResponse(BaseModel):
    success: bool
    message: str


# Patient models
class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: str  # ISO format date
    medical_record_number: str


class PatientResponse(BaseModel):
    patient_id: str
    first_name: str
    last_name: str
    date_of_birth: str
    medical_record_number: str
    created_at: str


# Measurement models
class MeasurementStartRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    sample_id: Optional[str] = Field(None, description="Sample identifier")
    exposure_duration: int = Field(..., description="Exposure time in milliseconds")
    notes: Optional[str] = Field(None, description="Optional notes")


class MeasurementStartResponse(BaseModel):
    success: bool
    measurement_id: Optional[str] = None
    run_id: Optional[str] = None
    error: Optional[str] = None


class MeasurementStopRequest(BaseModel):
    run_id: str


class MeasurementResult(BaseModel):
    measurement_id: str
    run_id: str
    patient_id: str
    sample_id: str
    timestamp: str
    status: str  # "SUCCESS", "FAILED", "ABORTED"
    exposure_actual: int
    qc_passed: bool
    operator_id: str


class MeasurementHistoryResponse(BaseModel):
    measurements: List[MeasurementResult]
    total: int


# Calibration models
class CalibrationStartResponse(BaseModel):
    success: bool
    calibration_id: Optional[str] = None
    error: Optional[str] = None


class CalibrationStatus(BaseModel):
    id: Optional[str]
    timestamp: Optional[str]
    valid: bool
    expires_at: Optional[str]
    distance_check: Optional[bool]
    snr_threshold: Optional[float]


# System health models
class SafetyInterlocks(BaseModel):
    overall_safe: bool
    key_switch: bool
    enable_button: bool
    door_closed: bool
    emergency_stop: bool
    radiation_safe: bool
    cooling_ok: bool
    power_ok: bool


class DetectorStatus(BaseModel):
    powered: bool
    status: str  # OFF, IDLE, READY, ERROR
    temperature: Optional[float] = None
    voltage: Optional[float] = None


class MotionStatus(BaseModel):
    powered: bool
    status: str  # OFF, IDLE, READY, ERROR
    is_homed: Optional[bool] = None
    position: Optional[float] = None


class GpioStatus(BaseModel):
    powered: bool
    status: str  # Active, Off


class PduStatus(BaseModel):
    powered: bool
    status: str  # Active, Off
    outputs: Optional[dict] = None  # e.g., {"main_power": true}


class SystemHealth(BaseModel):
    state: str  # IDLE, RUNNING, SAFE, etc.
    interlocks: SafetyInterlocks
    calibration: Optional[CalibrationStatus]
    pdu: Optional[PduStatus] = None
    gpio: Optional[GpioStatus] = None
    detector: Optional[DetectorStatus] = None
    motion: Optional[MotionStatus] = None
    uptime: int
    cloud_connected: bool
    last_heartbeat: str


# WebSocket event models
class SystemEvent(BaseModel):
    type: str  # "system_health", "measurement_update", "safety_alert"
    data: dict
    timestamp: str


# Error response
class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None
    details: Optional[str] = None


# Calibration models
class CalibrationQCCheck(BaseModel):
    """QC check result for calibration."""
    value: Optional[float]
    threshold: Optional[float]
    passed: Optional[bool]


class CalibrationRecord(BaseModel):
    """Calibration record with QC data."""
    calibration_id: str
    timestamp: str
    operator_id: str
    calibrant_material: Optional[str] = None
    overall_pass: bool
    
    # QC checks
    total_intensity: Optional[CalibrationQCCheck] = None
    goodness_of_fit: Optional[CalibrationQCCheck] = None
    snr: Optional[CalibrationQCCheck] = None
    ring_quality: Optional[CalibrationQCCheck] = None
    
    # PONI results
    distance_mm: Optional[float] = None
    beam_center_x: Optional[float] = None
    beam_center_y: Optional[float] = None
    wavelength_angstrom: Optional[float] = None
    
    # Validity
    expires_at: Optional[str] = None
    invalidated_at: Optional[str] = None
    invalidation_reason: Optional[str] = None
    formatted_report: Optional[str] = None


class CalibrationCreateRequest(BaseModel):
    """Request to create calibration record."""
    calibrant_material: str = Field(..., description="Calibrant material (e.g., LaB6, CeO2)")
    exposure_duration: int = Field(..., description="Exposure time in milliseconds")


class CalibrationResponse(BaseModel):
    """Calibration operation response."""
    success: bool
    calibration: Optional[CalibrationRecord] = None
    error: Optional[str] = None


# Extended measurement models
class MeasurementDetail(BaseModel):
    """Extended measurement details with hardware data."""
    measurement_id: str
    patient_id: str
    timestamp: str
    operator_id: str
    measurement_type: str
    sample_id: Optional[str] = None
    
    # Calibration context
    calibration_id: Optional[str] = None
    calibration_timestamp: Optional[str] = None
    calibration_valid: Optional[bool] = None
    
    # Hardware data
    exposure_duration_ms: Optional[int] = None
    beam_intensity_mean: Optional[float] = None
    beam_intensity_std: Optional[float] = None
    snr: Optional[float] = None
    detector_temperature: Optional[float] = None
    detector_voltage: Optional[float] = None
    
    # QC
    qc_status: str
    qc_notes: Optional[str] = None
    clinical_notes: Optional[str] = None
    
    # Upload tracking
    uploaded_to_cloud: bool = False
    upload_timestamp: Optional[str] = None


# UI Command and System Event models
class UICommandLogEntry(BaseModel):
    """UI command log entry for audit trail."""
    log_id: Optional[int] = None
    timestamp: str
    session_id: str
    operator_id: str
    command_type: str
    command_payload: Optional[str] = None
    resource_id: Optional[str] = None
    result: str  # "success" or "failure"
    error_message: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SystemEventLogEntry(BaseModel):
    """System event log entry."""
    event_id: Optional[int] = None
    timestamp: str
    event_type: str
    severity: str  # "INFO", "WARNING", "ERROR", "CRITICAL"
    component: Optional[str] = None
    message: str
    details: Optional[str] = None
    operator_id: Optional[str] = None


# Audit query models
class AuditLogQuery(BaseModel):
    """Query parameters for audit logs."""
    operator_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)


class UICommandLogResponse(BaseModel):
    """Response containing UI command logs."""
    commands: List[UICommandLogEntry]
    total: int


class SystemEventLogResponse(BaseModel):
    """Response containing system event logs."""
    events: List[SystemEventLogEntry]
    total: int


# Backup models
class BackupInfo(BaseModel):
    """Backup file information."""
    path: str
    size_mb: float
    created: str
    patient_count: Optional[int] = None
    measurement_count: Optional[int] = None
    calibration_count: Optional[int] = None
    schema_version: Optional[int] = None


class BackupListResponse(BaseModel):
    """List of available backups."""
    backups: List[BackupInfo]
    total: int


class BackupCreateResponse(BaseModel):
    """Backup creation response."""
    success: bool
    backup_path: Optional[str] = None
    size_mb: Optional[float] = None
    patient_count: Optional[int] = None
    measurement_count: Optional[int] = None
    error: Optional[str] = None
