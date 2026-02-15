// System operational states (aligned with hardware API)
export type SystemState = 
  | 'IDLE' 
  | 'PENDING_ARMED' 
  | 'RUNNING' 
  | 'STOPPING' 
  | 'SAFE';

// Device status types (aligned with hardware API)
export type PduDeviceStatus = 'Active' | 'Off';
export type GpioDeviceStatus = 'Active' | 'Off';
// Note: Hardware uses 'initialized' boolean field separately, not as a status
export type DetectorDeviceStatus = 'OFF' | 'IDLE' | 'EXPOSING' | 'READING' | 'ERROR';
export type MotionDeviceStatus = 'OFF' | 'IDLE' | 'MOVING' | 'HOMING' | 'ERROR' | 'LIMIT_HIT';

// Interlock status from orchestrator /api/state (all fields required for safety)
export interface CompactInterlocks {
  overall_safe: boolean;
  key_switch: boolean;
  enable_button: boolean;
  door_closed: boolean;
  emergency_stop: boolean;
  cooling_ok: boolean;
  power_ok: boolean;
  radiation_safe: boolean;  // Actual field name from orchestrator API
}

// User roles
export type UserRole = 'operator' | 'engineer' | 'admin';

export interface User {
  username: string;
  role: UserRole;
  permissions: string[];
  id?: string;  // Optional user ID
}

// Measurement data (aligned with orchestrator POST /api/measurements/start)
export interface MeasurementParams {
  patient_id: string;   // UUID
  exposure_ms: number;  // Exposure duration in milliseconds (renamed from exposure_duration)
  sample_id?: string;   // Optional sample identifier
  notes?: string;       // Optional notes
}

export interface MeasurementResult {
  runId: string;
  sampleId: string;
  timestamp: Date;
  status: 'SUCCESS' | 'FAILED' | 'ABORTED';
  exposureActual: number;
  beamIntensityMean: number;
  beamIntensityStd: number;
  snr: number;
  qcPassed: boolean;
  calibrationId: string;
}

// Calibration data
export interface CalibrationStatus {
  id: string;
  timestamp: Date;
  valid: boolean;
  expiresAt: Date;
  distanceCheck: boolean;
  snrThreshold: number;
  parameters: Record<string, number>;
}

// Calibration QC check result
export interface CalibrationQcCheck {
  passed: boolean;
  measured: number;
  threshold: number;
  details?: string;  // Optional
}

// PONI Calibration Results
export interface PoniCalibrationResult {
  success?: boolean;  // Optional - may not be present
  distance_mm: number | null;
  beam_center_x: number | null;
  beam_center_y: number | null;
  wavelength_angstrom: number | null;
}

// Calibration QC Report (from POST /api/calibration/start and GET /api/calibration/latest)
export interface CalibrationQcReport {
  success: boolean;
  calibration_id: string | null;
  timestamp: string | null;
  calibrant_material: string | null;
  overall_pass: boolean | null;
  qc_checks?: {
    total_intensity: CalibrationQcCheck;
    goodness_of_fit: CalibrationQcCheck;
    snr: CalibrationQcCheck;
    ring_quality: CalibrationQcCheck;
    poni: PoniCalibrationResult;
  };
  formatted_report: string | null;
  error: string | null;
}

// Compact device status from /api/state
export interface CompactDeviceStatus {
  powered: boolean;
  status: PduDeviceStatus | GpioDeviceStatus | DetectorDeviceStatus | MotionDeviceStatus;
}

export interface CompactDevices {
  pdu: CompactDeviceStatus;
  gpio: CompactDeviceStatus;
  detector: CompactDeviceStatus;
  motion: CompactDeviceStatus;
}

// Detailed device status from /api/health
export interface DetailedPduStatus {
  powered: boolean;
  status: PduDeviceStatus;
  uptime_seconds: number;
  outputs: {
    main_power: boolean;
    detector_power: boolean;
    motion_power: boolean;
    gpio_power: boolean;
  };
}

// GPIO state from orchestrator GET /api/gpio/state
export interface GpioState {
  key_switch_on: boolean;
  enable_button_active: boolean;
  enable_button_remaining_secs: number;
  door_closed: boolean;
  emergency_stop_active: boolean;
  beam_shutter_closed: boolean;
}

// Enable button status from orchestrator GET /api/gpio/enable-button
export interface EnableButtonStatus {
  active: boolean;
  remaining_secs: number;
}

// Detailed GPIO status (for /api/health if needed)
export interface DetailedGpioStatus {
  powered: boolean;
  status: GpioDeviceStatus;
  key_switch_on: boolean;
  activation_button_active: boolean;
  activation_remaining_secs: number | null;
  interlocks: CompactInterlocks;
  main_led: string;
  radiation_led: string;
}

export interface DetailedDetectorStatus {
  powered: boolean;
  status: DetectorDeviceStatus;
  initialized: boolean;
  uptime_seconds?: number;  // Optional until orchestrator implements full health query
  temperature: number | null;
  voltage: number | null;
  total_exposures?: number;  // Optional until orchestrator implements full health query
  last_exposure_time_ms?: number | null;  // Optional until orchestrator implements full health query
}

export interface DetailedMotionStatus {
  powered: boolean;
  status: MotionDeviceStatus;
  initialized: boolean;
  uptime_seconds?: number;  // Optional until orchestrator implements full health query
  is_homed: boolean;
  position: number | null;
  target_position?: number | null;  // Optional until orchestrator implements full health query
  total_moves?: number;  // Optional until orchestrator implements full health query
}


// Compact system state from /api/state
export interface SystemStateResponse {
  system_state: SystemState;
  devices: CompactDevices;
  interlocks: CompactInterlocks;
  timestamp: string;
}

// Detailed system health from /api/health
export interface SystemHealthResponse {
  system_state: SystemState;
  uptime_seconds: number;
  timestamp: string;
  pdu: DetailedPduStatus;
  gpio: DetailedGpioStatus;
  detector: DetailedDetectorStatus;
  motion: DetailedMotionStatus;
  calibration: {
    valid: boolean;
    timestamp: string;
    expires_at: string;
    distance_check: boolean;
    snr_threshold: number;
  };
  cloud_connected: boolean;
  last_heartbeat: string;
}


// Audit event
export interface AuditEvent {
  id: string;
  timestamp: Date;
  eventType: string;
  userId: string;
  description: string;
  metadata: Record<string, unknown>;
}

// Patient data
export interface PatientCreate {
  first_name: string;
  last_name: string;
  date_of_birth: string; // ISO date format
  medical_record_number: string;
}

export interface Patient {
  patient_id: string;
  first_name: string;
  last_name: string;
  date_of_birth: string; // ISO date format
  medical_record_number: string;
  created_at: string;
}

// API responses
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  errorCode?: string;
}
