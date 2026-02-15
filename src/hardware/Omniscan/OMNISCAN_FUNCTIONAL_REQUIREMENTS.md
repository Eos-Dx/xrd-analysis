# Omniscan Functional Requirements Specification

**Document Version:** 1.0  
**Date:** November 17, 2025  
**Classification:** FDA Design History File (DHF) - Functional Requirements  
**Compliance:** IEC 62304 Class B, ISO 14971, IEC 62366-1  
**Status:** Derived from User Requirements and Code Analysis

---

## Executive Summary

This document specifies the functional requirements for the Omniscan medical X-ray diffraction (XRD) diagnostic platform. These requirements translate user needs into specific system behaviors and capabilities that must be implemented and verified.

**Subsystems Covered:**
1. **Hardware Server (Rust)** - Safety-critical hardware control
2. **Orchestrator (Python)** - Clinical workflow coordination
3. **User Interface (React/TypeScript)** - Clinical operator interface

**Relationship to Other Documents:**
- **Derived From:** OMNISCAN_USER_REQUIREMENTS_COMPLETE.md
- **Traces To:** System requirements (SYS_OMNI-SERVER_xxx)
- **Verified By:** Test specifications and validation protocols

---

## Table of Contents

1. [Safety System Functions](#1-safety-system-functions)
2. [Interlock Management Functions](#2-interlock-management-functions)
3. [State Machine Functions](#3-state-machine-functions)
4. [Device Control Functions](#4-device-control-functions)
5. [Calibration Functions](#5-calibration-functions)
6. [Measurement Acquisition Functions](#6-measurement-acquisition-functions)
7. [Motion Control Functions](#7-motion-control-functions)
8. [Data Management Functions](#8-data-management-functions)
9. [Audit and Logging Functions](#9-audit-and-logging-functions)
10. [Security and Authentication Functions](#10-security-and-authentication-functions)
11. [Communication Functions](#11-communication-functions)
12. [User Interface Functions](#12-user-interface-functions)
13. [Error Handling Functions](#13-error-handling-functions)
14. [Configuration Management Functions](#14-configuration-management-functions)
15. [Health Monitoring Functions](#15-health-monitoring-functions)

---

## 1. Safety System Functions

### FR-SAFE-001: Safety State Machine
**Derived From:** USR_OMNI-SERVER_001  
**Priority:** CRITICAL

**Function:** Implement comprehensive safety state machine with defined states and valid transitions.

**Inputs:**
- Interlock status (emergency stop, door, radiation, cooling, power)
- Key switch position
- Enable button status
- User commands
- System events (timeouts, calibration expiry)

**Processing:**
1. Maintain current safety state
2. Evaluate interlock status continuously
3. Validate state transition requests against allowed transitions
4. Execute state transitions with audit logging
5. Emit state change notifications

**Outputs:**
- Current safety state
- State transition success/failure
- Audit log entries
- State change notifications

**States Implemented:**
- `LOCKED` - System locked, no operations
- `INITIALIZED` - Key switch ON, awaiting operations
- `IDLE` - Ready for operations
- `WARMING_UP` - X-ray source heat-up
- `CALIBRATED` - Valid calibration active
- `PENDING_ARMED` - Awaiting enable button
- `RUNNING` - Active measurement
- `STOPPING` - Controlled shutdown
- `SAFE` - Emergency safe state
- `CALIBRATION` - Calibration in progress
- `MAINTENANCE` - Maintenance mode

**State Transition Rules:**
- `LOCKED` → `IDLE`: Key switch ON + interlocks safe
- `IDLE` → `PENDING_ARMED`: Start command + interlocks safe + valid calibration
- `PENDING_ARMED` → `RUNNING`: Enable button active
- `RUNNING` → `STOPPING`: Stop command or measurement complete
- `STOPPING` → `IDLE`: Shutdown complete + interlocks safe
- Any state → `SAFE`: Interlock violation

**Error Handling:**
- Reject invalid state transitions with clear error message
- Automatically transition to SAFE on interlock violation
- Log all transition failures for debugging

**Implementation Files:**
- `omniscan-hw-server/src/safety/mod.rs`

---

### FR-SAFE-002: Interlock Monitoring
**Derived From:** USR_OMNI-SERVER_001  
**Priority:** CRITICAL

**Function:** Continuously monitor all safety interlocks and evaluate overall safety status.

**Inputs:**
- Key switch position (GPIO)
- Emergency stop status (GPIO)
- Door closed sensor (GPIO)
- Beam intensity (detector watchdog)
- Over-temperature sensor (optional)

**Processing:**
1. Poll GPIO inputs at minimum 10 Hz
2. Evaluate each interlock against safety criteria
3. Compute overall safety status (AND of all required interlocks)
4. Generate violation reason string listing all failures
5. Update interlock status structure
6. Detect state changes and trigger notifications

**Outputs:**
- `InterlockStatus` structure with individual and overall status
- Violation reason (if not safe)
- Last check timestamp
- Interlock change notifications

**Safety Criteria:**
- Key switch: Must be in "operate" position for operations
- Emergency stop: Must NOT be pressed (released state)
- Door: Must be fully closed (sensor contact made)
- Beam watchdog: Beam intensity within ±10% of target
- Over-temperature: Temperature below threshold (if sensor present)

**Response Time:**
- Interlock check: < 100ms
- Emergency stop detection: < 50ms
- State transition to SAFE: < 100ms

**Implementation Files:**
- `omniscan-hw-server/src/safety/mod.rs` (InterlockStatus)
- `omniscan-hw-server/src/devices/gpio/mod.rs`

---

### FR-SAFE-003: Automatic Safety Shutdown
**Derived From:** USR_OMNI-SERVER_001  
**Priority:** CRITICAL

**Function:** Automatically shut down hazardous operations upon interlock violation.

**Inputs:**
- Interlock status change (violation detected)
- Current system state

**Processing:**
1. Detect interlock violation
2. Immediately command beam shutdown if active
3. Halt motion operations if in progress
4. Transition safety state to SAFE
5. Log violation event with details
6. Emit emergency notification

**Outputs:**
- Beam power OFF command
- Motion STOP command
- Safety state transition to SAFE
- Audit log entry
- Emergency notification to UI

**Shutdown Sequence:**
1. Beam disable (hardware interlock + software command): < 100ms
2. Motion halt: < 200ms
3. State transition: < 100ms
4. Notification emission: < 500ms

**Implementation Files:**
- `omniscan-hw-server/src/safety/mod.rs` (automatic transition)
- `omniscan-hw-server/src/grpc/services.rs` (shutdown commands)

---

## 2. Interlock Management Functions

### FR-INTLK-001: Key Switch Control
**Derived From:** USR_OMNI-SERVER_001, USR_OMNI-HW_001  
**Priority:** CRITICAL

**Function:** Enforce key switch requirement for system operations.

**Inputs:**
- Key switch position from GPIO (ON/OFF)

**Processing:**
1. Read key switch position from GPIO
2. Update interlock status
3. Control allowed operations:
   - Key OFF: Only status queries allowed
   - Key ON: All operations allowed if other interlocks satisfied

**Outputs:**
- Key switch status (boolean)
- Operation authorization (per command type)

**Behavior:**
- Key OFF → LOCKED state (or SAFE if was operational)
- Key ON + safe interlocks → IDLE state
- Reject operational commands if key OFF

---

### FR-INTLK-002: Activation Button Control
**Derived From:** USR_OMNI-HW_002  
**Priority:** CRITICAL

**Function:** Require physical activation button press for hazardous operations.

**Inputs:**
- Activation button press event (GPIO)
- Operation request (command type)

**Processing:**
1. On button press: Set active flag, start 20-second countdown timer
2. On command request: Check if button currently active
3. On timer expiry: Clear active flag, emit expiry notification
4. Validate operation type requires activation button

**Outputs:**
- Activation status (active/inactive)
- Remaining time (seconds)
- Command authorization (approved/rejected)

**Hazardous Operations (Require Button):**
- Initialize Detector
- Initialize Motion
- Start Exposure
- Move Motion

**Safe Operations (No Button Required):**
- Read States (all query commands)
- Stop Operations
- Power Off
- Get Interlocks

**Timing:**
- Activation duration: 20 seconds
- Countdown resolution: 1 second
- UI update frequency: 1 Hz

**Implementation Files:**
- `omniscan-hw-server/src/devices/gpio/mod.rs` (activation logic)
- `omniscan-hw-server/docs/ENABLE_BUTTON_LOGIC.md`

---

## 3. State Machine Functions

### FR-STATE-001: State Transition Validation
**Derived From:** USR_OMNI-SERVER_001  
**Priority:** CRITICAL

**Function:** Validate and execute state transitions according to defined rules.

**Inputs:**
- Requested target state
- Current state
- Interlock status
- Calibration validity
- Enable button status
- Transition reason

**Processing:**
1. Validate transition is allowed from current to target state
2. Check prerequisites for target state:
   - Interlocks satisfied
   - Calibration valid (for operational states)
   - Enable button active (for armed → running)
3. Execute transition if valid
4. Log transition with reason
5. Emit state change notification

**Outputs:**
- Transition success/failure
- New current state (if successful)
- Error message (if failed)
- Audit log entry
- State change notification

**Validation Rules:**
- Maintain state transition matrix
- Reject transitions with unsatisfied prerequisites
- Allow emergency transitions to SAFE from any state

---

### FR-STATE-002: Calibration State Enforcement
**Derived From:** USR_OMNI-SERVER_002  
**Priority:** CRITICAL

**Function:** Enforce calibration validity before allowing measurements.

**Inputs:**
- Last calibration timestamp
- Calibration validity duration (24 hours)
- Measurement request

**Processing:**
1. Check time since last calibration
2. Validate calibration status (valid/expired/not performed)
3. If expired: Transition to LOCKED state
4. If measurement requested without valid calibration: Reject
5. Display calibration status to user

**Outputs:**
- Calibration validity (boolean)
- Time until expiry (hours)
- Measurement authorization (approved/rejected)
- Calibration status message

**Enforcement Logic:**
```
if (now - last_calibration_time) > 24 hours:
    state = LOCKED
    reject measurements with error: "Calibration expired"
else:
    allow measurements if other conditions met
```

**Implementation Files:**
- `omniscan-hw-server/src/calibration.rs`
- `omniscan-hw-server/src/safety/mod.rs`

---

## 4. Device Control Functions

### FR-DEV-001: Detector Initialization
**Derived From:** USR_OMNI-HW_001  
**Priority:** CRITICAL

**Function:** Initialize X-ray detector with safety checks.

**Inputs:**
- Initialization command with context
- Key switch status
- Activation button status
- Interlock status

**Processing:**
1. Verify prerequisites:
   - Key switch ON
   - Activation button active (< 20 seconds)
   - All interlocks safe
2. Send power ON command to detector
3. Initialize detector communication
4. Configure detector parameters
5. Verify detector ready status
6. Update device health status

**Outputs:**
- Detector state (powered, initialized, temperature, status)
- Success/failure result
- Audit log entry
- State change notification

**Timing:**
- Prerequisite check: < 500ms
- Power ON: 2-5 seconds
- Initialization: 5-10 seconds
- Total timeout: 15 seconds

**Error Conditions:**
- Prerequisites not met → Reject with clear message
- Detector communication failure → SAFE state
- Initialization timeout → SAFE state

**Implementation Files:**
- `omniscan-hw-server/src/grpc/services.rs` (DeviceInitialization service)
- `omniscan-hw-server/src/devices/detectors/mod.rs`

---

### FR-DEV-002: Motion System Initialization
**Derived From:** USR_OMNI-HW_001  
**Priority:** HIGH

**Function:** Initialize motion control system with safety checks.

**Inputs:**
- Initialization command with context
- Key switch status
- Activation button status
- Interlock status

**Processing:**
1. Verify prerequisites (same as detector)
2. Send power ON command to motion controller
3. Initialize motion controller communication
4. Configure motion parameters (velocity, acceleration)
5. Verify controller ready (NOT homed yet - homing during calibration)
6. Update motion health status

**Outputs:**
- Motion state (powered, initialized, homed status, position)
- Success/failure result
- Audit log entry

**Timing:**
- Power ON: 1-3 seconds
- Initialization: 3-7 seconds
- Total timeout: 10 seconds

**Note:** Homing is performed during calibration workflow, not during initialization.

---

### FR-DEV-003: Device Power Management
**Derived From:** USR_OMNI-HW_001  
**Priority:** HIGH

**Function:** Control power distribution to devices via PDU.

**Inputs:**
- Device type (detector, motion, auxiliary)
- Power command (ON/OFF)
- Authorization level

**Processing:**
1. Validate user authorization for power control
2. Check device current power state
3. Send power command to PDU
4. Verify power state change
5. Update device health status
6. Log power event

**Outputs:**
- Power state confirmation
- Device health update
- Audit log entry

**Future Enhancement:** Controlled power distribution unit (PDU) for production.

---

## 5. Calibration Functions

### FR-CAL-001: Daily Calibration Execution
**Derived From:** USR_OMNI-SERVER_002  
**Priority:** CRITICAL

**Function:** Execute complete calibration workflow with QC validation.

**Inputs:**
- Calibration command with operator context
- Calibrant material specification (LaB6, Si, etc.)
- Current device state

**Processing:**
1. Verify prerequisites:
   - Detector initialized and ready
   - Motion powered (will be homed as part of calibration)
   - All interlocks safe
   - Calibrant loaded (operator confirmation)
2. Home motion system to known position
3. Move to calibration position (nominal sample-detector distance)
4. Perform calibration exposure (standard duration)
5. Acquire calibration image data
6. Run QC checks on image:
   - Total intensity check
   - Goodness of fit check
   - Signal-to-noise ratio (SNR) check
   - Ring quality check
7. Calculate PONI geometry parameters
8. Validate results against acceptance criteria
9. Store calibration record
10. Update calibration validity timestamp
11. Transition to CALIBRATED state if passed

**Outputs:**
- Calibration status (PASS/FAIL)
- QC report with individual check results
- PONI file content
- Formatted report for operator
- Calibration record (stored in database)
- State transition (LOCKED → CALIBRATED if passed)

**QC Checks:**
- **Total Intensity:** Minimum counts threshold (verify beam strength)
- **Goodness of Fit:** RMS error vs reference pattern (< threshold)
- **SNR:** Signal-to-noise ratio (> minimum threshold)
- **Ring Quality:** Continuity and sharpness of diffraction rings

**Timing:**
- Motion homing: 30-60 seconds
- Calibration exposure: 30-120 seconds
- Image processing + QC: 10-30 seconds
- Total: 2-5 minutes

**Calibration Validity:**
- Duration: 24 hours from completion timestamp
- Automatic expiry check before each measurement
- Transition to LOCKED state on expiry

**Implementation Files:**
- `omniscan-hw-server/src/calibration.rs`
- `omniscan-hw-server/src/calibration_qc.rs`
- `omniscan-hw-server/src/grpc/services.rs` (Acquisition::CalibrateDetector)

---

### FR-CAL-002: Calibration Status Query
**Derived From:** USR_OMNI-SERVER_002  
**Priority:** HIGH

**Function:** Provide current calibration status to operators and UI.

**Inputs:**
- Status query request

**Processing:**
1. Check last calibration timestamp
2. Calculate time since calibration
3. Determine validity status (valid/expired/not performed)
4. Calculate time until expiry
5. Format user-friendly status message

**Outputs:**
- Calibration validity (boolean)
- Last calibration timestamp
- Time until expiry (hours)
- Status message in plain language
- QC report (if available)

**Status Messages:**
- Not performed: "Daily calibration has not been performed"
- Valid: "Calibration is valid and current (expires in X hours)"
- Expired: "Calibration has expired. Run daily calibration before measurements."
- Failed: "Calibration failed validation. Contact service engineer."

---

### FR-CAL-003: Calibration History
**Derived From:** USR_OMNI-SERVER_010  
**Priority:** MEDIUM

**Function:** Maintain historical calibration records for trending and diagnostics.

**Inputs:**
- Calibration record (after each calibration)
- Historical query request (optional date range)

**Processing:**
1. Store complete calibration record in database:
   - Calibration ID (UUID)
   - Timestamp
   - Operator
   - Calibrant material
   - QC results (all checks)
   - PONI parameters
   - Pass/fail status
2. On query: Retrieve records matching criteria
3. Calculate trend statistics
4. Generate trend graphs (if requested)

**Outputs:**
- Calibration record list
- Trend data (distance, beam center, SNR over time)
- Statistical analysis (mean, std deviation, drift detection)

**Retention:**
- Store all calibration records indefinitely
- Enable export for external analysis

---

## 6. Measurement Acquisition Functions

### FR-MEAS-001: Start Measurement
**Derived From:** USR_OMNI-SERVER_005  
**Priority:** CRITICAL

**Function:** Execute diagnostic X-ray measurement with precise timing and safety monitoring.

**Inputs:**
- Measurement command with context (command_id = measurement_id)
- Duration (seconds)
- Sample ID
- Operator ID
- Activation button status

**Processing:**
1. Verify prerequisites:
   - Valid calibration (< 24 hours)
   - Detector initialized and ready
   - All interlocks safe
   - Activation button active
2. Generate unique measurement_id (UUID)
3. Transition to PENDING_ARMED state
4. Wait for enable button confirmation (or fail if button expires)
5. Transition to RUNNING state
6. Start detector exposure for specified duration
7. Monitor beam intensity continuously (10 Hz)
8. Emit progress notifications (1 Hz) with elapsed/total time
9. On completion or timer expiry:
   - Stop detector exposure
   - Transition to STOPPING state
   - Save raw image data
   - Store measurement record
   - Transition to IDLE state
10. Emit completion notification

**Outputs:**
- Measurement ID (UUID)
- Raw XRD image data (2D array)
- Measurement metadata:
  - Timestamp (start/end)
  - Duration (actual vs programmed)
  - Beam intensity (mean, std deviation)
  - Quality flags
  - Calibration ID used
  - Sample ID
  - Operator ID
- Progress notifications (1 Hz during exposure)
- Completion status (completed/failed/stopped/aborted)

**Timing Requirements:**
- Exposure duration: Programmed ±50ms tolerance
- Beam intensity monitoring: 10 Hz
- Progress updates: 1 Hz
- Emergency stop response: < 100ms

**Beam Intensity Monitoring:**
- Measure beam intensity every 100ms
- Check against thresholds: target ±10%
- If out of range for > 500ms: Abort exposure, SAFE state

**Notification Format (1 Hz):**
```
RUN TICK:measurement:<measurement_id>:<elapsed>/<total>
```

**Error Conditions:**
- Invalid calibration → Reject: "Calibration expired"
- Interlocks unsafe → Reject: "Safety interlocks not satisfied"
- Enable button expired → Reject: "Enable button not active"
- Beam intensity out of range → Abort, transition to SAFE

**Implementation Files:**
- `omniscan-hw-server/src/grpc/services.rs` (Acquisition::StartExposure)
- `omniscan-hw-server/src/measurement/mod.rs`
- `omniscan-hw-server/docs/COMMANDS_SPEC.md`

---

### FR-MEAS-002: Stop Measurement
**Derived From:** USR_OMNI-SERVER_005  
**Priority:** HIGH

**Function:** Gracefully stop active measurement and save partial data.

**Inputs:**
- Stop command with context

**Processing:**
1. Verify measurement is active (RUNNING state)
2. Send stop command to detector
3. Transition to STOPPING state
4. Wait for detector to complete current frame
5. Save partial image data
6. Update measurement record with stopped status
7. Transition to IDLE state
8. Emit stopped notification

**Outputs:**
- Partial measurement data
- Updated measurement record (status = STOPPED)
- Stopped notification

**Timing:**
- Stop acknowledgment: < 1 second
- Detector shutdown: < 5 seconds
- State transition to IDLE: < 10 seconds

---

### FR-MEAS-003: Abort Measurement
**Derived From:** USR_OMNI-SERVER_005  
**Priority:** HIGH

**Function:** Emergency abort of measurement (discard data).

**Inputs:**
- Abort command or interlock violation

**Processing:**
1. Immediately disable beam (hardware + software)
2. Halt detector acquisition
3. Discard incomplete data
4. Update measurement record with aborted status
5. Transition to SAFE state
6. Emit abort notification

**Outputs:**
- Aborted status
- Audit log entry with abort reason
- Notification to UI

**Timing:**
- Beam disable: < 100ms
- State transition: < 200ms

---

### FR-MEAS-004: Get Exposure Results
**Derived From:** USR_OMNI-SERVER_006  
**Priority:** HIGH

**Function:** Retrieve measurement image data for processing and analysis.

**Inputs:**
- Measurement ID (UUID)
- Optional encoding preference (raw, zstd compressed)

**Processing:**
1. Lookup measurement record by ID
2. Load raw image data from storage
3. Prepare response with metadata:
   - Image dimensions (rows, cols)
   - Data type (uint16, float32)
   - Encoding (raw or compressed)
4. Optionally compress with zstd
5. Package binary payload

**Outputs:**
- Measurement ID
- Image shape (rows, cols)
- Data type enum
- Encoding enum
- Binary image data (bytes)

**Data Format:**
- Row-major binary array
- Efficient for NumPy reconstruction: `np.frombuffer(data, dtype).reshape(rows, cols)`

**Implementation Files:**
- `omniscan-hw-server/src/grpc/services.rs` (Acquisition::GetExposureResults)
- `omniscan-hw-server/docs/COMMANDS_SPEC.md`

---

## 7. Motion Control Functions

### FR-MOT-001: Motion Homing
**Derived From:** USR_OMNI-MOTION_001  
**Priority:** HIGH

**Function:** Execute multi-phase homing sequence to find home position.

**Inputs:**
- Home command with context
- Interlock status

**Processing:**
1. Verify prerequisites:
   - Motion initialized
   - Interlocks safe
2. Execute homing phases:
   - **Init:** Prepare for homing, clear position
   - **Search:** Move toward home switch at search speed
   - **Backoff:** Move away from switch after contact
   - **Latch:** Approach switch slowly for precision
   - **Done:** Set home position, mark as homed
3. Emit position updates (10 Hz) during motion
4. Emit phase updates with each phase transition
5. On completion: Set homed flag, store home position

**Outputs:**
- Homed status (boolean)
- Home position (mm)
- Motion events (HOME_STARTED, HOME_COMPLETED)
- Position notifications (10 Hz)

**Notification Format (10 Hz):**
```
TICK:motion_home:<command_id>:pos=<mm>:phase=<init|search|backoff|latch|done>
```

**Timing:**
- Search phase: Variable (depends on initial position)
- Latch phase: 5-10 seconds
- Total: 30-60 seconds typical

---

### FR-MOT-002: Motion Move to Position
**Derived From:** USR_OMNI-MOTION_001  
**Priority:** HIGH

**Function:** Move motion axis to absolute position with real-time feedback.

**Inputs:**
- Target position (mm)
- Command context
- Interlock status

**Processing:**
1. Verify prerequisites:
   - Motion initialized and homed
   - Interlocks safe
   - Target within limits
2. Calculate required move distance
3. Start non-blocking move
4. Emit position updates (10 Hz) during motion
5. Detect limit conditions
6. On completion:
   - Verify final position
   - Update motion state
   - Emit completion event

**Outputs:**
- Move acknowledgment (immediate)
- Position notifications (10 Hz)
- Motion events (MOVE_STARTED, MOVE_COMPLETED, LIMIT_HIT)
- Final position

**Notification Format (10 Hz):**
```
TICK:motion:<command_id>:pos=<mm>
```

**Timing:**
- Acknowledgment: < 1 second
- Move time: Base 2s + (0.1s × distance_mm)
- Position update rate: 10 Hz

**Limit Detection:**
- Software limits: Check before move
- Hardware limits: Stop immediately, emit LIMIT_HIT event

---

### FR-MOT-003: Motion Velocity Control
**Derived From:** USR_OMNI-MOTION_001  
**Priority:** MEDIUM

**Function:** Adjust motion velocity for different operations.

**Inputs:**
- Velocity setpoint (mm/s)
- Command context

**Processing:**
1. Validate velocity within allowed range
2. Send velocity command to motion controller
3. Verify velocity acceptance
4. Update motion parameters

**Outputs:**
- Velocity acknowledgment
- Updated motion health status

**Velocity Ranges:**
- Minimum: 0.1 mm/s
- Maximum: 100 mm/s (configurable)
- Typical calibration: 10 mm/s
- Typical measurement: 5 mm/s

---

## 8. Data Management Functions

### FR-DATA-001: Measurement Data Storage
**Derived From:** USR_OMNI-SERVER_007  
**Priority:** CRITICAL

**Function:** Store measurement data with fault tolerance and traceability.

**Inputs:**
- Measurement data (image, metadata)
- Measurement ID
- Sample ID
- Operator context

**Processing:**
1. Begin transaction
2. Store measurement record in database:
   - Measurement ID (UUID)
   - Timestamp (start, end)
   - Duration (programmed, actual)
   - Sample ID
   - Operator ID
   - Device serial number
   - Calibration ID used
   - Status (completed/failed/stopped/aborted)
   - Quality metrics (beam intensity stats, SNR)
3. Write raw image data to file storage:
   - Format: Binary array (uint16 or float32)
   - Filename: `<measurement_id>.dat`
   - Checksum: SHA-256 for integrity
4. Write metadata sidecar:
   - Format: JSON
   - Filename: `<measurement_id>.json`
5. Commit transaction
6. On failure: Rollback transaction, preserve partial data with error flag

**Outputs:**
- Storage success confirmation
- File paths
- Checksum
- Database record ID

**Fault Tolerance:**
- Transaction-based writes (atomic commit)
- Automatic rollback on failure
- Graceful shutdown detection
- Data recovery on restart

**Implementation Files:**
- `omniscan-hw-server/src/measurement/mod.rs`
- `omniscan-hw-server/src/audit/mod.rs`

---

### FR-DATA-002: Metadata Traceability
**Derived From:** USR_OMNI-SERVER_009  
**Priority:** CRITICAL

**Function:** Capture complete metadata for regulatory traceability.

**Inputs:**
- Measurement event
- System state at time of measurement

**Processing:**
1. Capture device information:
   - Device serial number
   - Hardware configuration
   - Software version
   - Firmware versions
2. Capture operator information:
   - Operator ID
   - Authentication level
   - Authentication timestamp
3. Capture temporal information:
   - Measurement timestamp (UTC with timezone)
   - Duration
   - Completion status
4. Capture calibration linkage:
   - Calibration ID
   - Calibration timestamp
   - Calibration validity status
5. Capture measurement parameters:
   - Exposure duration
   - Beam intensity (mean, std)
   - Detector configuration
   - Motion position
6. Capture quality metrics:
   - SNR
   - Beam stability
   - Any quality flags
7. Capture environmental data:
   - System state at start/end
   - Active warnings
   - Interlock status history
8. Store as JSON sidecar with measurement

**Outputs:**
- Complete metadata JSON file
- Traceability links in database

**Metadata Structure:**
```json
{
  "measurement_id": "uuid",
  "sample_id": "string",
  "operator": {"id": "string", "auth_level": "string"},
  "device": {"serial": "string", "hw_config": {}, "sw_version": "string"},
  "timestamp": {"start": "ISO8601", "end": "ISO8601"},
  "calibration": {"id": "uuid", "timestamp": "ISO8601", "valid": true},
  "parameters": {"duration_s": 60, "beam_intensity": {"mean": 1234, "std": 12}},
  "quality": {"snr": 45.6, "beam_stability": 0.98},
  "environment": {"state": "RUNNING", "interlocks": {...}}
}
```

---

### FR-DATA-003: Data Export
**Derived From:** USR_OMNI-SERVER_012  
**Priority:** MEDIUM

**Function:** Export measurement data in standard formats for analysis.

**Inputs:**
- Export request
- Measurement ID(s)
- Format specification (CSV, JSON, DICOM)

**Processing:**
1. Retrieve measurement records
2. Load image data and metadata
3. Convert to requested format:
   - **CSV:** Tabular metadata + pointer to image file
   - **JSON:** Complete metadata + base64 image or file reference
   - **DICOM:** DICOM-wrapped XRD data (future)
4. Package for export
5. Log export event (audit trail)

**Outputs:**
- Exported data file(s)
- Export manifest
- Audit log entry

---

## 9. Audit and Logging Functions

### FR-AUDIT-001: Command Logging
**Derived From:** USR_OMNI-SERVER_021  
**Priority:** CRITICAL

**Function:** Log all commands with complete context for audit trail.

**Inputs:**
- Command context (command_id, user, reason, timestamp)
- Service name
- Command name
- Command inputs (parameters)
- Execution result (success/failure)
- Error message (if failed)
- Execution time (ms)

**Processing:**
1. Create command log entry:
   - Command ID (UUID)
   - Session ID
   - Timestamp (UTC)
   - Command type ("Service.Command" format)
   - Operator ID (from context)
   - Orchestrator ID (from mTLS if available)
   - Command data (JSON)
   - Result (success/failure)
   - Execution time (ms)
   - Error message (optional)
2. Write to audit database (append-only)
3. Optional: Emit to external audit system

**Outputs:**
- Audit log entry (immutable)
- Log confirmation

**Database Schema:**
```sql
CREATE TABLE command_logs (
    id UUID PRIMARY KEY,
    session_id VARCHAR,
    timestamp TIMESTAMP WITH TIME ZONE,
    command_type VARCHAR,
    command_data JSONB,
    hw_server_command_id UUID,
    user_context VARCHAR,
    result VARCHAR,
    execution_time_ms INTEGER,
    error_message TEXT
);
```

**Implementation Files:**
- `omniscan-hw-server/src/audit/mod.rs`
- `omniscan-orchestrator/src/omniscan_orchestrator/database.py`

---

### FR-AUDIT-002: State Transition Logging
**Derived From:** USR_OMNI-SERVER_021  
**Priority:** CRITICAL

**Function:** Log all safety state transitions with reasons.

**Inputs:**
- Previous state
- New state
- Transition reason
- Timestamp
- Interlock status at time of transition

**Processing:**
1. Create state transition log entry
2. Include reason category:
   - User command
   - Interlock violation
   - System timeout
   - Calibration expiry
   - Maintenance mode
   - Emergency abort
3. Store in audit database
4. Emit notification

**Outputs:**
- State transition audit entry
- Notification

---

### FR-AUDIT-003: Audit Trail Export
**Derived From:** USR_OMNI-SERVER_021  
**Priority:** HIGH

**Function:** Export audit trail in readable formats for regulatory review.

**Inputs:**
- Export request
- Date range (optional)
- Filter criteria (user, command type, etc.)

**Processing:**
1. Query audit database with filters
2. Retrieve matching log entries
3. Format according to requested type:
   - **CSV:** Tabular format with all columns
   - **PDF:** Formatted report with summary
   - **JSON:** Structured data for programmatic access
4. Add cryptographic verification (digital signature)
5. Package export with manifest

**Outputs:**
- Export file
- Manifest with entry count and date range
- Digital signature

**Implementation Files:**
- `omniscan-orchestrator/src/omniscan_orchestrator/audit.py`

---

## 10. Security and Authentication Functions

### FR-SEC-001: User Authentication
**Derived From:** USR_OMNI-SERVER_016  
**Priority:** CRITICAL

**Function:** Authenticate users with role-based access control.

**Inputs:**
- Username
- Password
- Optional: Certificate (for privileged roles)

**Processing:**
1. Validate credentials against user database
2. Verify password hash (bcrypt or Argon2)
3. Check account status (active/locked)
4. Check password expiry
5. For privileged operations: Verify certificate
6. Generate session token (JWT)
7. Log authentication event

**Outputs:**
- Authentication result (success/failure)
- Session token (if successful)
- User role
- Session ID

**Roles Supported:**
- **Clinical Operator:** Measurements, calibration
- **Senior Staff:** Audit logs, historical data
- **Maintenance Engineer:** Maintenance mode, diagnostics
- **Administrator:** User management, configuration

**Implementation Files:**
- `omniscan-orchestrator/src/omniscan_orchestrator/auth_manager.py`

---

### FR-SEC-002: mTLS Certificate Validation
**Derived From:** USR_OMNI-SERVER_004, USR_OMNI-SERVER_016  
**Priority:** CRITICAL

**Function:** Validate client certificates for orchestrator-to-hardware-server communication.

**Inputs:**
- Client certificate (from TLS handshake)
- Expected device UUID

**Processing:**
1. Extract certificate from TLS connection
2. Verify certificate signature against CA
3. Check certificate validity (not expired)
4. Extract device UUID from certificate Subject
5. Verify device UUID matches server configuration
6. Extract engineer ID from certificate
7. Log authentication

**Outputs:**
- Certificate validation result
- Device UUID
- Engineer ID
- Session authorization

**Implementation Files:**
- `omniscan-hw-server/src/auth.rs`
- `omniscan-hw-server/src/certificates.rs`
- `omniscan-certificate-center/` (certificate generation)

---

### FR-SEC-003: Data Encryption
**Derived From:** USR_OMNI-SERVER_008  
**Priority:** CRITICAL

**Function:** Encrypt patient data at rest using AES-256-GCM.

**Inputs:**
- Plaintext data
- Encryption key (from TPM or key store)

**Processing:**
1. Retrieve encryption key securely
2. Generate random nonce (IV)
3. Encrypt data using AES-256-GCM
4. Compute authentication tag
5. Package: nonce || ciphertext || tag

**Outputs:**
- Encrypted data blob
- Authentication tag

**Key Management:**
- Keys stored in TPM or system keyring
- Automatic key rotation (future)
- Secure key deletion

**Implementation Files:**
- `omniscan-hw-server/src/encryption.rs`
- `omniscan-orchestrator/src/omniscan_orchestrator/encryption.py`

---

## 11. Communication Functions

### FR-COMM-001: gRPC Service Interface
**Derived From:** USR_OMNI-SERVER_008 (implied)  
**Priority:** CRITICAL

**Function:** Provide gRPC API for orchestrator-to-hardware-server communication.

**Services Implemented:**
- **Acquisition:** StartExposure, Stop, Abort, GetState, CalibrateDetector, GetLastCalibration
- **Motion:** MoveTo, MoveRelative, Home, Stop, SetVelocity, GetPosition
- **DeviceInitialization:** InitializeDetector, InitializeMotion, PowerOffDetector, PowerOffMotion
- **DeviceControl:** PowerDevice, GetDetectorHealth, GetMotionHealth, GetDeviceState
- **Safety:** GetInterlockStatus, ResetInterlocks, CheckSafetyToOperate
- **Health:** Liveness, Readiness, GetAggregateHealth
- **StateMonitor:** SubscribeToStateUpdates, GetFullServerState, GetGpioState, GetDetectorState, GetMotionState
- **CommandDiscovery:** ListServices, ListCommands (introspection)

**Command Context (Required for All Commands):**
```protobuf
message CommandContext {
  string command_id = 1;  // UUID
  string user = 2;        // Operator ID
  string reason = 3;      // Free text reason
  Timestamp timestamp = 4; // UTC timestamp
}
```

**Implementation Files:**
- `omniscan-hw-server/src/grpc/services.rs`
- `omniscan-hw-server/protoc/hub.v1.proto`

---

### FR-COMM-002: State Change Notifications
**Derived From:** Implicit from architecture  
**Priority:** HIGH

**Function:** Broadcast state change notifications to subscribed clients.

**Inputs:**
- State change event (component, change_type)

**Processing:**
1. Detect state change (safety state, GPIO, device state)
2. Create notification message:
   - Component (CMD, RUN, GPIO, DETECTOR, MOTION)
   - Change type (specific change description)
   - Timestamp
3. Broadcast to all subscribers via stream

**Outputs:**
- Stream of StateChangeNotification messages

**Notification Types:**
- **CMD:** Command lifecycle (START, DONE)
- **RUN:** Long-running operation progress (measurement, calibration, motion)
- **GPIO:** Interlock changes, enable button
- **DETECTOR:** Power, status changes
- **MOTION:** Power, position, homing status changes

**Format Examples:**
```
CMD:START:Acquisition.StartExposure:<command_id>
RUN:TICK:measurement:<run_id>:<elapsed>/<total>
GPIO:CHANGE:enable_button:active:19
```

**Implementation Files:**
- `omniscan-hw-server/src/grpc/state_monitor_service.rs`
- `omniscan-hw-server/docs/STATE_CHANGE_NOTIFICATIONS.md`

---

### FR-COMM-003: WebSocket Real-Time Updates
**Derived From:** Implicit from UI requirements  
**Priority:** HIGH

**Function:** Provide WebSocket connection for UI real-time updates.

**Inputs:**
- Session ID (authentication)
- Subscription requests

**Processing:**
1. Accept WebSocket connection with session auth
2. Subscribe to hardware server state changes (via gRPC stream)
3. Transform state changes to WebSocket messages
4. Broadcast to connected UI clients
5. Handle client subscriptions (selective updates)

**Outputs:**
- WebSocket messages:
  - `system_health`: Periodic health updates
  - `gpio_update`: Enable button, interlocks
  - `state_change`: Safety state transitions
  - `calibration_start`: Calibration initiated
  - `calibration_complete`: Calibration results
  - `measurement_progress`: Measurement updates

**Implementation Files:**
- `omniscan-orchestrator/` (WebSocket server)
- `omniscan-ui/src/services/websocket.ts` (client)

---

## 12. User Interface Functions

### FR-UI-001: Dashboard Display
**Derived From:** USR_OMNI-SERVER_011  
**Priority:** HIGH

**Function:** Display system status dashboard with real-time updates.

**Inputs:**
- System state (from API/WebSocket)
- Interlock status
- Calibration status
- Device health
- Enable button status

**Processing:**
1. Subscribe to real-time updates (WebSocket)
2. Poll system state every 3 seconds (fallback)
3. Display status panels:
   - Safety state (large, prominent)
   - Interlocks (with icons: ✓ or ✗)
   - Calibration status (valid/expired/hours remaining)
   - Device status (detector, motion)
   - Enable button (countdown timer if active)
4. Color-code status:
   - Green: Safe and ready
   - Yellow: Warning (calibration expiring, etc.)
   - Red: Fault or unsafe
5. Show last measurement details

**Outputs:**
- Dashboard UI with status panels
- Visual indicators (colors, icons)
- Real-time countdown timers

**Implementation Files:**
- `omniscan-ui/src/pages/Dashboard.tsx`

---

### FR-UI-002: Measurement Interface
**Derived From:** USR_OMNI-SERVER_011  
**Priority:** HIGH

**Function:** Provide interface for starting and monitoring measurements.

**Inputs:**
- Sample ID (manual or barcode)
- Operator selection
- Measurement duration

**Processing:**
1. Validate inputs (sample ID format)
2. Check prerequisites (calibration valid, interlocks safe)
3. Display enable button status and countdown
4. On Start button click:
   - Send StartExposure command to API
   - Display progress bar with real-time updates
   - Show elapsed/total time
   - Show beam intensity graph (if available)
5. Allow Stop button during measurement
6. On completion:
   - Display success/failure status
   - Show measurement ID
   - Offer navigation to results

**Outputs:**
- Measurement form
- Progress display
- Results confirmation

**Implementation Files:**
- `omniscan-ui/src/pages/MeasurementPage.tsx`

---

### FR-UI-003: Calibration Interface
**Derived From:** USR_OMNI-SERVER_002  
**Priority:** HIGH

**Function:** Guide operator through calibration workflow with clear pass/fail indication.

**Inputs:**
- Calibrant material selection
- Operator confirmation (standard loaded)

**Processing:**
1. Display current calibration status
2. Check prerequisites (devices ready, interlocks safe)
3. Display step-by-step instructions:
   - "Load calibration standard"
   - "Press enable button"
   - "Click Start Calibration"
4. On Start:
   - Disable UI during calibration
   - Display progress bar
   - Show status: "Homing...", "Exposing...", "Processing..."
5. On completion:
   - Display QC report with check results:
     - Total Intensity: PASS/FAIL
     - Goodness of Fit: PASS/FAIL
     - SNR: PASS/FAIL
     - Ring Quality: PASS/FAIL
   - Overall: PASS (green) or FAIL (red)
6. If FAIL:
   - Display corrective actions
   - Offer "Retry" or "Contact Service"

**Outputs:**
- Calibration wizard UI
- Progress display
- QC report card with pass/fail indicators
- Corrective action guidance

**Implementation Files:**
- `omniscan-ui/src/pages/CalibrationPage.tsx`

---

### FR-UI-004: Error Display
**Derived From:** USR_OMNI-SERVER_017  
**Priority:** HIGH

**Function:** Display user-friendly error messages with troubleshooting steps.

**Inputs:**
- Error response from API
- Error code
- Technical details

**Processing:**
1. Parse error response
2. Map error code to user-friendly message
3. Generate troubleshooting steps based on error type
4. Format error display:
   - Plain language description
   - Numbered troubleshooting checklist
   - Contact information (if escalation needed)
   - Error reference code
5. Display as modal or notification

**Example Error Messages:**
```
X-ray Detector Not Responding (Error: DET-001)

Please check:
1. Detector power cable is firmly connected
2. Network cable is connected to detector
3. Detector power indicator shows green light

If all checks pass and problem persists:
Contact Service: 1-800-555-OMNI
Reference: DET-001, Session: abc-123-xyz
```

**Implementation Files:**
- `omniscan-ui/src/services/errors.ts`
- `omniscan-ui/src/components/common/NotificationToast.tsx`

---

## 13. Error Handling Functions

### FR-ERR-001: Error Response Formatting
**Derived From:** USR_OMNI-SERVER_017, USR_OMNI-SERVER_018  
**Priority:** HIGH

**Function:** Format error responses with user-friendly and technical information.

**Inputs:**
- Error type
- Error context
- System state

**Processing:**
1. Generate two-tier error information:
   - **User-facing:** Plain language, actionable steps
   - **Technical:** Error codes, stack traces, system state
2. Store technical details in logs
3. Return user-friendly message in API response

**Outputs:**
- Error response with:
  - `user_message`: Plain language description
  - `troubleshooting_steps`: Array of actions
  - `error_code`: Reference code (e.g., "DET-001")
  - `should_contact_support`: Boolean
  - `support_info`: Contact details
  - `technical_details`: (optional, for admins)

**Error Categories:**
- **Safety:** Interlock violations, unsafe conditions
- **Device:** Communication failures, initialization errors
- **Calibration:** Expired, QC failure
- **Measurement:** Prerequisites not met, beam failures
- **System:** Configuration errors, resource failures

---

### FR-ERR-002: Graceful Degradation
**Derived From:** USR_OMNI-SERVER_013  
**Priority:** HIGH

**Function:** Maintain safe state during partial system failures.

**Inputs:**
- Component failure event
- Current system state

**Processing:**
1. Detect component failure (detector, motion, network)
2. Assess impact on safety and operations
3. Disable affected operations only
4. Transition to safe state if hazard present
5. Display clear status to user:
   - What failed
   - What operations are disabled
   - What operations remain available
   - Recovery steps
6. Log failure for diagnostics
7. Enable automatic recovery attempt (if safe)

**Example:**
- Motion controller failure during idle:
  - Disable motion-dependent operations (calibration)
  - Allow detector-only operations (if safe)
  - Display: "Motion system unavailable - calibration disabled until restored"

---

## 14. Configuration Management Functions

### FR-CFG-001: Configuration Loading
**Derived From:** Implicit from system design  
**Priority:** HIGH

**Function:** Load and validate system configuration on startup.

**Inputs:**
- Configuration file (TOML format)
- Environment variables (optional overrides)

**Processing:**
1. Load configuration file
2. Parse TOML structure
3. Validate all required fields present
4. Validate value ranges (timeouts, thresholds)
5. Apply environment variable overrides
6. Store in memory configuration

**Outputs:**
- Validated configuration structure
- Validation errors (if invalid)

**Configuration Sections:**
- **Device:** Device type, GUI mode, interlocks
- **Timeouts:** All command timeouts (seconds)
- **Workflow:** Enable button timeout, calibration interval
- **Certificates:** mTLS configuration, device UUID
- **Calibration:** QC thresholds, acceptance criteria
- **Storage:** Database paths, data directories
- **Logging:** Log level, retention

**Implementation Files:**
- `omniscan-hw-server/src/config.rs`

---

### FR-CFG-002: Runtime Configuration Update
**Derived From:** USR_OMNI-SERVER_019 (implied)  
**Priority:** MEDIUM

**Function:** Allow runtime configuration updates for non-safety-critical parameters.

**Inputs:**
- Configuration update request
- New configuration values
- Authorization credentials

**Processing:**
1. Verify user authorization (admin or engineer)
2. Validate new configuration
3. Check if update requires restart (safety parameters)
4. If safe to apply immediately:
   - Update in-memory configuration
   - Write to configuration file
   - Log configuration change
5. If restart required:
   - Validate and save
   - Notify user restart needed
   - Log pending configuration

**Outputs:**
- Update success/failure
- Restart required flag
- Updated configuration

**Updateable Parameters:**
- Timeouts (non-safety)
- Logging levels
- Network settings
- UI preferences

**Restricted Parameters (Require Restart):**
- Safety thresholds
- Interlock configuration
- Calibration interval
- Device identifiers

---

## 15. Health Monitoring Functions

### FR-HEALTH-001: System Health Aggregation
**Derived From:** USR_OMNI-SERVER_013  
**Priority:** HIGH

**Function:** Aggregate health status from all system components.

**Inputs:**
- Detector health
- Motion health
- GPIO status
- Interlock status
- Calibration status
- Network connectivity

**Processing:**
1. Query health from each component
2. Evaluate overall health:
   - All components OK → Healthy
   - Non-critical component warning → Degraded
   - Safety-critical component failure → Fault
3. Generate health summary
4. Compute readiness for operations:
   - Hardware ready: Detector + Motion powered and idle
   - Safety ready: All interlocks pass
   - Calibration ready: Valid calibration
   - System ready: All above satisfied

**Outputs:**
- Aggregate health status
- Component health details
- Readiness flags
- Health report

**Health Endpoint Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-11-17T12:00:00Z",
  "isHardwareReady": true,
  "devices": {
    "detector": {"powered": true, "status": "Idle"},
    "motion": {"powered": true, "status": "Idle", "is_homed": true}
  },
  "safety": {"overall_safe": true},
  "calibration": {"valid": true, "hours_until_expiry": 18.5}
}
```

**Implementation Files:**
- `omniscan-hw-server/src/api/mod.rs` (health_check)

---

### FR-HEALTH-002: Liveness Probe
**Derived From:** Implicit from system design  
**Priority:** MEDIUM

**Function:** Provide liveness probe for monitoring systems.

**Inputs:**
- Probe request

**Processing:**
1. Respond immediately if server is running

**Outputs:**
- HTTP 200 OK or gRPC Empty response

**Use Case:** Kubernetes/monitoring tools checking if process is alive.

---

### FR-HEALTH-003: Readiness Probe
**Derived From:** Implicit from system design  
**Priority:** MEDIUM

**Function:** Provide readiness probe indicating system is ready for operations.

**Inputs:**
- Readiness request

**Processing:**
1. Check if server initialization complete
2. Check if critical components are operational
3. Respond based on readiness

**Outputs:**
- Ready: HTTP 200 OK / gRPC Empty
- Not Ready: HTTP 503 Service Unavailable / gRPC UNAVAILABLE with reason

**Use Case:** Load balancers checking if server can accept traffic.

---

## Appendix A: Functional Requirements Traceability Matrix

| Functional Req | User Req | System Req | Implementation File |
|----------------|----------|------------|---------------------|
| FR-SAFE-001 | USR_001 | SYS_003 | safety/mod.rs |
| FR-SAFE-002 | USR_001 | SYS_002 | safety/mod.rs, devices/gpio/mod.rs |
| FR-SAFE-003 | USR_001 | SYS_001 | safety/mod.rs, grpc/services.rs |
| FR-INTLK-001 | USR_001, USR_HW_001 | SYS_002 | devices/gpio/mod.rs |
| FR-INTLK-002 | USR_HW_002 | SYS_002 | devices/gpio/mod.rs |
| FR-STATE-001 | USR_001 | SYS_003 | safety/mod.rs |
| FR-STATE-002 | USR_002 | SYS_004 | calibration.rs, safety/mod.rs |
| FR-DEV-001 | USR_HW_001 | SYS_011 | grpc/services.rs (DeviceInit) |
| FR-DEV-002 | USR_HW_001 | SYS_011 | grpc/services.rs (DeviceInit) |
| FR-DEV-003 | USR_HW_001 | - | devices/pdu/mod.rs |
| FR-CAL-001 | USR_002 | SYS_004 | calibration.rs, calibration_qc.rs |
| FR-CAL-002 | USR_002 | SYS_004 | calibration.rs |
| FR-CAL-003 | USR_010 | SYS_017 | calibration.rs |
| FR-MEAS-001 | USR_005 | SYS_005, SYS_012 | grpc/services.rs (Acquisition) |
| FR-MEAS-002 | USR_005 | SYS_005 | grpc/services.rs (Acquisition) |
| FR-MEAS-003 | USR_005 | SYS_005 | grpc/services.rs (Acquisition) |
| FR-MEAS-004 | USR_006 | SYS_013 | grpc/services.rs (GetExposureResults) |
| FR-MOT-001 | USR_MOTION_001 | SYS_020 | grpc/services.rs (Motion) |
| FR-MOT-002 | USR_MOTION_001 | SYS_020 | grpc/services.rs (Motion) |
| FR-MOT-003 | USR_MOTION_001 | SYS_020 | grpc/services.rs (Motion) |
| FR-DATA-001 | USR_007 | SYS_006, SYS_014 | measurement/mod.rs, audit/mod.rs |
| FR-DATA-002 | USR_009 | SYS_016 | measurement/mod.rs |
| FR-DATA-003 | USR_012 | SYS_025 | - (future) |
| FR-AUDIT-001 | USR_021 | SYS_007, SYS_024 | audit/mod.rs |
| FR-AUDIT-002 | USR_021 | SYS_007 | safety/mod.rs |
| FR-AUDIT-003 | USR_021 | SYS_024 | orchestrator/audit.py |
| FR-SEC-001 | USR_016 | SYS_009, SYS_021 | orchestrator/auth_manager.py |
| FR-SEC-002 | USR_004, USR_016 | SYS_009 | auth.rs, certificates.rs |
| FR-SEC-003 | USR_008 | SYS_015 | encryption.rs |
| FR-COMM-001 | USR_008 (implied) | SYS_008 | grpc/services.rs |
| FR-COMM-002 | Architecture | - | grpc/state_monitor_service.rs |
| FR-COMM-003 | Architecture | - | orchestrator/websocket |
| FR-UI-001 | USR_011 | SYS_022 | ui/pages/Dashboard.tsx |
| FR-UI-002 | USR_011 | SYS_022 | ui/pages/MeasurementPage.tsx |
| FR-UI-003 | USR_002 | SYS_004 | ui/pages/CalibrationPage.tsx |
| FR-UI-004 | USR_017 | SYS_022 | ui/services/errors.ts |
| FR-ERR-001 | USR_017, USR_018 | SYS_022 | errors.rs |
| FR-ERR-002 | USR_013 | SYS_018 | safety/mod.rs |
| FR-CFG-001 | Architecture | - | config.rs |
| FR-CFG-002 | USR_019 (implied) | SYS_023 | api/maintenance.rs |
| FR-HEALTH-001 | USR_013 | - | api/mod.rs |
| FR-HEALTH-002 | Architecture | - | grpc/services.rs (Health) |
| FR-HEALTH-003 | Architecture | - | grpc/services.rs (Health) |

---

## Appendix B: Functional Requirements Coverage

**Total Functional Requirements:** 45

**By Priority:**
- CRITICAL: 19 requirements
- HIGH: 19 requirements
- MEDIUM: 7 requirements

**By Subsystem:**
- Hardware Server (Rust): 32 requirements
- Orchestrator (Python): 6 requirements
- User Interface (React): 4 requirements
- Cross-cutting: 3 requirements

**Implementation Status:**
- Implemented: 42 requirements
- Future/Partial: 3 requirements

---

## Document Control

**Revision History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-17 | Omniscan Development Team | Initial functional requirements derived from user requirements and code analysis |

**Review and Approval:**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Software Development Lead | | | |
| Quality Assurance | | | |
| Regulatory Affairs | | | |

**Next Review:** Prior to FDA Design History File submission

**Document Classification:** FDA DHF - Functional Requirements Phase

---

**End of Document**
