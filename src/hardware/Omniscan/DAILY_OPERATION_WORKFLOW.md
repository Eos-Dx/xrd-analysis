# Omniscan Daily Operation Workflow

**Medical Device Software - FDA Compliant**  
**Document Version**: 1.0  
**Last Updated**: 2025-10-25

## Overview

This document describes the complete daily operational workflow for the Omniscan X-ray diffraction medical diagnostic device, from power-on to patient measurements. It defines the sequence of operations, safety interlocks, LED indicators, and data management procedures.

---

## System Components

### Hardware Components
- **X-ray Source**: Generates X-rays for diffraction analysis
- **Power Supply**: Controls power delivery to X-ray source and equipment
- **Detector**: Captures X-ray diffraction patterns
- **Motion Control System**: XY-stage for sample positioning
- **Key Switch**: Physical security control
- **LED Indicators**: Multi-color status indicators
- **Beam Block**: Physical beam safety interlock
- **Watchdog Systems**: Safety monitoring subsystems
- **Sound Indicators**: Audio feedback for operations

### Software Components
- **Hardware Server** (Rust): Safety-critical control authority at `C:\dev\Omniscan\omniscan-hw-server\`
- **Orchestrator** (Python): Workflow management and UI interface at `C:\dev\Omniscan\omniscan-orchestrator\`
- **Local Database**: SQLite storage for audit logs and measurements
- **Cloud Synchronization**: Optional cloud data upload

---

## Daily Operation Sequence

### Phase 1: System Power-On and Initialization

#### Step 1.1: Key Switch Activation
**Action**: Clinician arrives and turns the key switch to "ON" position

**Server Response**:
- Detects key switch state change via GPIO monitoring
- Transitions from `LOCKED` state to initialization
- Unblocks server authentication
- **LED Status**: Changes from **RED** → **ORANGE**
- Logs event: `KEY_SWITCH_ACTIVATED` with timestamp and operator context

**Safety Checks**:
- ✅ Key switch must be in "ON" position
- ✅ No emergency stop condition active
- ✅ All hardware watchdogs operational

**Implementation Notes**:
- Server GPIO monitoring (continuous polling at 100ms intervals)
- State machine transition: `LOCKED` → `INITIALIZED`
- Audit log entry with cryptographic integrity

---

#### Step 1.2: Clinician Authentication
**Action**: Clinician logs in through orchestrator UI

**Process**:
1. Clinician enters credentials via orchestrator interface
2. Orchestrator authenticates user and establishes session
3. Orchestrator sends authentication request to hardware server via gRPC
4. Server verifies key switch state before accepting login

**Server Response**:
- If key switch is OFF: **REJECT** with error code `ERR_KEY_SWITCH_OFF`
- If key switch is ON: **ACCEPT** and create authenticated session
- Logs session creation with user ID and role

**UI Behavior**:
- **Success**: Display device dashboard and power-on controls
- **Failure (key not turned)**: Display error message: "Device locked - turn key to operate"

**Implementation Notes**:
- Server endpoint: `POST /api/v1/auth/login` or gRPC `AuthService.Login`
- Session token generation (JWT or session ID)
- Audit log: `USER_LOGIN_SUCCESS` or `USER_LOGIN_FAILED`

---

### Phase 2: System Startup and Initialization

#### Step 2.1: Power-On Sequence Initiation
**Action**: Clinician clicks "Activate" button in orchestrator UI

**Requirements**:
- ✅ Key switch must be in "ON" position
- ✅ Clinician authenticated and logged in
- ✅ All safety interlocks satisfied
- ⏱️ Button must be clicked within **20 seconds** of prompt

**Server Response**:
1. Starts power-on sequence timer (20 second countdown)
2. Waits for physical activation button confirmation
3. If timeout: Returns to idle state and requires re-initiation

**Implementation Notes**:
- Server state: `IDLE` → `PENDING_ARMED`
- Orchestrator displays countdown timer
- Server endpoint: `POST /api/v1/device/activate` or gRPC `DeviceControl.Activate`

---

#### Step 2.2: Physical Confirmation and System Startup
**Action**: Clinician presses physical enable button on device within timeout period

**Server Response** (Parallel Execution):

| Subsystem | Action | Status Indicator |
|-----------|--------|------------------|
| **Power Supply** | Turn on power to X-ray source | Progress: 0-100% |
| **Detector** | Initialize and calibrate detector | Status: Initializing |
| **Motion Control** | Activate motors and execute homing sequence | Position: Homing → Ready |
| **Watchdogs** | Verify all safety watchdog systems | Status: All OK |
| **Beam Block** | Verify beam block presence and position | Status: Engaged |
| **Sound System** | Play startup sound (audible confirmation) | Sound: Active |

**Progress Reporting**:
- Server sends real-time progress updates to orchestrator via gRPC streaming
- Orchestrator displays progress bar and subsystem status
- Each subsystem reports completion percentage

**LED Indicators**:
- Main Status LED: **ORANGE** (system starting)
- Radiation Indicator LED: **GREEN** → **ORANGE** (Caution: Ionizing Radiation)

**Completion Criteria**:
- ✅ All subsystems initialized successfully
- ✅ All watchdogs reporting OK
- ✅ Motion control homed and ready
- ✅ Detector powered and calibrated
- ✅ Beam block verified in place

**Final State**:
- Main Status LED: **GREEN** (system ready)
- Radiation Indicator LED: **ORANGE** (caution - equipment energized)
- Server state: `IDLE` (ready for operations)
- Log event: `SYSTEM_STARTUP_COMPLETE`

**Implementation Notes**:
- Parallel execution using async tasks (Rust Tokio)
- gRPC streaming: `DeviceControl.StreamStartupProgress`
- Timeout for full startup: 60 seconds maximum
- Automatic abort if any subsystem fails

---

### Phase 3: X-ray Source Heat-Up

#### Step 3.1: Power Source Warm-Up
**Purpose**: Allow X-ray source to reach nominal operating temperature and current

**Current Implementation** (Stub):
- **Heat-up Duration**: 10 minutes (fixed timer)
- **Status**: Waiting for nominal conditions

**Future Implementation** (With Watchdogs):
- Monitor X-ray source temperature (thermal sensor)
- Monitor X-ray source current (power monitoring)
- Dynamic completion based on actual measurements

**Target Parameters**:
- **Nominal Current**: [To be defined based on hardware specifications]
- **Nominal Temperature**: [To be defined based on hardware specifications]
- **Stability Window**: Parameters must remain stable for 30 seconds

**Server Behavior**:
- Enters `WARMING_UP` state
- Periodically reports heat-up progress (every 10 seconds)
- Blocks measurement operations until complete
- Logs temperature and current readings

**Orchestrator UI**:
- Display heat-up timer: "X-ray source warming up: X:XX remaining"
- Show real-time temperature and current graphs (future)
- Indicate when system is ready for calibration

**Implementation Notes**:
- Server state: `WARMING_UP` → `IDLE` (after 10 minutes)
- gRPC endpoint: `DeviceControl.GetWarmupStatus`
- Future: Temperature and current thresholds in configuration

---

### Phase 4: Daily Calibration

#### Step 4.1: Calibrant Measurement
**Action**: Clinician selects calibrant sample and initiates calibration measurement

**Process**:
1. Clinician places calibrant sample in measurement position
2. Orchestrator sends calibration request to server
3. Server verifies system is ready for calibration
4. Server executes calibration exposure
5. Server captures diffraction pattern from detector
6. Orchestrator performs quality control analysis

**Server Responsibilities**:
- Execute calibration exposure (standardized parameters)
- Capture diffraction data from detector
- Store raw calibration data with timestamp
- Assign unique measurement ID
- Log calibration event

**Orchestrator Responsibilities**:
- Perform quality control analysis on calibration data
- Validate peak positions and intensities
- Determine pass/fail status
- Store calibration results in local database
- Update system calibration status

**Quality Control** (Current Stub):
- **Status**: All calibration measurements automatically PASS
- **Future**: Implement real QC procedures:
  - Peak position validation
  - Intensity threshold checks
  - Standard deviation analysis
  - Comparison with reference patterns

**Calibration Success**:
- Server updates `last_calibration_time` (24-hour validity)
- System transitions to `CALIBRATED` state
- Patient measurement window becomes available
- **LED Status**: Remains **GREEN**
- Log event: `CALIBRATION_PASSED`

**Calibration Failure** (Future):
- System remains in `LOCKED` state
- Patient measurements blocked
- Display error and required corrective actions
- Log event: `CALIBRATION_FAILED` with reason

**Implementation Notes**:
- Server endpoint: `POST /api/v1/calibration/measure` or gRPC `Acquisition.CalibrateDetector`
- Quality control in orchestrator Python code
- 24-hour calibration validity enforced by server
- Measurement data stored with UUID identifier

---

#### Step 4.2: Cloud Synchronization (Optional)
**Action**: Upload calibration data to cloud storage (if internet available)

**Process**:
1. Orchestrator checks internet connectivity
2. If available, upload calibration data to cloud
3. If unavailable, queue for later upload
4. Update local database with upload status

**Data Uploaded**:
- Diffraction pattern (raw data)
- Quality control results
- Calibration metadata (timestamp, operator, device ID)
- Unique measurement ID

**Security Requirements**:
- TLS-encrypted transmission
- Authentication token required
- No patient identifiable information (PII) in calibration data

**Implementation Notes**:
- Orchestrator handles cloud communication
- Asynchronous upload (non-blocking)
- Retry mechanism for failed uploads
- Local storage retained even after successful upload

---

### Phase 5: Patient Measurements

#### Step 5.1: Patient Measurement Authorization
**Action**: Clinician selects patient and initiates measurement

**Prerequisites**:
- ✅ Valid calibration within 24 hours
- ✅ System in `CALIBRATED` or `IDLE` state
- ✅ All safety interlocks satisfied
- ✅ Key switch in "ON" position

**Authorization Check**:
- Server verifies calibration validity
- Server checks safety interlocks
- If all checks pass: Enable measurement

**Data Privacy**:
- **Orchestrator**: Stores patient name, demographics, and metadata
- **Server**: Receives only **unique measurement ID** (UUID)
- No patient identifiable information sent to server

**Implementation Notes**:
- Server has no access to patient PII
- Complete data isolation between server and orchestrator
- HIPAA compliance: PII remains in orchestrator only

---

#### Step 5.2: Measurement Execution
**Action**: Execute X-ray diffraction measurement

**Process**:
1. Orchestrator sends measurement request with unique ID (no patient info)
2. Server transitions to `PENDING_ARMED` state
3. Clinician presses physical enable button (20-second timeout)
4. Server transitions to `RUNNING` state
5. **Radiation LED**: **ORANGE** → **RED** (active radiation)
6. Server controls X-ray exposure, detector capture, motion if needed
7. Server captures diffraction pattern from detector
8. Server stores measurement data with unique ID
9. Server returns measurement data to orchestrator
10. **Radiation LED**: **RED** → **ORANGE**
11. Server transitions to `IDLE` state

**Safety During Measurement**:
- Continuous interlock monitoring
- Beam watchdog active
- Emergency stop available at all times
- Automatic abort on any interlock violation
- **Sound**: Active radiation warning (audible alert)

**Data Handling**:
- Server: Stores measurement data with UUID only
- Orchestrator: Links UUID to patient metadata
- Both store complete audit trail

**Implementation Notes**:
- Server endpoint: `POST /api/v1/measurement/execute` or gRPC `Acquisition.StartExposure`
- Real-time streaming of measurement progress
- Timeout for enable button: 20 seconds
- Maximum exposure time: [To be defined]

---

#### Step 5.3: Data Storage and Cloud Upload
**Action**: Store measurement results and upload to cloud

**Local Storage**:

| Location | Data Stored | Retention Period |
|----------|-------------|------------------|
| **Server** | Raw diffraction data, measurement ID, timestamp, operator ID | **1 year minimum** |
| **Orchestrator** | Patient metadata, measurement results, QC data | **Per regulatory requirements** |

**Cloud Upload** (If Internet Available):
1. Orchestrator uploads measurement data to cloud
2. Server data remains local (not uploaded directly)
3. Orchestrator links patient metadata with cloud record
4. Update upload status in local database

**Data Integrity**:
- Server: Cryptographic checksums for all stored data
- Orchestrator: Database integrity checks
- Both: Append-only audit logs

**Critical Error Condition**:
- If server data goes missing or corrupted
- Server must issue **CRITICAL ERROR** to orchestrator
- Display error message: "Data integrity violation - contact service"
- Block further operations until resolved
- Log critical event with maximum severity

**Implementation Notes**:
- Server SQLite database at specified path
- Orchestrator SQLite database (separate)
- Automated data retention enforcement (1 year on server)
- Daily integrity checks on server database

---

### Phase 6: System Shutdown

#### Step 6.1: Normal Shutdown
**Action**: Clinician completes work and initiates shutdown

**Process**:
1. Clinician logs out of orchestrator
2. Orchestrator sends shutdown request to server
3. Server executes controlled shutdown:
   - Stop all active operations
   - Safe X-ray source (no emission)
   - Power down detector
   - Disable motion control
   - Write final audit log entries
   - Close database connections
4. Clinician turns key switch to "OFF" position
5. **LED Status**: All LEDs → **RED** (system locked)

**Audit Logging**:
- Log: `USER_LOGOUT` with session duration
- Log: `SYSTEM_SHUTDOWN_NORMAL` with uptime
- Log: `KEY_SWITCH_DEACTIVATED`

**Implementation Notes**:
- Graceful shutdown with timeout (30 seconds)
- Force shutdown if timeout exceeded
- All data committed to database before shutdown

---

#### Step 6.2: Critical Shutdown Events
**Definition**: Any unplanned server shutdown or termination

**Examples**:
- Power failure
- System crash
- Emergency stop activation
- Critical software error

**Server Response** (On Restart):
- Detect unplanned shutdown from audit log
- Create critical event entry: `CRITICAL_SHUTDOWN_DETECTED`
- Timestamp gap analysis (last log entry vs current time)
- Flag for investigation and review

**Orchestrator Response**:
- Display critical error notification
- Require administrator review before resuming operations
- Generate incident report

**Compliance Requirement**:
- **All server shutdowns must be logged**
- Unplanned shutdowns flagged as critical
- Investigation required for root cause analysis
- FDA traceability for all critical events

**Implementation Notes**:
- Server startup checks for incomplete shutdown
- Audit log sequence number validation
- Cryptographic integrity verification on startup

---

## LED Status Indicators

### Main Status LED
| Color | Meaning | State |
|-------|---------|-------|
| **RED** | System Locked | Key switch OFF or error condition |
| **ORANGE** | System Initializing | Startup sequence in progress |
| **GREEN** | System Ready | Ready for operations |

### Ionizing Radiation Warning LED
| Color | Meaning | State |
|-------|---------|-------|
| **GREEN** | Safe | No radiation hazard |
| **ORANGE** | Caution | Equipment energized - potential for radiation |
| **RED** | Active Radiation | X-ray beam active - do not enter |

---

## Data Management and Compliance

### Server Database (SQLite)
**Location**: `[To be configured in server.toml]`

**Contents**:
- Audit logs (all commands, state transitions, errors)
- Measurement data (raw diffraction patterns)
- Measurement IDs (UUIDs only)
- Operator IDs (no names)
- Timestamps (UTC)
- Hardware events (interlocks, watchdogs)

**Retention Policy**:
- **Minimum**: 1 year
- **Automated**: Old data archived after retention period
- **Critical Error**: If data missing or corrupted

**Security**:
- AES-256-GCM encryption
- Append-only audit log
- Daily cryptographic signatures
- Tamper detection

---

### Orchestrator Database (SQLite)
**Location**: `[To be configured in orchestrator config]`

**Contents**:
- Patient metadata (name, demographics)
- Measurement results (processed data)
- Quality control data
- Measurement ID to patient mapping
- Cloud upload status
- User sessions and authentication logs

**Retention Policy**:
- Per regulatory and institutional requirements
- Typically: 7+ years for medical records

**Security**:
- Encryption at rest
- Access control (user authentication required)
- Audit trail for all data access

---

### Data Privacy and Separation

**Key Principle**: Server has **ZERO** access to patient identifiable information (PII)

| System | Has Access To | Does NOT Have Access To |
|--------|---------------|------------------------|
| **Server** | Measurement ID (UUID), operator ID, timestamp | Patient name, demographics, medical history |
| **Orchestrator** | Patient PII, measurement results, metadata | Direct hardware control |

**Rationale**:
- HIPAA compliance
- Defense-in-depth security
- Minimize attack surface
- Simplify FDA compliance

**Implementation**:
- Server stores measurements under UUID only
- Orchestrator maintains UUID ↔ patient mapping
- Cloud upload handled by orchestrator (not server)
- Server audit logs contain no PII

---

## Safety and Interlocks

### Required Interlocks (All Must Be Satisfied)
- ✅ Key switch in "ON" position
- ✅ Emergency stop NOT pressed
- ✅ Safety door closed (door interlock sensor)
- ✅ Beam block present and verified
- ✅ All watchdog systems operational
- ✅ Enable button pressed (for measurements only)

### Interlock Violation Response
1. **Immediate**: Disable X-ray beam (hardware cutoff)
2. **Server**: Transition to `SAFE` state
3. **Orchestrator**: Display safety violation error
4. **LED**: Main status → **RED**
5. **Audit**: Log interlock violation with details
6. **Recovery**: Requires interlock reset and operator acknowledgment

---

## Error Handling and Recovery

### Error Categories

#### Category 1: Safety-Critical Errors
- Interlock violations
- Watchdog failures
- Emergency stop activation
- Beam intensity exceeded

**Response**: Immediate shutdown, safe state, operator notification

---

#### Category 2: Operational Errors
- Calibration expired
- Communication timeout
- Measurement timeout
- Device not responding

**Response**: Block operation, display error, allow retry or recovery

---

#### Category 3: Data Errors
- Database corruption
- Data integrity failure
- Upload failure
- Storage full

**Response**: Critical error notification, block operations, require service

---

### Recovery Procedures
1. **Interlock Reset**: Acknowledge error, verify conditions, reset via UI
2. **Calibration Expired**: Execute new calibration measurement
3. **Communication Lost**: Re-establish connection, verify system state
4. **Data Integrity**: Contact service, restore from backup if needed

---

## Implementation Requirements

### Server (Rust) Requirements
1. GPIO monitoring for key switch state (100ms polling)
2. State machine implementation (`LOCKED`, `INITIALIZED`, `IDLE`, `PENDING_ARMED`, `RUNNING`, `WARMING_UP`, `SAFE`, `CALIBRATED`)
3. LED control via GPIO or hardware API
4. Sound generation for startup and warnings
5. Parallel subsystem initialization (Tokio async)
6. Real-time progress streaming (gRPC)
7. Calibration timestamp tracking and 24-hour enforcement
8. Measurement data storage with UUID only
9. Audit logging with cryptographic integrity
10. Critical shutdown detection on startup
11. 1-year data retention enforcement

### Orchestrator (Python) Requirements
1. User authentication and session management
2. Key switch status verification before login
3. Activate button UI with 20-second countdown
4. Startup progress display (progress bars per subsystem)
5. Heat-up timer display
6. Calibration quality control analysis
7. Patient metadata management (local only)
8. Measurement request with UUID generation
9. Cloud upload with internet connectivity check
10. Database storage with patient data
11. Error display and user notifications

---

## Future Enhancements

### Phase 1: Hardware Watchdogs (Short-term)
- Real-time X-ray source temperature monitoring
- Real-time X-ray source current monitoring
- Dynamic heat-up completion (replace 10-minute timer)
- Over-temperature protection and shutdown

### Phase 2: Calibration Quality Control (Medium-term)
- Automated peak detection and analysis
- Reference pattern comparison
- Pass/fail criteria based on standards
- Trend analysis for calibration drift

### Phase 3: Advanced Safety Features (Long-term)
- Redundant beam watchdog systems
- Predictive maintenance alerts
- Automatic incident reporting
- Advanced error recovery procedures

---

## Compliance Notes

### FDA Requirements
- Complete audit trail (both server and orchestrator)
- Safety interlock enforcement
- Data integrity and retention
- User authentication and authorization
- Incident reporting for critical events

### HIPAA Requirements
- Patient data isolation (server has no PII)
- Encrypted data storage
- Access controls and audit logs
- Secure data transmission

### IEC 62304 Requirements
- Software safety classification (Class B)
- Risk analysis and mitigation
- Software verification and validation
- Change control procedures

---

**Document End**

*This workflow represents the complete daily operation of the Omniscan medical device system and serves as the specification for software development.*
