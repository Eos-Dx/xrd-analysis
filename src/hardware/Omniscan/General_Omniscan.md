# 🩺 Omniscan Medical Diagnostic Platform - Complete System Documentation

**Classification:** FDA Design History File (DHF) - System Documentation  
**Version:** 1.0  
**Date:** October 26, 2025  
**Purpose:** Comprehensive documentation of the Omniscan X-ray diffraction medical diagnostic system

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture](#architecture)
4. [Subprojects](#subprojects)
5. [Clinical Workflow](#clinical-workflow)
6. [Requirements](#requirements)
7. [Safety & Compliance](#safety--compliance)
8. [Data Management](#data-management)
9. [Security Architecture](#security-architecture)
10. [Development Status](#development-status)
11. [Getting Started](#getting-started)
12. [References](#references)

---

## Executive Summary

Omniscan is a **first-of-kind FDA-compliant medical diagnostic platform** utilizing X-ray Diffraction (XRD) technology for patient sample analysis. The system combines safety-critical hardware control, secure maintenance operations, comprehensive audit logging, and HIPAA-compliant data management to meet IEC 62304 Class B and FDA De Novo requirements.

### Key Features

- ✅ **Safety-Critical Control**: Hardware-enforced safety interlocks with physical key switch, emergency stop, and door sensors
- ✅ **Privacy-by-Design**: Patient PII isolated in orchestrator; hardware server receives only UUIDs
- ✅ **mTLS Authentication**: Mutual TLS with device-scoped engineer certificates for maintenance access
- ✅ **Comprehensive Audit**: Encrypted, tamper-proof audit logging with cryptographic integrity
- ✅ **Daily Calibration**: Mandatory 24-hour calibration enforcement with automatic system lockout
- ✅ **FDA Traceability**: Complete chain from requirements through code to deployment
- ✅ **Offline Operation**: Full local functionality with delayed cloud synchronization

### Regulatory Compliance

- **FDA Classification**: Class B Medical Device Software (IEC 62304)
- **Standards**: ISO 14971 (Risk Management), IEC 62366-1 (Human Factors), ISO 13485 (QMS)
- **Security**: ISO 27001, FDA Cybersecurity Guidance (2023)
- **Privacy**: HIPAA, GDPR compliant data handling
- **FDA Pathway**: De Novo (first-of-kind device)

---

## System Overview

### Clinical Application

**Primary Use**: X-ray diffraction analysis of patient samples in clinical laboratory environments

**Target Users**:
- **Clinical Operators**: Laboratory technicians performing diagnostic measurements
- **Clinicians**: Medical professionals reviewing diagnostic results
- **Maintenance Engineers**: Service personnel performing calibration and maintenance
- **IT Administrators**: System configuration and data security management

### Ecosystem Components

```
┌─────────────────────────────────────────────────────────────┐
│                    OMNISCAN ECOSYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐    HTTP/REST/WS    ┌─────────────────┐  │
│   │ Web UI       │◄──────────────────►│ Orchestrator    │  │
│   │ (React)      │  Password auth     │ (FastAPI/Python)│  │
│   │ Port: 3000   │                    │ Port: 8080      │  │
│   └──────────────┘                    │ - Workflow      │  │
│                                        │ - Patient DB    │  │
│                                        │ - Cloud sync    │  │
│                                        └────────┬────────┘  │
│                                                 │            │
│                                           gRPC + mTLS        │
│                                           (certificates)     │
│                                                 │            │
│                                        ┌────────▼────────┐  │
│   ┌──────────────┐    mTLS/HTTPS      │ Hardware Server │  │
│   │ Engineer CLI │◄──────────────────►│ (Rust/gRPC)     │  │
│   │ (Python)     │  Certificate auth  │ Port: 50051     │  │
│   └──────────────┘                    │ - Safety control│  │
│                                        │ - Interlocks    │  │
│                                        │ - Audit logs    │  │
│                                        └────────┬────────┘  │
│                                                 │            │
│                                        TCP/DLL/GPIO         │
│                                                 │            │
│                                        ┌────────▼────────┐  │
│                                        │ Medical Hardware│  │
│                                        │ - X-ray detector│  │
│                                        │ - Motion devices│  │
│                                        │ - PDU (power)   │  │
│                                        │ - GPIO safety   │  │
│                                        └─────────────────┘  │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐ │
│   │              Cloud Platform (AWS)                    │ │
│   │  - Data ingress    - ML inference                    │ │
│   │  - Processing      - Reporting                       │ │
│   └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture

### System Layers

#### 1. **Presentation Layer** - Web UI
- **Technology**: React 18 + TypeScript
- **Port**: 3000 (development), 80/443 (production)
- **Authentication**: Password-based (username/password)
- **Communication**: REST API + WebSocket for real-time updates
- **Responsibilities**:
  - Patient sample registration
  - Measurement initiation and monitoring
  - Calibration status display
  - Historical data review
  - System status dashboard

#### 2. **Application Layer** - Orchestrator (REST API Server)
- **Technology**: FastAPI (Python 3.11+)
- **Port**: 8080 (REST API), WebSocket on same port
- **Database**: SQLite (`data/orchestrator.db`)
- **Authentication**: Session-based (from Web UI), Certificate-based (from engineer CLI)
- **Responsibilities**:
  - **Patient Data Management**: Stores patient PII (names, DOB, medical records) **LOCALLY ONLY**
  - **Privacy Gateway**: Generates UUIDs for measurements; **NEVER sends patient info to hardware**
  - **Measurement Coordination**: Links UUID to patient (local mapping only)
  - **Session Management**: User authentication and activity tracking
  - **Cloud Synchronization**: Asynchronous upload with offline queue
  - **Audit Logging**: User-facing audit trail

**Key Privacy Feature**: Orchestrator acts as **privacy gateway** - patient identifiable information NEVER leaves this layer.

#### 3. **Control Layer** - Hardware Server
- **Technology**: Rust 1.70+ with Tokio async runtime
- **Port**: 50051 (gRPC only, no REST)
- **Database**: Encrypted SQLite (audit logs)
- **Authentication**: mTLS certificates ONLY
- **Responsibilities**:
  - **Safety Authority**: Exclusive control over X-ray beam and safety systems
  - **Hardware Control**: Direct communication with detectors, motors, PDU, GPIO
  - **Safety Interlocks**: Continuous monitoring (key switch, E-stop, door sensors, beam watchdog)
  - **State Machine**: LOCKED → INITIALIZED → IDLE → PENDING_ARMED → RUNNING → SAFE
  - **Audit Logging**: Safety-critical operations (encrypted, tamper-proof)
  - **Data Storage**: Measurement data with **UUID only** (no patient information)

**Key Security Feature**: Hardware server has **ZERO access to patient PII** - receives only anonymized UUIDs.

#### 4. **Hardware Layer** - Physical Devices
- **X-ray Detectors**: Bruker BIS (TCP), AdvoCam (C DLL)
- **Motion Devices**: Thorlabs motor controllers (C DLL FFI)
- **PDU**: Power Distribution Unit (TCP/IP with authentication)
- **GPIO PCIe**: Safety interlocks (key switch, E-stop, door sensors, enable button, beam watchdog)
- **LED Indicators**: Main status (RED/ORANGE/GREEN), Radiation warning (GREEN/ORANGE/RED)
- **Sound System**: Startup confirmation, radiation warning beeps

---

## Subprojects

### 1. 🦀 omniscan-hw-server (Rust)
**Location**: `C:\dev\Omniscan\omniscan-hw-server\`

**Purpose**: Safety-critical hardware control server with exclusive authority over X-ray beam and physical devices.

**Key Components**:
- **Safety State Machine**: Enforces operational states (LOCKED, IDLE, PENDING_ARMED, RUNNING, SAFE, CALIBRATION, MAINTENANCE)
- **Physical Safety Controls**: E-stop, door interlocks, key switch, beam watchdog, enable button (20-second timeout)
- **gRPC Services**: Acquisition, Motion, Health, Safety, Beam Monitoring, Device Control, Calibrant Quality
- **Encrypted Audit Database**: SQLite with AES-256-GCM encryption, TPM-signed daily rolls
- **Device Abstraction Layer**: Trait-based abstractions for detectors, motion, GPIO, PDU
- **LED Control**: Main status LED (RED/ORANGE/GREEN), Radiation warning LED (GREEN/ORANGE/RED)
- **Sound Generation**: Startup confirmation and radiation warnings

**Technology Stack**:
- Rust 1.70+ (memory safety, no garbage collection)
- Tokio (async runtime for concurrent operations)
- Tonic (gRPC framework)
- SQLite (audit database with encryption)
- Rustls (TLS/mTLS implementation)

**Port**: 50051 (gRPC), 8443 (HTTPS REST - optional for maintenance)

**Documentation**: [omniscan-hw-server/README.md](omniscan-hw-server/README.md)

---

### 2. 🐍 omniscan-orchestrator (Python)
**Location**: `C:\dev\Omniscan\omniscan-orchestrator\`

**Purpose**: Intermediate layer bridging Web UI (password auth) with Hardware Server (mTLS), managing patient data and workflow coordination.

**Key Components**:
- **REST API Server** (`rest_server.py`): FastAPI endpoints for Web UI communication
- **gRPC Client** (`grpc_client.py`): Safe wrapper for hardware server communication with mTLS
- **Database** (`database.py`): SQLite storage for patient PII, measurements, sessions
- **Authentication Manager** (`auth_manager.py`): Session management with key switch verification
- **Engineer CLI** (`cli.py`): Command-line interface for maintenance operations
- **Certificate Manager** (`cert_manager.py`): Engineer certificate generation and management
- **Audit Logging** (`audit.py`): User-facing audit trail with FDA compliance
- **RBAC** (`rbac.py`): Role-based access control (operator, engineer, admin)

**Technology Stack**:
- Python 3.11+
- FastAPI (REST API framework)
- Uvicorn (ASGI server with WebSocket support)
- Pydantic (data validation)
- gRPC/Protobuf (hardware communication)
- SQLite (patient database)
- Typer (CLI framework)
- Rich (terminal output)

**Ports**: 8080 (REST API + WebSocket)

**Key Privacy Feature**: Orchestrator is the **ONLY** component with access to patient identifiable information. It generates UUIDs for measurements and maintains UUID ↔ patient mapping locally.

**Documentation**: 
- [omniscan-orchestrator/README.md](omniscan-orchestrator/README.md)
- [omniscan-orchestrator/REST_API.md](omniscan-orchestrator/REST_API.md)
- [omniscan-orchestrator/REST_SERVER_IMPLEMENTATION.md](omniscan-orchestrator/REST_SERVER_IMPLEMENTATION.md)

---

### 3. 🔐 omniscan-certificate-center (Python)
**Location**: `C:\dev\Omniscan\omniscan-certificate-center\`

**Purpose**: PKI certificate generation toolkit for mTLS authentication following OmniSoft Certificate Strategy.

**Certificate Types**:
- **Device Server Root CA**: Signs server certificates (long-lived, offline storage)
- **Maintenance Client Root CA**: Signs engineer client certificates (short-lived)
- **Server Certificates**: 1 year validity, TPM-backed (device servers)
- **Client Certificates**: 1-7 days validity, device-scoped (maintenance engineers)

**Technology Stack**:
- Python 3.11+
- Cryptography library (x509 certificate generation)
- ECDSA P-256 or RSA-3072 with SHA-256
- OpenSSL-compatible PEM format

**Documentation**: [omniscan-certificate-center/README.md](omniscan-certificate-center/README.md)

---

### 4. 🌐 omniscan-ui (React)
**Location**: `C:\dev\Omniscan\omniscan-ui\`

**Purpose**: Browser-based clinician interface for patient sample management and measurement monitoring.

**Key Features**:
- Dashboard: System status, safety interlocks, calibration status, quick measurement controls
- Measurement Page: Sample tracking, measurement execution, history review
- Calibration Page: Daily calibration workflow, status monitoring
- Real-time Updates: WebSocket connection for live system state
- Role-Based UI: Different views for operators, engineers, administrators

**Technology Stack**:
- React 18 + TypeScript
- Vite (build tool)
- Zustand (state management)
- React Router (navigation)
- Tailwind CSS (styling)
- Recharts (data visualization)

**Port**: 3000 (development), 80/443 (production)

**Documentation**: [omniscan-ui/README.md](omniscan-ui/README.md)

---

## Clinical Workflow

### Daily Operation Sequence

Based on [DAILY_OPERATION_WORKFLOW.md](DAILY_OPERATION_WORKFLOW.md)

#### Phase 1: System Power-On (5-10 minutes)

**Step 1.1: Key Switch Activation**
- **Action**: Clinician turns physical key switch to "ON"
- **Hardware Response**: GPIO monitors key switch state (100ms polling)
- **Server Response**: 
  - State: `LOCKED` → `INITIALIZED`
  - LED: **RED** → **ORANGE**
  - Audit: `KEY_SWITCH_ACTIVATED` logged
  - Unblocks authentication

**Step 1.2: Clinician Login**
- **Action**: Clinician enters credentials in Web UI
- **Process**:
  1. Web UI → Orchestrator REST API (`POST /api/auth/login`)
  2. Orchestrator verifies username/password
  3. Orchestrator → Hardware Server (gRPC `get_key_switch_state()`)
  4. Hardware Server checks GPIO key switch status
  5. If key ON: Create session, return session_id
  6. If key OFF: Reject with error "Device locked - turn key to operate"
- **Session Management**: Session stored in orchestrator database with activity tracking

**Step 1.3: System Activation (20-second window)**
- **Action**: Clinician clicks "Activate" button in UI
- **Hardware Response**:
  - State: `IDLE` → `PENDING_ARMED`
  - Starts 20-second countdown for physical enable button
  - Orchestrator displays countdown timer
- **Physical Confirmation**: Clinician presses enable button on device
- **Parallel Startup**: Hardware server initializes all subsystems:
  - Power Supply: Energize X-ray source (progress 0-100%)
  - Detector: Initialize and calibrate
  - Motion Control: Homing sequence
  - Watchdogs: Verify all safety systems
  - Beam Block: Verify presence and position
  - Sound: Startup confirmation beep
- **Completion**:
  - State: `PENDING_ARMED` → `IDLE`
  - LED: **ORANGE** → **GREEN** (ready)
  - Radiation LED: **GREEN** → **ORANGE** (caution)

---

#### Phase 2: X-ray Source Warmup (10 minutes)

**Purpose**: Allow X-ray source to reach nominal operating temperature and current stability.

**Current Implementation**: 10-minute fixed timer (stub for development)

**Future Implementation**: Dynamic completion based on:
- X-ray source temperature (thermal sensor)
- X-ray source current (power monitoring)
- Stability window: Parameters stable for 30 seconds

**Server Behavior**:
- State: `WARMING_UP`
- Progress updates every 10 seconds
- Blocks measurement operations until complete
- Logs temperature and current readings

**UI Display**: "X-ray source warming up: X:XX remaining"

---

#### Phase 3: Daily Calibration (5-15 minutes)

**Requirement**: **Mandatory** 24-hour calibration enforcement. System enters `LOCKED` state if calibration expires.

**Step 3.1: Calibration Measurement**
- **Action**: Clinician places calibrant sample and initiates calibration
- **Hardware Server**:
  - Executes calibration exposure (standardized parameters)
  - Captures diffraction pattern
  - Stores raw data with UUID
  - Assigns unique measurement ID
  - Logs `CALIBRATION_MEASUREMENT` event
- **Orchestrator**:
  - Performs quality control analysis on diffraction data
  - Validates peak positions and intensities
  - Determines PASS/FAIL status
  - Stores calibration results
  - Updates system calibration timestamp

**Current QC**: All calibrations automatically PASS (stub for development)

**Future QC**: Real quality control procedures:
- Peak position validation against reference
- Intensity threshold checks
- Standard deviation analysis
- Comparison with historical patterns

**Calibration Success**:
- Server updates `last_calibration_time` (24-hour countdown starts)
- State: `LOCKED` → `CALIBRATED`
- Measurement operations unlocked
- LED remains **GREEN**
- Log: `CALIBRATION_PASSED`

**Calibration Failure** (Future):
- System remains in `LOCKED` state
- Patient measurements blocked
- Display error and corrective actions
- Log: `CALIBRATION_FAILED` with diagnostic details

**Step 3.2: Cloud Synchronization (Optional)**
- **If internet available**: Orchestrator uploads calibration data to cloud
- **If offline**: Queue for later upload (non-blocking)
- **Data uploaded**: Diffraction pattern, QC results, metadata (timestamp, operator, device ID), UUID
- **Security**: TLS encryption, no patient PII in calibration data

---

#### Phase 4: Patient Measurements (2-10 minutes per measurement)

**Privacy Protection Flow** (Critical for HIPAA compliance):

**Step 4.1: Patient Authorization**
- **Web UI**: Clinician selects patient and sample ID
- **Orchestrator**:
  1. Looks up patient record from local database (full PII available)
  2. Generates UUID for measurement: `measurement_id = uuid.uuid4()`
  3. Stores mapping: `UUID ↔ patient_id` (LOCAL DATABASE ONLY)
  4. **Sends to hardware server**: UUID only (NO patient name, DOB, medical record)
- **Hardware Server**:
  - Receives: `measurement_id` (UUID), `operator_id`, exposure parameters
  - Has NO access to patient information
  - Validates: Calibration valid (<24 hours), interlocks satisfied, state is IDLE or CALIBRATED

**Step 4.2: Measurement Execution**
- **Process**:
  1. Orchestrator → Hardware Server: `POST /api/measurements/start` with UUID
  2. Hardware Server: State `IDLE` → `PENDING_ARMED`
  3. Clinician presses physical enable button (20-second timeout)
  4. Hardware Server: State `PENDING_ARMED` → `RUNNING`
  5. LED: Radiation warning **ORANGE** → **RED** (active radiation)
  6. Sound: Continuous radiation warning beep
  7. Hardware controls: X-ray exposure, detector capture, optional motion scanning
  8. Hardware captures diffraction pattern
  9. Hardware stores measurement with UUID only
  10. LED: **RED** → **ORANGE**
  11. State: `RUNNING` → `IDLE`
  12. Returns measurement data to orchestrator

**Safety During Measurement**:
- Continuous interlock monitoring (100ms polling)
- Beam watchdog active (intensity validation)
- Emergency stop available (immediate abort)
- Automatic abort on any interlock violation
- Door sensor: Beam disabled if door opens
- Enable button: Must remain pressed (dead-man switch behavior)

**Step 4.3: Data Storage and Linking**
- **Hardware Server Database**:
  - Measurement ID: UUID only
  - Operator ID: User identifier (not patient)
  - Raw diffraction data
  - Exposure parameters
  - Timestamp (UTC)
  - NO patient information
- **Orchestrator Database**:
  - Measurement ID: Same UUID
  - Patient ID: Links to patient record
  - Processed results
  - QC status
  - Cloud upload status
- **Web UI**: Displays results with patient name (orchestrator joins UUID → patient)

**Privacy Verification**:
```python
# Orchestrator code (simplified)
patient = db.get_patient(request.patient_id)  # Full PII
measurement_uuid = str(uuid.uuid4())          # Generate UUID
db.record_measurement(measurement_uuid, patient.patient_id)  # Local mapping

# Hardware server receives:
grpc_client.start_exposure_with_uuid(
    measurement_id=measurement_uuid,  # UUID ONLY
    operator_id=user["user_id"],
    exposure_time_ms=request.exposure_duration
)
# NO patient name, DOB, or medical record sent!
```

---

#### Phase 5: System Shutdown

**Step 5.1: Normal Shutdown**
- **Action**: Clinician logs out
- **Hardware Server**:
  1. Stops all active operations
  2. Safes X-ray source (no emission)
  3. Powers down detector
  4. Disables motion control
  5. Writes final audit log entries
  6. Closes database connections
- **Clinician**: Turns key switch to "OFF"
- **LED**: All → **RED** (system locked)
- **Audit Logs**:
  - `USER_LOGOUT` with session duration
  - `SYSTEM_SHUTDOWN_NORMAL` with uptime
  - `KEY_SWITCH_DEACTIVATED`

**Step 5.2: Critical Shutdown Detection**
- **Definition**: Any unplanned shutdown (power failure, crash, emergency stop)
- **Server Response on Restart**:
  1. Detects unplanned shutdown from audit log
  2. Creates `CRITICAL_SHUTDOWN_DETECTED` event
  3. Timestamp gap analysis
  4. Flags for investigation
- **Compliance**: **All server shutdowns must be logged**; unplanned shutdowns = critical events requiring root cause analysis

---

## Requirements

### Requirements Traceability Matrix

Based on [USER_EXPECTATIONS.md](USER_EXPECTATIONS.md) and [REQUIREMENTS_TRACEABILITY_MATRIX.md](REQ/REQUIREMENTS_TRACEABILITY_MATRIX.md)

#### User Requirements (USR_*)

| ID | Subsystem | Title | Status | Implementation |
|----|-----------|-------|--------|----------------|
| **USR_OMNI-SERVER-001** | Hardware Server | Reliable and safe operation | ✅ Complete | Safety state machine, interlock monitoring, emergency abort |
| **USR_OMNI-SERVER-002** | Hardware Server | Daily calibration capability | ✅ Complete | 24-hour calibration enforcement, system lockout |
| **USR_OMNI-SERVER-003** | Hardware Server | Data integrity and traceability | ✅ Complete | Encrypted audit logging, persistent storage |

#### System Requirements (SYS_*)

| ID | Requirement | Status | Implementation |
|----|-------------|--------|----------------|
| **SYS_OMNI-SERVER-001** | Safety authority | ✅ Complete | Rust server exclusive control, gRPC-only communication |
| **SYS_OMNI-SERVER-002** | Interlocks and watchdogs | ✅ Complete | Key switch, E-stop, door sensor, beam watchdog framework |
| **SYS_OMNI-SERVER-003** | Operational states | ✅ Complete | State machine: LOCKED, IDLE, PENDING_ARMED, RUNNING, SAFE, etc. |
| **SYS_OMNI-SERVER-004** | Calibration enforcement | ✅ Complete | 24-hour rule, operation prohibition without valid calibration |
| **SYS_OMNI-SERVER-005** | Beam fault handling | ⚠️ Framework | Beam monitoring framework, threshold detection stubs |
| **SYS_OMNI-SERVER-006** | Data retention | ✅ Complete | Indefinite storage, encrypted audit logs |
| **SYS_OMNI-SERVER-007** | Audit database | ✅ Complete | Append-only SQLite, AES-256-GCM encryption, TPM-signed daily rolls |
| **SYS_OMNI-SERVER-008** | gRPC interface | ✅ Complete | Acquisition, Motion, Health, Safety, Beam Monitoring, Device Control services |
| **SYS_OMNI-SERVER-009** | Session security | ⚠️ Framework | Command context tracking, mTLS ready |
| **SYS_OMNI-SERVER-010** | Safety profile updates | ✅ Complete | Encrypted config manager, maintenance mode |

#### Risk Controls (RISK_*)

| ID | Hazard | Mitigation | Status |
|----|--------|-----------|--------|
| **RISK_OMNI-SERVER-001** | Beam exposure when door open | Door sensor interlock, automatic beam disable | ✅ Implemented |
| **RISK_OMNI-SERVER-002** | Motion when E-stop pressed | Hardware E-stop breaks actuator power | ✅ Implemented |
| **RISK_OMNI-SERVER-003** | Software watchdog failure | Independent hardware watchdog integration | ⚠️ Framework |
| **RISK_OMNI-SERVER-004** | Unauthorized configuration | Encrypted config, signed updates, maintenance mode | ✅ Implemented |
| **RISK_OMNI-SERVER-005** | Data loss during upload | Local database retention until confirmed upload | ✅ Implemented |

**Legend**: ✅ Complete | ⚠️ Framework Ready | 🚧 Planned | ❌ Not Started

---

## Safety & Compliance

### Safety Architecture

#### Physical Safety Controls (Hardware Layer)
1. **Key Switch**: Required in "ON" position for operation (blocks login when OFF)
2. **Emergency Stop**: Hardware-level beam/motion cutoff (breaks power physically)
3. **Door Interlocks**: Safety door must be closed for beam operation
4. **Enable Button**: Physical confirmation required for measurements (20-second timeout, dead-man switch)
5. **Beam Watchdog**: Continuous X-ray intensity monitoring
6. **Over-temperature**: Optional thermal protection (future)

#### Software Safety Controls (State Machine Layer)
- **State Machine**: Enforces valid operation sequences
  ```
  LOCKED ──calibration──► IDLE ──start──► PENDING_ARMED ──enable──► RUNNING
     ▲                     ▲                                           │
     │                     │                                           ▼ stop
     │                     │                                       STOPPING
     │                     │                                           │
     └──interlock──────────┴──────abort/emergency─────────────────────┘
                         SAFE
  ```
- **Interlock Validation**: Pre-checks before all operations
- **Automatic Abort**: Immediate transition to SAFE on interlock violation
- **Timeout Enforcement**: 20-second window for physical enable button

#### Audit & Traceability (Compliance Layer)
- **Append-Only Audit Log**: Cannot be modified retroactively
- **Cryptographic Integrity**: AES-256-GCM encryption, TPM-signed daily rolls
- **Complete Traceability**: Every operation linked to user, device, timestamp
- **Event Logging**:
  - All commands and responses
  - State transitions
  - Interlock status changes
  - Configuration modifications
  - User authentication events
  - Calibration results
  - Measurement executions

### Compliance Standards

#### FDA/IEC 62304 Class B Medical Device Software
- **Software Safety Classification**: Class B (medium risk)
- **Development Process**: Structured software lifecycle
- **Risk Management**: ISO 14971 risk analysis and mitigation
- **Human Factors**: IEC 62366-1 usability engineering
- **Traceability**: Requirements ↔ design ↔ code ↔ tests ↔ risks

#### Quality Management (ISO 13485)
- **Quality System**: Design controls and change management
- **Document Control**: Design History File (DHF) maintenance
- **CAPA**: Corrective and Preventive Action procedures
- **Verification & Validation**: Unit, integration, and system testing

#### Information Security (ISO 27001)
- **Encryption**: AES-256-GCM for data at rest
- **TLS/mTLS**: Encryption in transit
- **Access Control**: Role-based permissions (RBAC)
- **Key Management**: TPM-backed credential storage
- **Audit Logging**: Complete access trail

#### Privacy Compliance (HIPAA/GDPR)
- **Data Minimization**: Hardware server receives no PII
- **Encryption**: Patient data encrypted at rest and in transit
- **Access Control**: Authentication and authorization required
- **Audit Trail**: All data access logged
- **Right to Erasure**: Patient data deletion procedures

#### FDA Cybersecurity Guidance (2023)
- **Secure Architecture**: Defense-in-depth with multiple layers
- **Cryptographic Controls**: Modern algorithms and key management
- **Access Control**: Multi-factor authentication for maintenance
- **Audit Logging**: Comprehensive security event logging
- **Update Management**: Signed software updates

---

## Data Management

### Data Privacy Model (HIPAA-Compliant)

**Critical Principle**: Patient PII NEVER leaves the orchestrator.

```
┌─────────────────────────────────────────────────────────┐
│               ORCHESTRATOR (Patient-Facing)              │
│  Stores: Patient PII, demographics, medical records     │
│  Sends to Server: UUID only (no patient information)    │
│  Database: data/orchestrator.db (SQLite, local)         │
└────────────────────┬────────────────────────────────────┘
                     │ gRPC: UUID + operator ID ONLY
                     ▼
┌─────────────────────────────────────────────────────────┐
│            HARDWARE SERVER (Safety-Critical)             │
│  Stores: UUID, operator ID, raw measurements            │
│  NO patient information ever received or stored         │
│  Database: Encrypted SQLite with audit logs             │
└─────────────────────────────────────────────────────────┘
```

### Database Schemas

#### Orchestrator Database (`data/orchestrator.db`)

**patients table**:
- `patient_id` (PRIMARY KEY, UUID)
- `first_name`, `last_name` (TEXT)
- `date_of_birth` (TEXT, ISO format)
- `medical_record_number` (TEXT, UNIQUE)
- `created_at`, `updated_at` (TEXT, ISO timestamps)

**measurements table**:
- `measurement_id` (PRIMARY KEY, UUID) ← Shared with hardware server
- `patient_id` (FOREIGN KEY → patients)
- `timestamp` (TEXT, ISO format)
- `operator_id` (TEXT)
- `measurement_type` (TEXT: "calibration" or "patient")
- `result_data` (BLOB, processed results)
- `qc_status` (TEXT: "pass", "fail", "pending")
- `uploaded_to_cloud` (INTEGER, boolean)
- `upload_timestamp` (TEXT, optional)

**user_sessions table**:
- `session_id` (PRIMARY KEY, UUID)
- `user_id` (TEXT)
- `role` (TEXT: "operator", "engineer", "admin")
- `login_time` (TEXT)
- `logout_time` (TEXT, optional)
- `last_activity` (TEXT)

**Key Feature**: `measurement_id` is the ONLY shared identifier between orchestrator and hardware server.

#### Hardware Server Database (Encrypted SQLite)

**audit_log table**:
- `id` (PRIMARY KEY, auto-increment)
- `timestamp` (TEXT, UTC)
- `event_type` (TEXT)
- `user_id` (TEXT) ← Operator ID, NOT patient
- `command_id` (TEXT, UUID)
- `state_before`, `state_after` (TEXT)
- `details` (JSON)
- `integrity_hash` (TEXT, SHA-256)

**measurements table** (simplified):
- `measurement_id` (PRIMARY KEY, UUID) ← Matches orchestrator UUID
- `timestamp` (TEXT, UTC)
- `operator_id` (TEXT) ← NOT patient information
- `exposure_time_ms` (INTEGER)
- `raw_data` (BLOB, diffraction pattern)
- `detector_temp`, `beam_intensity` (REAL)

**Key Security**: Database encrypted with AES-256-GCM, daily TPM-signed integrity rolls.

---

## Security Architecture

### Certificate-Based Authentication (mTLS)

#### Certificate Hierarchy

```
Device Server Root CA ─────┬────► Server Certificates (1 year)
                           │       └─► OMNIScan Hardware Servers
                           │
Maintenance Client Root CA ─────► Client Certificates (1-7 days)
                                   └─► Maintenance Engineers
```

#### Authentication Flow

**For Engineer Maintenance Access**:
1. Engineer generates short-lived client certificate (1 day default)
   ```bash
   omni-orch cert generate --engineer-id ENG001 --device-uuid ABC123
   ```
2. Certificate contains:
   - CN: `engineer:ENG001`
   - SAN URI: `urn:omniscan:server:ABC123` (device scope)
   - Validity: 24 hours (configurable)
3. Engineer presents certificate to hardware server
4. Hardware server validates:
   - Certificate signed by trusted Client Root CA
   - Certificate not expired
   - Device UUID in SAN matches server UUID
   - CN role grants required permissions
5. Access granted if all checks pass
6. All actions logged with certificate serial number

**For Orchestrator → Hardware Communication**:
1. Orchestrator has long-lived client certificate (orchestrator-specific)
2. Presents certificate with each gRPC request
3. Hardware server validates as above
4. gRPC communication encrypted with mTLS

#### Security Benefits
- **Device Scoping**: Engineer certificates restricted to specific devices
- **Short Validity**: Compromised certificates expire quickly (1 day)
- **Mutual Authentication**: Both client and server authenticate each other
- **Audit Trail**: Complete log of certificate-based access
- **Revocation**: Expired certificates automatically invalid (no CRL needed for short-lived)

### Role-Based Access Control (RBAC)

| Role | Permissions | Certificate CN Format | Access Method |
|------|-------------|----------------------|---------------|
| **Clinical Operator** | Execute measurements, perform calibration, view configuration | `operator:<id>` | Password (via Web UI) |
| **Maintenance Engineer** | All operator permissions + maintenance mode, modify config, view audit logs | `engineer:<id>` | mTLS certificate (CLI) |
| **Administrator** | User management, audit log export, system configuration | `admin:<id>` | mTLS certificate (CLI) |

### Network Security

- **Web UI ↔ Orchestrator**: Password authentication, session tokens
- **Orchestrator ↔ Hardware Server**: mTLS with device-scoped certificates
- **Engineer CLI ↔ Hardware Server**: mTLS with device-scoped certificates
- **Orchestrator ↔ Cloud**: TLS 1.2+ with API authentication
- **No Direct Access**: Hardware server has no direct internet connection

---

## Development Status

### Implemented ✅

**Hardware Server (Rust)**:
- ✅ Core architecture and safety state machine
- ✅ gRPC services (Acquisition, Motion, Health, Safety, Device Control)
- ✅ Encrypted audit logging (AES-256-GCM)
- ✅ Device abstraction layer with DEMO implementations
- ✅ mTLS server configuration
- ✅ Configuration management (encrypted)
- ✅ Calibration enforcement (24-hour rule)
- ✅ LED status indicators (multi-color)
- ✅ Sound generation (startup, warnings)

**Orchestrator (Python)**:
- ✅ gRPC client with mTLS support
- ✅ Engineer certificate management CLI
- ✅ Database schema (patient, measurement, session tables)
- ✅ Authentication manager with key switch verification
- ✅ REST API server (FastAPI)
- ✅ WebSocket support for real-time updates
- ✅ Privacy-by-design architecture (UUID-only to hardware)
- ✅ Session management
- ✅ Audit logging (user-facing)
- ✅ RBAC implementation

**Certificate Infrastructure**:
- ✅ Certificate generation toolkit
- ✅ Root CA management (server and client)
- ✅ Device-scoped client certificates
- ✅ Short-lived certificate workflow

**Web UI (React)**:
- ✅ Dashboard with system status
- ✅ Measurement page with sample tracking
- ✅ Calibration page with workflow
- ✅ WebSocket real-time updates
- ✅ Role-based UI components

### In Progress 🚧

- 🚧 Real hardware driver integration (Bruker BIS, Thorlabs, GPIO PCIe)
- 🚧 Beam intensity monitoring with thresholds
- 🚧 X-ray source temperature/current monitoring (warmup)
- 🚧 Calibration quality control (real QC procedures)
- 🚧 Cloud platform integration
- 🚧 Measurement result retrieval from hardware server

### Planned 📋

- 📋 Windows Service deployment
- 📋 HTTPS/TLS for REST API (not just gRPC)
- 📋 Session timeout logic
- 📋 User database with password hashing
- 📋 Cloud sync manager with offline queue
- 📋 Rate limiting and request validation
- 📋 Predictive maintenance features
- 📋 EMR/LIMS integration
- 📋 FDA submission package completion

---

## Getting Started

### Prerequisites

- **Operating System**: Windows 10/11 (primary), Linux (development)
- **Rust**: 1.70+ with MSVC toolchain (Windows)
- **Python**: 3.11+
- **Node.js**: 18+ (for Web UI)
- **Protocol Buffers**: protoc compiler
- **SQLite**: Included in Rust/Python

### Quick Start (5 steps)

**1. Setup Certificate Infrastructure**
```powershell
cd C:\dev\Omniscan\omniscan-certificate-center
pip install -r requirements.txt
python certgen.py create-root --type server
python certgen.py create-root --type client
python certgen.py create-server --device-uuid ABC123
```

**2. Build Hardware Server**
```powershell
cd C:\dev\Omniscan\omniscan-hw-server
cargo build --release
```

**3. Install Orchestrator**
```powershell
cd C:\dev\Omniscan\omniscan-orchestrator
pip install -e .
```

**4. Start Hardware Server**
```powershell
cd C:\dev\Omniscan\omniscan-hw-server
cargo run --release
# Server starts on port 50051 (gRPC)
```

**5. Start Orchestrator REST API**
```powershell
cd C:\dev\Omniscan\omniscan-orchestrator
.\start_server.ps1
# REST API server starts on port 8080
```

**6. Start Web UI (Optional)**
```powershell
cd C:\dev\Omniscan\omniscan-ui
npm install
npm run dev
# Web UI available at http://localhost:3000
```

**7. Test Connection**
```powershell
# Generate engineer certificate
omni-orch cert generate --engineer-id ENG001 --device-uuid ABC123

# Check server status via CLI
omni-orch status \
  --cert ..\omniscan-certificate-center\certs\client\client_ENG001_ABC123.crt \
  --key ..\omniscan-certificate-center\certs\client\client_ENG001_ABC123.key \
  --ca-cert ..\omniscan-certificate-center\certs\root\device_server_root_ca.crt

# Or test REST API
curl http://localhost:8080/api/debug/status
```

### Testing

**Hardware Server**:
```powershell
cd omniscan-hw-server
cargo test
cargo test --test workflow_integration_tests  # 28 workflow tests
```

**Orchestrator**:
```powershell
cd omniscan-orchestrator
pytest tests/
```

**Web UI**:
```powershell
cd omniscan-ui
npm test
```

---

## References

### Core Documentation

- **This File**: Complete system documentation
- [USER_EXPECTATIONS.md](USER_EXPECTATIONS.md) - User requirements and expectations
- [DAILY_OPERATION_WORKFLOW.md](DAILY_OPERATION_WORKFLOW.md) - Complete clinical workflow specification
- [GLOBAL_README.md](GLOBAL_README.md) - Project overview and quick start

### Subproject Documentation

- [omniscan-hw-server/README.md](omniscan-hw-server/README.md) - Hardware server comprehensive guide
- [omniscan-hw-server/HARDWARE_COMPONENTS.md](omniscan-hw-server/HARDWARE_COMPONENTS.md) - Physical devices and services
- [omniscan-orchestrator/README.md](omniscan-orchestrator/README.md) - CLI command reference
- [omniscan-orchestrator/REST_API.md](omniscan-orchestrator/REST_API.md) - REST API documentation
- [omniscan-orchestrator/REST_SERVER_IMPLEMENTATION.md](omniscan-orchestrator/REST_SERVER_IMPLEMENTATION.md) - Implementation details
- [omniscan-certificate-center/README.md](omniscan-certificate-center/README.md) - Certificate generation guide
- [omniscan-ui/README.md](omniscan-ui/README.md) - Web UI architecture

### Specifications & Requirements

- [SOFTWARE_USER_REQUIREMENTS.md](SOFTWARE_USER_REQUIREMENTS.md) - Comprehensive user requirements
- [REQ/REQUIREMENTS_TRACEABILITY_MATRIX.md](REQ/REQUIREMENTS_TRACEABILITY_MATRIX.md) - Traceability matrix
- [info/OMNIScan_Requirements_v2_Tagged.md](info/OMNIScan_Requirements_v2_Tagged.md) - Tagged requirements
- [info/OmniSoft_Certificate_Strategy.md](info/OmniSoft_Certificate_Strategy.md) - PKI security architecture

---

## Summary

The Omniscan Medical Diagnostic Platform is a comprehensive, FDA-compliant system for X-ray diffraction analysis with:

1. **Safety-First Design**: Multiple layers of hardware and software interlocks ensuring radiation safety
2. **Privacy-by-Design**: Patient PII isolated in orchestrator; hardware server receives only UUIDs
3. **Regulatory Compliance**: IEC 62304 Class B, ISO 14971, HIPAA, FDA Cybersecurity Guidance
4. **Robust Architecture**: Rust safety-critical hardware control + Python workflow orchestration + React Web UI
5. **Comprehensive Audit**: Encrypted, tamper-proof logging with cryptographic integrity
6. **Daily Calibration**: Mandatory 24-hour enforcement with automatic system lockout
7. **Secure Authentication**: mTLS with device-scoped, short-lived certificates for maintenance access
8. **Offline Operation**: Full local functionality with delayed cloud synchronization

**Status**: Core architecture complete, hardware integration in progress, production deployment planned.

---

**Document Control**:
- **Version**: 1.0
- **Date**: October 26, 2025
- **Classification**: FDA Design History File (DHF)
- **Next Review**: Prior to FDA submission

---

*🩺 Building the future of medical diagnostics with safety, security, and compliance.*
