# Omniscan Orchestrator - Architecture & System Overview

## System Overview

The Omniscan Orchestrator is a Python-based middleware layer that manages communication between the React UI, the Rust hardware server, and a local SQLite database. It provides REST APIs, gRPC client functionality, and database management for a medical X-ray diffraction diagnostic device.

## Architecture Components

### 1. **Communication Layers**

```
React UI (Browser)
    ↓ REST API / WebSocket
Orchestrator (Python FastAPI)
    ↓ gRPC
Hardware Server (Rust)
    ↓ Direct Hardware Control
Devices (Detector, Motion, GPIO)
```

### 2. **Core Services**

- **REST API Server** (`rest_server.py`): FastAPI-based REST endpoints for UI
- **gRPC Client** (`grpc_client.py`): Communication with hardware server
- **Database Service** (`database.py`): SQLite database for patient records, measurements, audit logs
- **CLI** (`cli.py`): Command-line interface for engineers and operators

### 3. **Database Architecture**

**Location**: `C:\dev\Omniscan\omniscan-orchestrator\data\orchestrator.db`
**Type**: SQLite 3 with encryption at rest

#### Core Tables:
- **patients**: Patient information (PII - never transmitted)
- **measurements**: Measurement records linked to patients and calibrations
- **calibration_log**: Daily calibration records with QC data
- **ui_command_log**: Complete audit trail of UI commands
- **system_event_log**: System-level events and errors
- **user_sessions**: Session management with timeout tracking

#### Privacy Architecture:
- **Patient PII stays local**: Only UUIDs transmitted to hardware server
- **Audit trail**: All operations logged with timestamp, operator, and context
- **Calibration context**: Every measurement linked to the calibration used
- **Encryption**: Database file encrypted at rest (AES-256)

### 4. **Safety Architecture**

#### Enable Button Workflow
Potentially harmful operations require physical confirmation via an Enable Button:
- **Harmful Operations**: Initialize detector, initialize motion, start exposure
- **Safe Operations**: Read states, stop operations, power off
- **Timeout**: 20 seconds (configurable on hardware server)

#### Safety Interlocks
All checked before harmful operations:
- Key switch ON
- Emergency stop not pressed
- Door closed
- Cooling system OK
- Power supply OK
- Enable button active

### 5. **State Management**

#### Server States:
- **IDLE**: Ready for operations
- **PENDING_ARMED**: Waiting for safety confirmation
- **RUNNING**: Measurement in progress
- **STOPPING**: Graceful shutdown
- **SAFE**: Safe state after emergency stop
- **LOCKED**: Requires calibration
- **CALIBRATION**: Calibration in progress
- **MAINTENANCE**: Maintenance mode active

#### Device States:
- **Detector**: OFF, INIT, IDLE, EXPOSING, READING, ERROR
- **Motion**: OFF, INIT, IDLE, MOVING, HOMING, ERROR, LIMIT_HIT

## Key Design Principles

### 1. **Privacy-First**
- Patient PII never leaves orchestrator database
- Only UUIDs transmitted to hardware and cloud
- Complete audit trail for HIPAA/GDPR compliance

### 2. **Safety-First**
- Physical confirmation for harmful operations
- Automatic safe state transitions on errors
- Comprehensive interlock checking

### 3. **Version Compatibility**
- Command Discovery service for API introspection
- Protocol version validation (major version must match)
- Startup compatibility checks block incompatible versions

### 4. **Audit & Compliance**
- Cryptographic hash chaining in audit logs
- Append-only log files with 7-year retention
- FDA 21 CFR Part 11 compliant
- HIPAA and GDPR ready

## Data Flow Examples

### Daily Calibration
```
1. Clinician clicks "Start Calibration" in UI
2. UI → POST /api/calibration/start (orchestrator)
3. Orchestrator → calibrate_detector() (hardware server via gRPC)
4. Hardware server performs calibration, returns QC results
5. Orchestrator stores in calibration_log table
6. UI receives WebSocket event: calibration_complete
7. All subsequent measurements reference this calibration_id
```

### Patient Measurement
```
1. Clinician searches for patient by MRN (UI)
2. UI → GET /api/patients/search?mrn=123 (orchestrator)
3. Orchestrator queries local database (PII stays local)
4. Clinician starts measurement
5. Orchestrator generates measurement UUID
6. Orchestrator → start_exposure(UUID) (hardware server)
   - Only UUID transmitted, no patient info
7. Hardware server performs measurement
8. Results returned to orchestrator
9. Orchestrator stores in measurements table with patient_id link
10. UI displays results
```

## System Initialization

### Hardware Server Startup
1. Initialize GPIO system
2. Create state machine
3. Start gRPC server
4. Register all services (Acquisition, Motion, Safety, Health, etc.)

### Orchestrator Startup
1. Initialize database (create tables if needed)
2. Connect to hardware server via gRPC
3. Validate protocol compatibility
4. Start REST API server
5. Start WebSocket server for real-time updates

### Compatibility Check
On startup, orchestrator validates:
- Protocol version matches (major version)
- Required commands available on server
- Server version compatibility
- **Blocks startup if incompatible** with message: "I cannot communicate with the server, I am outdated"

## Configuration

### Hardware Server Config
```toml
# config/server.toml
[workflow]
enable_button_timeout = 20  # seconds
calibration_validity_hours = 24
max_measurement_duration_sec = 300

[safety]
require_enable_button = true
enforce_interlocks = true
```

### Orchestrator Config
- Database path: `data/orchestrator.db`
- Backup path: `data/backups/`
- Audit log path: `data/audit_logs/`
- Session timeout: 30 minutes

## Protocol Versioning

**Current**: Protocol v1.0.0

### Breaking Changes (bump major):
- Remove commands
- Change request/response fields (non-optional)
- Change field types

### Non-Breaking Changes (bump minor):
- Add new commands
- Add optional fields
- Add new services

## Deployment Architecture

### Development
```
Windows Development Machine
├── omniscan-hw-server (Rust) - Port 50051 (gRPC)
├── omniscan-orchestrator (Python) - Port 8080 (REST)
└── omniscan-ui (React) - Port 3000 (Dev Server)
```

### Production
```
Clinical Workstation
├── Hardware Server (Windows Service)
├── Orchestrator (Windows Service)
└── UI (Electron App or Browser)
```

## Monitoring & Maintenance

### Regular Tasks
- **Daily**: Automatic database backup (encrypted)
- **Weekly**: Vacuum database to optimize performance
- **Monthly**: Test backup restore procedure
- **Quarterly**: Review access logs
- **Annually**: Rotate encryption keys

### Alerts
- Database size > 10 GB
- Backup failure
- Query performance > 5 seconds
- Unusual access patterns
- Failed authentication attempts

## Error Handling

### Error Classification
- **SAF-xxx**: Safety-critical errors (automatic safe state transition)
- **DEV-xxx**: Device errors (detector, motion, GPIO)
- **SYS-xxx**: System errors (database, network, gRPC)
- **AUTH-xxx**: Authentication/authorization errors
- **VAL-xxx**: Validation errors (bad input, QC failures)

### Error Response Format
```python
{
    "error_code": "SAF-001",
    "message": "Safety door open - cannot start exposure",
    "operator_action": "Close safety door and verify door sensor",
    "technical_details": "Door sensor reading: 0 (expected: 1)",
    "timestamp": "2025-11-04T12:30:00Z"
}
```

## Summary

The Omniscan Orchestrator serves as the **privacy-first, safety-critical middleware** that:
- Keeps patient PII local while enabling distributed operations
- Enforces physical safety confirmation for harmful operations
- Provides complete audit trails for regulatory compliance
- Ensures version compatibility across system components
- Manages database persistence with encryption and backups
- Exposes REST APIs for UI and CLI tools for engineers
