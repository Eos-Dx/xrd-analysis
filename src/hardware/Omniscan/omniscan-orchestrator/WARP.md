# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Omniscan Orchestrator** is a Python-based medical device middleware that manages communication between a React UI, a Rust hardware server, and a local SQLite database. It's part of an X-ray diffraction diagnostic system with strict privacy, safety, and compliance requirements (FDA 21 CFR Part 11, HIPAA, GDPR).

### Key Architecture Principles

1. **Privacy-First**: Patient PII never leaves the orchestrator database - only UUIDs are transmitted to the hardware server and cloud
2. **Safety-Critical**: Physical Enable Button confirmation required for potentially harmful operations (X-ray activation, motion system)
3. **Medical Device Compliance**: Complete audit trails with cryptographic hash chaining for regulatory compliance
4. **Version Compatibility**: Protocol version validation blocks incompatible client-server versions at startup

### System Communication Flow
```
React UI (Browser)
    ↓ REST API / WebSocket
Orchestrator (Python FastAPI) - Port 8080
    ↓ gRPC over mTLS
Hardware Server (Rust) - Port 50051
    ↓ Direct Hardware Control
Devices (Detector, Motion, GPIO)
```

## Common Development Commands

### Environment Setup
```powershell
# Create virtual environment (Python 3.11 required)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install in development mode
pip install -e .

# Install test dependencies
pip install -e ".[test]"
```

### Running Services

#### Start REST API Server
```powershell
# Using start script (recommended)
.\start_server.ps1

# Direct uvicorn (development)
python -m uvicorn omniscan_orchestrator.rest_server:app --host 0.0.0.0 --port 8080 --reload

# With conda environment
conda activate eosdx
python -m uvicorn omniscan_orchestrator.rest_server:app --host 0.0.0.0 --port 8081 --reload
```

### Testing

#### Run Full Test Suite
```powershell
# Run all tests with coverage (requires ≥80% per FDA requirements)
pytest

# Run specific test markers
pytest -m unit              # Unit tests only (fast)
pytest -m integration       # Integration tests (medium)
pytest -m safety            # Safety-critical tests
pytest -m audit             # Audit trail tests

# Run single test file
pytest tests/test_database.py

# Run with verbose output
pytest -v

# Skip coverage check during development
pytest --no-cov
```

#### Coverage Requirements
- **Minimum**: 80% code coverage (FDA requirement per VER_OMNI-SERVER_001)
- **Reports**: HTML (htmlcov/), XML (coverage.xml), and terminal output
- **Failures**: Tests fail if coverage drops below 80% (`--cov-fail-under=80`)

### CLI Tools

#### Main CLI (omni-orch) - Direct gRPC Communication
```powershell
# Check hardware server status
omni-orch status --server localhost:50051

# Get full health information
omni-orch health

# Check safety interlocks
omni-orch interlocks

# Get server state
omni-orch state

# Get detector health
omni-orch detector-health

# Get motion controller health
omni-orch motion-health

# Interactive session (maintenance)
omni-orch interactive --cert <cert> --key <key> --ca-cert <ca>

# Maintenance mode operations
omni-orch enter-maintenance --ttl 900  # 15 minutes
omni-orch renew --ttl 900
omni-orch exit-maintenance

# Configuration management
omni-orch get-config
omni-orch set-config --file config.json
```

#### REST CLI (omni-rest) - REST API Testing
```powershell
# Authentication
omni-rest auth login --username operator --password <pwd> --base-url http://localhost:8081
omni-rest auth logout

# System endpoints
omni-rest system health
omni-rest system state

# Patient management
omni-rest patients create --first-name John --last-name Doe --date-of-birth 1980-01-15 --mrn MRN123
omni-rest patients search --mrn MRN123
omni-rest patients get --patient-id <uuid>

# Measurements
omni-rest measurements start --patient-id <uuid> --exposure-ms 5000 --sample-id left_breast_1
omni-rest measurements stop --run-id <uuid>
omni-rest measurements abort --run-id <uuid>
```

### Database Operations

#### Inspect Database
```powershell
# View calibration history
python scripts/peek_calibrations.py

# Direct SQLite access
sqlite3 data/orchestrator.db
```

### Code Quality

#### Linting & Type Checking
```powershell
# The project uses pytest for testing but does not have dedicated lint/typecheck commands
# When adding code quality tools, add them to pyproject.toml [project.optional-dependencies]
```

## Code Architecture

### Core Modules

#### `rest_server.py` - FastAPI REST API
- **Purpose**: Exposes REST endpoints for UI, manages WebSocket connections
- **Authentication**: Session-based with X-Session-Id header
- **Startup**: Validates protocol compatibility with hardware server, blocks if incompatible
- **Key Endpoints**: /api/auth/*, /api/state, /api/patients/*, /api/measurements/*, /api/calibration/*

#### `grpc_client.py` - gRPC Client
- **Purpose**: Communicates with Rust hardware server over gRPC
- **Security**: Mutual TLS (mTLS) with certificate-based authentication
- **Services**: Acquisition, Motion, DeviceInitialization, StateMonitor, Safety, Health, CommandDiscovery
- **Enable Button**: Validates button state before harmful operations (20-second timeout)

#### `database.py` - SQLite Persistence
- **Purpose**: Local database for patient records, measurements, audit logs
- **Encryption**: AES-256 at rest (placeholders implemented, production encryption pending)
- **Privacy**: Patient PII never transmitted, only UUIDs leave the database
- **Tables**:
  - `patients` - Patient PII (local only)
  - `measurements` - Measurement records linked to patients/calibrations
  - `calibration_log` - Daily calibration QC data
  - `ui_command_log` - Complete audit trail of UI commands
  - `system_event_log` - System-level events and errors
  - `user_sessions` - Session management with timeout tracking

#### `audit.py` - Audit Logging Service
- **Purpose**: Cryptographic audit trail for regulatory compliance
- **Features**: SHA-256 hash chaining, append-only logs, 7-year retention
- **Format**: Daily rotation, CSV/JSON export for regulatory review

#### `auth_manager.py` & `rbac.py` - Authentication & Authorization
- **RBAC Roles**:
  - **Clinical Operators**: Measurements and calibration only
  - **Maintenance Engineers**: Full diagnostic access and maintenance mode
  - **Administrators**: User management and system configuration
- **Session Management**: 30-minute timeout, rate-limited failed attempts

#### `backup.py` - Database Backup System
- **Features**: Automated daily backups, 30-day retention, encrypted backups
- **Restore**: Integrity verification before restore

#### `cert_manager.py` - Certificate Management
- **Purpose**: Generate short-lived engineer certificates (default 1 day)
- **Scope**: Device-scoped via SAN (Subject Alternative Name)
- **Revocation**: Delete certificate files to revoke access

#### `cli.py` - Main CLI (omni-orch)
- **Purpose**: Direct gRPC client for engineers
- **Features**: Interactive session (REPL), status queries, maintenance mode

#### `rest_cli.py` - REST CLI (omni-rest)
- **Purpose**: REST API wrapper for testing and automation
- **Session Cache**: Stores session ID in ~/.omniscan_rest_session

#### `interactive.py` - Interactive REPL Session
- **Purpose**: Engineer-focused REPL for rapid development and diagnostics
- **Commands**: status, enter-maintenance, get-config, set-config, etc.

### Proto Files
- **Location**: `proto/hub.proto`
- **Generation**: Proto files from hardware server are used to generate Python gRPC stubs
- **Services**: Acquisition, Motion, Safety, Health, DeviceInitialization, StateMonitor, CommandDiscovery

## Safety & Compliance

### Enable Button Workflow
Potentially harmful operations require physical confirmation via Enable Button:
- **Required For**: Initialize detector, initialize motion, start exposure, motion commands
- **Not Required For**: Read states, stop operations, power off, view data
- **Timeout**: 20 seconds (configurable on hardware server)
- **Validation**: Check `enable_button_active` and `remaining_secs` before operations

### Safety Interlocks
All checked before harmful operations:
- Key switch ON
- Emergency stop not pressed
- Door closed
- Cooling system OK
- Power supply OK
- Enable button active

### Privacy Architecture
- **Patient PII** (first name, last name, DOB, MRN): **NEVER transmitted** outside orchestrator database
- **Transmitted**: Only measurement UUIDs to hardware server and cloud
- **Database joins**: Link measurement results back to patient records locally
- **Audit logs**: Only patient UUIDs, never PII

### Audit Requirements
- **Cryptographic integrity**: SHA-256 hash chaining prevents tampering
- **Append-only**: Logs cannot be modified after write
- **Retention**: 7 years (configurable)
- **Events logged**: Authentication, measurements, calibration, configuration changes, system events

## Error Handling

### Error Code Classification
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

## Development Guidelines

### State Management
**Server States**: IDLE, PENDING_ARMED, RUNNING, STOPPING, SAFE, LOCKED, CALIBRATION, MAINTENANCE
**Device States**:
- **Detector**: OFF, INIT, IDLE, EXPOSING, READING, ERROR
- **Motion**: OFF, INIT, IDLE, MOVING, HOMING, ERROR, LIMIT_HIT

### Protocol Versioning
**Current**: Protocol v1.0.0
- **Breaking changes** (bump major): Remove commands, change required fields, change field types
- **Non-breaking** (bump minor): Add new commands, add optional fields, add new services
- **Compatibility check**: Orchestrator validates major version match at startup, blocks if incompatible

### Database Migrations
When modifying schema:
1. Add migration logic to `database.py`
2. Test with existing database files
3. Document migration in IMPLEMENTATION_GUIDE.md
4. Consider backup/restore compatibility

### Testing Strategy
- **Unit tests** (`-m unit`): Individual module testing, mocked dependencies
- **Integration tests** (`-m integration`): Cross-module interactions
- **Safety tests** (`-m safety`): Safety-critical functionality (UR-SAFE-001)
- **Audit tests** (`-m audit`): Audit trail integrity (UR-AUDIT-001)
- **RBAC tests** (`-m rbac`): Role-based access control (UR-USER-002)

### Adding New Endpoints
1. Define route in `rest_server.py`
2. Add authentication via `Depends(get_current_user)`
3. Validate inputs with Pydantic models (models.py)
4. Log operation to audit trail
5. Add CLI wrapper in `rest_cli.py`
6. Add integration test in `tests/test_rest_api_endpoints.py`

### Configuration Files
- **pyproject.toml**: Python project configuration, dependencies, build system
- **pytest.ini**: Test configuration, coverage requirements, test markers
- **.gitignore**: Excludes database files, virtual environments, IDE files
- **config/**: (if exists) Hardware server configuration files

## Important File Locations
- **Database**: `data/orchestrator.db` (SQLite, encrypted at rest)
- **Backups**: `data/backups/` (30-day retention)
- **Audit Logs**: `data/audit_logs/` (7-year retention)
- **Certificates**: Usually in `C:\dev\Omniscan\omniscan-certificate-center\certs\`

## Documentation References
- **ARCHITECTURE.md**: System architecture, data flow, state management
- **API_REFERENCE.md**: Complete REST API endpoint documentation
- **SECURITY_COMPLIANCE.md**: Security, privacy, audit, and compliance requirements
- **USER_GUIDE.md**: Operator workflows, Enable Button guide, interactive session guide
- **IMPLEMENTATION_GUIDE.md**: Development workflows, database implementation, REST API patterns
- **COMMANDS_SPEC_IMPLEMENTATION.md**: gRPC command specifications and alignment
- **COMMANDS_SPEC_ALIGNMENT.md**: Protocol version compatibility matrix

## Windows-Specific Notes
- **PowerShell**: Primary shell for development (pwsh version 5.1+)
- **Paths**: Use backslashes for absolute Windows paths (e.g., `C:\dev\Omniscan\...`)
- **Virtual Environment Activation**: `.venv\Scripts\Activate.ps1` (not `source .venv/bin/activate`)
- **Default Ports**: REST API typically on 8080 or 8081, gRPC hardware server on 50051
