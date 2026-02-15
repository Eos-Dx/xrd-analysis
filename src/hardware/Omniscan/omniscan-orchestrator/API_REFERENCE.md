# Omniscan Orchestrator - API Reference

## Overview

This document consolidates all API documentation for the Omniscan Orchestrator, including REST API endpoints, CLI commands, and gRPC client methods.

---

## REST API

**Base URL**: `http://localhost:8080/api` (configurable)  
**Authentication**: All endpoints require `X-Session-Id` header (except auth endpoints)

### Authentication

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "operator",
  "password": "password"
}

Response (200):
{
  "session_id": "uuid",
  "user_id": "operator",
  "role": "operator",
  "expires_at": "2025-11-04T13:00:00Z"
}
```

#### Logout
```http
POST /api/auth/logout
Headers: X-Session-Id: <session_id>

Response (200):
{ "message": "Logged out successfully" }
```

### System State

#### Get System State
```http
GET /api/state
Headers: X-Session-Id: <session_id>

Response (200):
{
  "state": "IDLE",
  "timestamp": "2025-11-04T12:30:00Z",
  "interlocks": {
    "overall_safe": true,
    "key_switch": true,
    "enable_button": false,
    "door_closed": true,
    "emergency_stop": false,
    "cooling_ok": true,
    "power_ok": true,
    "beam_shutter_closed": true
  },
  "devices": {
    "detector": {
      "powered": true,
      "initialized": true,
      "status": "DETECTOR_IDLE",
      "temperature": 25.3,
      "voltage": 220.0,
      "total_exposures": 42
    },
    "motion": {
      "powered": true,
      "initialized": true,
      "status": "MOTION_IDLE",
      "position": 100.5,
      "is_homed": true,
      "total_moves": 15
    }
  },
  "calibration": {
    "valid": true,
    "expires_at": null
  }
}
```

#### Get Health
```http
GET /api/health
Headers: X-Session-Id: <session_id>

Response (200):
{
  "status": "healthy",
  "grpc_connected": true,
  "database_ok": true,
  "uptime_seconds": 3600
}
```

### GPIO & Safety

#### Get GPIO State
```http
GET /api/gpio/state
Headers: X-Session-Id: <session_id>

Response (200):
{
  "key_switch_on": true,
  "enable_button_active": false,
  "enable_button_remaining_secs": 0,
  "door_closed": true,
  "emergency_stop_active": false,
  "beam_shutter_closed": true
}
```

#### Get Enable Button Status
```http
GET /api/gpio/enable-button
Headers: X-Session-Id: <session_id>

Response (200):
{
  "active": true,
  "remaining_secs": 15
}
```

### Hardware Control

#### Initialize Device
```http
POST /api/hardware/{device}/init
Headers: X-Session-Id: <session_id>
Path Parameters: device = "detector" | "motion"

Success (200):
{
  "success": true,
  "device": "detector",
  "status": "initialized",
  "detail": {
    "powered": true,
    "initialized": true,
    "temperature": 25.3
  }
}

Enable Button Not Active (412):
{
  "error": "Enable button not active",
  "message": "Click ENABLE button in GPIO panel to initialize detector",
  "instructions": "You have 20 seconds after clicking the button"
}
```

#### Stop/Power Off Device
```http
POST /api/hardware/{device}/stop
Headers: X-Session-Id: <session_id>
Path Parameters: device = "detector" | "motion"

Response (200):
{
  "success": true,
  "device": "detector",
  "status": "powered_off"
}
```

### Patients

#### Create Patient
```http
POST /api/patients
Headers: X-Session-Id: <session_id>
Content-Type: application/json

{
  "first_name": "John",
  "last_name": "Doe",
  "date_of_birth": "1980-01-15",
  "medical_record_number": "MRN123"
}

Response (201):
{
  "patient_id": "uuid",
  "first_name": "John",
  "last_name": "Doe",
  "date_of_birth": "1980-01-15",
  "medical_record_number": "MRN123",
  "created_at": "2025-11-04T12:30:00Z"
}
```

#### Search Patients
```http
GET /api/patients/search?mrn=MRN123
Headers: X-Session-Id: <session_id>

Response (200):
{
  "patients": [
    {
      "patient_id": "uuid",
      "first_name": "John",
      "last_name": "Doe",
      "date_of_birth": "1980-01-15",
      "medical_record_number": "MRN123"
    }
  ]
}
```

### Measurements

#### Start Measurement
```http
POST /api/measurements/start
Headers: X-Session-Id: <session_id>
Content-Type: application/json

{
  "patient_id": "uuid",
  "exposure_ms": 5000,
  "sample_id": "left_breast_1"
}

Response (200):
{
  "measurement_id": "uuid",
  "status": "started",
  "patient_id": "uuid",
  "exposure_ms": 5000
}

Enable Button Not Active (412):
{
  "error": "Enable button not active",
  "message": "Click ENABLE button to start measurement"
}
```

#### Stop Measurement
```http
POST /api/measurements/stop
Headers: X-Session-Id: <session_id>
Content-Type: application/json

{
  "run_id": "uuid"
}

Response (200):
{
  "status": "stopped",
  "measurement_id": "uuid"
}
```

#### Get Measurement History
```http
GET /api/measurements/history?patient_id=uuid&limit=50
Headers: X-Session-Id: <session_id>

Response (200):
{
  "measurements": [
    {
      "measurement_id": "uuid",
      "timestamp": "2025-11-04T12:30:00Z",
      "operator_id": "operator",
      "sample_id": "left_breast_1",
      "exposure_duration_ms": 5000,
      "qc_status": "pass",
      "calibration_id": "cal-uuid"
    }
  ]
}
```

### Calibration

#### Start Calibration
```http
POST /api/calibration/start
Headers: X-Session-Id: <session_id>

Response (200):
{
  "calibration_id": "uuid",
  "status": "started",
  "timestamp": "2025-11-04T12:30:00Z"
}
```

#### Get Calibration Status
```http
GET /api/calibration/status
Headers: X-Session-Id: <session_id>

Response (200):
{
  "status": "completed",
  "calibration_id": "uuid",
  "overall_pass": true,
  "qc_checks": [
    {
      "name": "total_intensity",
      "value": 12500.0,
      "threshold": 10000.0,
      "passed": true
    }
  ]
}
```

#### Get Latest Calibration
```http
GET /api/calibration/latest
Headers: X-Session-Id: <session_id>

Response (200):
{
  "calibration_id": "uuid",
  "timestamp": "2025-11-04T08:00:00Z",
  "operator_id": "operator",
  "overall_pass": true,
  "expires_at": "2025-11-05T08:00:00Z"
}
```

### Command Discovery

#### Get Server Capabilities
```http
GET /api/v1/server/capabilities

Response (200):
{
  "server_version": "0.1.0",
  "protocol_version": "1.0.0",
  "build_time": "2025-11-04T10:00:00Z",
  "supported_features": ["calibration", "patient_measurements"],
  "device_type": "omniscan"
}
```

#### List Commands
```http
GET /api/v1/server/commands?service=Acquisition

Response (200):
{
  "commands": [
    {
      "service_name": "Acquisition",
      "command_name": "StartExposure",
      "description": "Start X-ray exposure",
      "request_fields": [
        {
          "name": "exposure_time_ms",
          "type": "uint32",
          "required": true,
          "description": "Exposure duration in milliseconds"
        }
      ],
      "safety_requirements": ["enable_button_active", "interlocks_safe"]
    }
  ],
  "server_info": {
    "server_version": "0.1.0",
    "protocol_version": "1.0.0"
  }
}
```

### WebSocket

#### Real-Time Updates
```javascript
// Connect
const ws = new WebSocket('ws://localhost:8080/ws');

// Events received:
{
  "type": "state_change",
  "data": {
    "state": "RUNNING",
    "timestamp": "2025-11-04T12:30:00Z"
  }
}

{
  "type": "gpio_state_change",
  "data": {
    "enable_button_active": true,
    "enable_button_remaining_secs": 15
  }
}

{
  "type": "calibration_complete",
  "data": {
    "calibration_id": "uuid",
    "overall_pass": true
  }
}

{
  "type": "measurement_complete",
  "data": {
    "measurement_id": "uuid",
    "status": "completed"
  }
}
```

---

## CLI Reference

**Command**: `omni-orch` or `omni-rest` (for REST CLI wrapper)

### Authentication

```bash
# Login (caches session)
omni-rest auth login -u operator -p 'password' --base-url http://localhost:8081

# Logout
omni-rest auth logout --base-url http://localhost:8081
```

### Server Discovery

```bash
# Get server capabilities
omni-rest server capabilities --base-url http://localhost:8081

# List all commands
omni-rest server commands --base-url http://localhost:8081

# List commands by service
omni-rest server commands --service Acquisition --base-url http://localhost:8081

# Check readiness
omni-rest server readiness --base-url http://localhost:8081
omni-rest server readiness --service Acquisition --base-url http://localhost:8081
```

### System

```bash
# Get system state
omni-rest system state --base-url http://localhost:8081

# Get health
omni-rest system health --base-url http://localhost:8081
```

### Patients

```bash
# Create patient
omni-rest patients create \
  --first-name John \
  --last-name Doe \
  --date-of-birth 1980-01-01 \
  --medical-record-number MRN123 \
  --base-url http://localhost:8081

# Search patients
omni-rest patients search --mrn MRN123 --base-url http://localhost:8081
```

### Measurements

```bash
# Start measurement
omni-rest measurements start \
  --patient-id <UUID> \
  --exposure-ms 5000 \
  --base-url http://localhost:8081

# Stop measurement
omni-rest measurements stop --run-id <UUID> --base-url http://localhost:8081

# Abort measurement
omni-rest measurements abort --run-id <UUID> --base-url http://localhost:8081

# Get history
omni-rest measurements history --limit 50 --base-url http://localhost:8081
```

### Calibration

```bash
# Start calibration
omni-rest calibration start --base-url http://localhost:8081

# Get status
omni-rest calibration status --base-url http://localhost:8081

# Get latest
omni-rest calibration latest --base-url http://localhost:8081
```

### GPIO & Safety

```bash
# Get GPIO state
omni-rest gpio state --base-url http://localhost:8081

# Get enable button status
omni-rest gpio enable-button --base-url http://localhost:8081
```

### Hardware Control

```bash
# Initialize device
omni-rest hardware init --device detector --base-url http://localhost:8081
omni-rest hardware init --device motion --base-url http://localhost:8081

# Stop device
omni-rest hardware stop --device detector --base-url http://localhost:8081
```

### Motion Control

```bash
# Stop motion
omni-rest motion stop --base-url http://localhost:8081
```

### Debug

```bash
# Get server status
omni-rest debug status --base-url http://localhost:8081

# Test connection
omni-rest debug connection --base-url http://localhost:8081
```

### Smoke Testing

```bash
# Run smoke tests (validates multiple endpoints)
omni-rest smoke -u operator -p 'password' --base-url http://localhost:8081
```

### Engineer Certificate Commands

```bash
# Generate engineer certificate
omni-orch cert generate \
  --engineer-id ENG001 \
  --device-uuid ABC123 \
  --validity-days 1

# List certificates
omni-orch cert list

# View certificate info
omni-orch cert info --cert-path <path-to-cert>

# Use certificate with commands
omni-orch status \
  --cert <cert-path> \
  --key <key-path> \
  --ca-cert <ca-path> \
  --base-url https://localhost:8443/api/v1
```

### Interactive Session

```bash
# Start interactive session
omni-orch interactive \
  --cert <cert-path> \
  --key <key-path> \
  --ca-cert <ca-path>

# Inside session:
omniscan> status
omniscan> enter-maintenance 900
omniscan> get-config
omniscan> device-power on
omniscan> measure-start 60 calibrant
omniscan> quit
```

---

## gRPC Client Methods

### Connection

```python
from omniscan_orchestrator.grpc_client import OmniscanGrpcClient

# Connect to server
client = OmniscanGrpcClient(server_address="localhost:50051")
```

### System State

```python
# Get full server state
state = client.get_full_server_state()

# Get GPIO state
gpio = client.get_gpio_state()

# Check enable button
button = client.check_enable_button()
# Returns: {"active": bool, "remaining_secs": int}
```

### Device Control

```python
# Initialize detector (requires enable button)
result = client.initialize_detector(user="operator@hospital.com")

# Initialize motion (requires enable button)
result = client.initialize_motion(user="operator@hospital.com")

# Power off detector (safe operation)
result = client.power_off_detector(user="operator@hospital.com")

# Power off motion (safe operation)
result = client.power_off_motion(user="operator@hospital.com")
```

### Measurements

```python
# Start measurement (requires enable button)
result = client.start_measurement(
    exposure_time_ms=5000,
    user="operator@hospital.com"
)

# Stop measurement (safe operation)
result = client.stop_measurement()

# Get measurement result
result = client.get_measurement_result()
```

### Calibration

```python
# Start calibration
result = client.calibrate_detector(user="operator@hospital.com")

# Get calibration status
status = client.get_calibration_status()
```

### Command Discovery

```python
# Get server capabilities
caps = client.get_server_capabilities()

# List available commands
commands = client.list_commands()

# Validate compatibility
compat = client.validate_compatibility(
    client_version="0.1.0",
    protocol_version="1.0.0",
    required_commands=["Acquisition.StartExposure"]
)
```

---

## Error Codes

### HTTP Status Codes

- **200 OK**: Success
- **201 Created**: Resource created
- **400 Bad Request**: Invalid input
- **401 Unauthorized**: Missing or invalid session
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **412 Precondition Failed**: Enable button not active or safety interlocks failed
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Hardware server unavailable

### Application Error Codes

- **SAF-001**: Safety door open
- **SAF-002**: Emergency stop active
- **SAF-003**: Enable button not active
- **DEV-001**: Detector error
- **DEV-002**: Motion error
- **SYS-001**: Database error
- **SYS-002**: gRPC connection error
- **AUTH-001**: Authentication failed
- **AUTH-002**: Session expired
- **VAL-001**: Invalid input
- **VAL-002**: Calibration QC failed

---

## Rate Limiting

- Login attempts: 3 per 5 minutes per IP
- API calls: 100 per minute per session
- WebSocket messages: No limit

---

## Deprecation Notices

### Deprecated Methods

```python
# Deprecated (use initialize_detector/power_off_detector instead)
client.power_device("detector", power_on=True, user="operator")
client.power_device("detector", power_on=False, user="operator")

# GPIO initialization removed (auto-initialized at startup)
client.power_device("gpio", power_on=True, user="operator")  # ❌ Returns error
```

---

## Examples

### Complete Measurement Workflow (REST)

```bash
# 1. Login
SESSION=$(curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"operator","password":"password"}' \
  | jq -r '.session_id')

# 2. Check system state
curl -H "X-Session-Id: $SESSION" http://localhost:8080/api/state

# 3. Click enable button (physical action)

# 4. Initialize detector
curl -X POST -H "X-Session-Id: $SESSION" \
  http://localhost:8080/api/hardware/detector/init

# 5. Start measurement
curl -X POST -H "X-Session-Id: $SESSION" \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"uuid","exposure_ms":5000}' \
  http://localhost:8080/api/measurements/start

# 6. Check status (poll or use WebSocket)
curl -H "X-Session-Id: $SESSION" \
  http://localhost:8080/api/measurements/status

# 7. Logout
curl -X POST -H "X-Session-Id: $SESSION" \
  http://localhost:8080/api/auth/logout
```

### Complete Measurement Workflow (Python)

```python
from omniscan_orchestrator.grpc_client import OmniscanGrpcClient

# Connect
client = OmniscanGrpcClient("localhost:50051")

# Check enable button
button = client.check_enable_button()
if not button["active"]:
    print("Please click ENABLE button")
    # Wait for button...

# Initialize detector
result = client.initialize_detector(user="operator")
if "error" in result:
    print(f"Error: {result['error']}")
else:
    # Start measurement
    result = client.start_measurement(
        exposure_time_ms=5000,
        user="operator"
    )
    
    # Wait for completion...
    
    # Get result
    measurement = client.get_measurement_result()
    print(f"Completed: {measurement}")
```
