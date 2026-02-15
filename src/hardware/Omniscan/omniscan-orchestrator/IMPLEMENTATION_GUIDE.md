# Omniscan Orchestrator - Implementation Guide

## Overview

This document consolidates implementation details, development workflows, and technical specifications for the Omniscan Orchestrator.

---

## System Implementation Status

### ✅ Completed Features

#### 1. Database Implementation
- **Schema**: All tables created (patients, measurements, calibration_log, ui_command_log, system_event_log, user_sessions)
- **Encryption**: Placeholders implemented, production encryption pending
- **Backup**: Automated backup system with 30-day retention
- **Audit**: Cryptographic hash chaining in audit logs

#### 2. REST API Server
- **Endpoints**: Authentication, system state, GPIO, hardware control, patients, measurements, calibration
- **WebSocket**: Real-time updates for state changes
- **Compatibility Check**: Startup validation blocks incompatible versions
- **Session Management**: 30-minute timeout with tracking

#### 3. gRPC Client
- **Services**: Acquisition, Motion, DeviceInitialization, StateMonitor, Safety, Health, CommandDiscovery
- **Enable Button**: Check and validate before harmful operations
- **Timeout Protection**: 13-second timeout for calibration
- **Error Handling**: Structured error responses

#### 4. CLI Tools
- **omni-orch**: Direct gRPC client commands
- **omni-rest**: REST API wrapper for testing
- **Interactive Session**: REPL for rapid development
- **Certificate Management**: Generate, list, view engineer certificates

#### 5. Security Features
- **RBAC**: Three-tier permission system
- **mTLS**: Certificate-based authentication
- **Audit Trail**: Complete logging with cryptographic integrity
- **Privacy**: Zero PII transmission architecture

---

## Development Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
cd omniscan-orchestrator
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Environment Configuration

```bash
# Set environment variables (PowerShell)
$env:OMNISCAN_DB_PATH = "C:\dev\Omniscan\omniscan-orchestrator\data\orchestrator.db"
$env:OMNISCAN_GRPC_SERVER = "localhost:50051"
$env:OMNISCAN_REST_PORT = "8080"
```

### Running Services

#### Hardware Server (Rust)
```bash
cd omniscan-hw-server
cargo run --release
# Starts gRPC server on port 50051
```

#### Orchestrator REST Server
```bash
cd omniscan-orchestrator
python -m omniscan_orchestrator.rest_server
# Starts REST API on port 8080
```

#### CLI Interactive Session
```bash
omni-orch interactive \
  --cert <cert-path> \
  --key <key-path> \
  --ca-cert <ca-path>
```

---

## Database Implementation

### Schema Creation

**Location**: `database.py`

```python
from omniscan_orchestrator.database import OrchestratorDatabase

# Initialize database (creates all tables)
db = OrchestratorDatabase("data/orchestrator.db")
```

### Recording Measurements

```python
from datetime import datetime
from omniscan_orchestrator.database import MeasurementRecord

# Get current calibration
calibration = db.get_current_calibration()

# Record measurement
measurement = MeasurementRecord(
    measurement_id="meas-uuid-123",
    patient_id="patient-uuid-456",
    timestamp=datetime.utcnow(),
    operator_id="dr_smith",
    measurement_type="patient",
    sample_id="left_breast_1",
    calibration_id=calibration.calibration_id,
    calibration_timestamp=calibration.timestamp,
    calibration_valid=True,
    exposure_duration_ms=5000,
    beam_intensity_mean=12500.5,
    snr=45.2,
    detector_temperature=25.3,
    qc_status="pass"
)

db.record_measurement(measurement)
```

### Audit Logging

```python
from omniscan_orchestrator.database import UICommandLog

# Log UI command
command = UICommandLog(
    timestamp=datetime.utcnow(),
    session_id="session-123",
    operator_id="dr_smith",
    command_type="measurement_start",
    command_payload='{"patient_id": "uuid-456"}',
    resource_id="meas-uuid-123",
    result="success",
    ip_address="192.168.1.100",
    user_agent="Mozilla/5.0..."
)

db.log_ui_command(command)
```

### Backup System

```python
from omniscan_orchestrator.backup import DatabaseBackupManager

backup_mgr = DatabaseBackupManager(
    db_path="data/orchestrator.db",
    backup_dir="data/backups",
    retention_days=30
)

# Perform daily backup
results = backup_mgr.perform_daily_backup()
print(f"Backup: {results['backup_path']}, Size: {results['size_mb']:.2f} MB")

# List backups
backups = backup_mgr.list_backups()

# Restore from backup
backup_mgr.restore_backup(backup_path="data/backups/orchestrator_2025-11-04.db")
```

---

## REST API Implementation

### Adding New Endpoints

**Location**: `rest_server.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from omniscan_orchestrator.models import ResponseModel
from omniscan_orchestrator.auth import get_current_user

router = APIRouter()

@router.get("/api/custom/endpoint")
async def custom_endpoint(user: Dict = Depends(get_current_user)):
    """Custom endpoint implementation."""
    try:
        # Access gRPC client
        global grpc_client
        result = grpc_client.custom_method()
        
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Register router
app.include_router(router)
```

### WebSocket Event Broadcasting

```python
from fastapi import WebSocket

# Broadcast event to all connected clients
async def broadcast_event(event_type: str, data: dict):
    for client in connected_clients:
        await client.send_json({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        })

# Example: Broadcast calibration complete
await broadcast_event("calibration_complete", {
    "calibration_id": "uuid-123",
    "overall_pass": True
})
```

---

## gRPC Client Development

### Adding New gRPC Methods

**Location**: `grpc_client.py`

```python
class OmniscanGrpcClient:
    def __init__(self, server_address: str):
        self.channel = grpc.insecure_channel(server_address)
        # Add new service stub
        self.custom_service = hub_pb2_grpc.CustomServiceStub(self.channel)
    
    def custom_method(self, param: str) -> Dict[str, Any]:
        """Custom gRPC method implementation."""
        try:
            request = hub_pb2.CustomRequest(
                ctx=self._create_context(),
                parameter=param
            )
            response = self.custom_service.CustomRPC(request)
            
            return {
                "success": True,
                "result": response.result
            }
        except grpc.RpcError as e:
            return {
                "error": e.details(),
                "code": str(e.code())
            }
```

### Error Handling Pattern

```python
def safe_grpc_call(self, method, *args, **kwargs):
    """Wrapper for safe gRPC calls with error handling."""
    try:
        response = method(*args, **kwargs)
        return {"success": True, "data": response}
    except grpc.RpcError as e:
        error_code = str(e.code())
        details = e.details()
        
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            return {"error": "Hardware server unavailable", "reconnecting": True}
        elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
            return {"error": details, "precondition_failed": True}
        else:
            return {"error": details, "code": error_code}
```

---

## Command Discovery Implementation

### Protocol Versioning

**Current Protocol**: v1.0.0

#### Breaking Changes (bump major):
- Remove commands from service
- Change non-optional request/response fields
- Change field types

#### Non-Breaking Changes (bump minor):
- Add new commands
- Add optional fields
- Add new services

### Compatibility Check

On startup, orchestrator validates compatibility:

```python
# In rest_server.py startup
compat = grpc_client.validate_compatibility(
    client_version="0.1.0",
    protocol_version="1.0.0",
    required_commands=[
        "Acquisition.StartExposure",
        "Motion.MoveTo",
        "Safety.GetInterlockStatus"
    ]
)

if not compat["compatible"]:
    print(f"❌ INCOMPATIBILITY: {compat['message']}")
    print(f"   Missing: {compat['missing_commands']}")
    print("❌ I cannot communicate with the server, I am outdated")
    sys.exit(1)
```

### Adding New Commands

**Hardware Server Side** (`omniscan-hw-server`):

1. Update `proto/hub/v1/hub.proto` with new RPC method
2. Rebuild proto files: `cargo build`
3. Implement service method in Rust
4. Add command descriptor to `command_discovery_service.rs`

**Orchestrator Side**:

1. Regenerate Python proto files:
```bash
python -m grpc_tools.protoc \
  -I../omniscan-hw-server/proto \
  --python_out=src/omniscan_orchestrator/hub/v1 \
  --grpc_python_out=src/omniscan_orchestrator/hub/v1 \
  ../omniscan-hw-server/proto/hub/v1/hub.proto
```

2. Add gRPC client method in `grpc_client.py`
3. Add REST API endpoint in `rest_server.py`
4. Update compatibility check to require new command

---

## Testing Implementation

### Unit Tests

**Location**: `tests/unit/`

```python
import pytest
from omniscan_orchestrator.grpc_client import OmniscanGrpcClient

def test_get_server_capabilities():
    client = OmniscanGrpcClient("localhost:50051")
    caps = client.get_server_capabilities()
    
    assert "server_version" in caps
    assert "protocol_version" in caps
    assert caps["protocol_version"] == "1.0.0"

def test_compatibility_check():
    client = OmniscanGrpcClient("localhost:50051")
    result = client.validate_compatibility(
        client_version="0.1.0",
        protocol_version="1.0.0"
    )
    
    assert result["compatible"] == True
```

### Integration Tests

```python
def test_measurement_workflow():
    # Setup
    client = OmniscanGrpcClient("localhost:50051")
    
    # Check enable button
    button = client.check_enable_button()
    assert button["active"] == True
    
    # Initialize detector
    result = client.initialize_detector(user="test_operator")
    assert "error" not in result
    
    # Start measurement
    result = client.start_measurement(
        exposure_time_ms=5000,
        user="test_operator"
    )
    assert result["status"] == "started"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_command_discovery.py

# Run with coverage
pytest --cov=omniscan_orchestrator tests/
```

---

## API Alignment Fixes

### Completed Fixes

#### 1. `/api/state` Endpoint
**Problem**: UI polls `/api/state` but endpoint was missing

**Solution**: Added endpoint in `rest_server.py`:
```python
@app.get("/api/state")
async def get_system_state(user: Dict = Depends(get_current_user)):
    full_state = grpc_client.get_full_server_state()
    gpio_state = grpc_client.get_gpio_state()
    
    return {
        "state": full_state.get("safety_state"),
        "interlocks": {...},
        "devices": {...},
        "calibration": {...},
        "timestamp": ...
    }
```

#### 2. Device Status Enums
**Problem**: UI expects `DETECTOR_INIT` / `MOTION_INIT` status values

**Solution**: Updated hardware server `hub.proto`:
```protobuf
enum DetectorStatus {
  DETECTOR_OFF = 0;
  DETECTOR_INIT = 1;  // NEW
  DETECTOR_IDLE = 2;
  DETECTOR_EXPOSING = 3;
  DETECTOR_READING = 4;
  DETECTOR_ERROR = 5;
}
```

#### 3. System State Enum
**Problem**: UI has `LOCKED`, `CALIBRATION`, `MAINTENANCE` states not in proto

**Solution**: Extended `ServerState` enum in `hub.proto`:
```protobuf
enum ServerState {
  STATE_UNSPECIFIED = 0;
  IDLE = 1;
  PENDING_ARMED = 2;
  RUNNING = 3;
  STOPPING = 4;
  SAFE = 5;
  LOCKED = 6;        // NEW
  CALIBRATION = 7;   // NEW
  MAINTENANCE = 8;   // NEW
}
```

---

## Enable Button Workflow Implementation

### Orchestrator Changes

#### 1. New gRPC Methods
```python
# Check enable button status
def check_enable_button(self) -> Dict[str, Any]:
    try:
        gpio_state = self.gpio.GetGPIOState(hub_pb2.Empty())
        return {
            "active": gpio_state.activation_button_active,
            "remaining_secs": gpio_state.activation_remaining_secs
        }
    except grpc.RpcError as e:
        return {"error": e.details()}

# Initialize detector (requires enable button)
def initialize_detector(self, user: str) -> Dict[str, Any]:
    try:
        request = hub_pb2.InitializeRequest(
            ctx=self._create_context(user=user),
            device_type="detector"
        )
        response = self.device_init.InitializeDevice(request)
        return {"status": "initialized", "success": True}
    except grpc.RpcError as e:
        return {"error": e.details(), "code": str(e.code())}
```

#### 2. REST API Updates
```python
@app.post("/api/hardware/{device}/init")
async def initialize_device(device: str, user: Dict = Depends(get_current_user)):
    # Check enable button first
    button = grpc_client.check_enable_button()
    if not button.get("active"):
        raise HTTPException(
            status_code=412,
            detail={
                "error": "Enable button not active",
                "message": f"Click ENABLE button to initialize {device}",
                "instructions": "You have 20 seconds after clicking"
            }
        )
    
    # Proceed with initialization
    if device == "detector":
        result = grpc_client.initialize_detector(user=user["user_id"])
    elif device == "motion":
        result = grpc_client.initialize_motion(user=user["user_id"])
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result
```

---

## Database Performance Optimization

### Indexes

Created for optimal query performance:

```sql
-- Patient lookup by MRN
CREATE INDEX idx_patients_mrn ON patients(medical_record_number);

-- Measurement queries
CREATE INDEX idx_measurements_patient ON measurements(patient_id);
CREATE INDEX idx_measurements_timestamp ON measurements(timestamp DESC);
CREATE INDEX idx_measurements_calibration ON measurements(calibration_id);

-- Calibration queries
CREATE INDEX idx_calibration_timestamp ON calibration_log(timestamp DESC);
CREATE INDEX idx_calibration_valid ON calibration_log(overall_pass, expires_at);

-- Audit log queries
CREATE INDEX idx_ui_log_timestamp ON ui_command_log(timestamp DESC);
CREATE INDEX idx_ui_log_operator ON ui_command_log(operator_id);
CREATE INDEX idx_system_log_timestamp ON system_event_log(timestamp DESC);
CREATE INDEX idx_system_log_severity ON system_event_log(severity);
```

### Query Patterns

```python
# Efficient patient lookup
patients = db.execute(
    "SELECT * FROM patients WHERE medical_record_number = ?",
    (mrn,)
).fetchall()

# Recent measurements with calibration context
measurements = db.execute("""
    SELECT m.*, c.overall_pass AS calibration_passed
    FROM measurements m
    LEFT JOIN calibration_log c ON m.calibration_id = c.calibration_id
    WHERE m.patient_id = ?
    ORDER BY m.timestamp DESC
    LIMIT ?
""", (patient_id, limit)).fetchall()

# Today's audit trail
commands = db.execute("""
    SELECT * FROM ui_command_log
    WHERE DATE(timestamp) = DATE('now')
    ORDER BY timestamp DESC
""").fetchall()
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Database encryption enabled (replace placeholders)
- [ ] Backup encryption enabled
- [ ] Windows Credential Manager configured
- [ ] TLS/mTLS certificates generated and installed
- [ ] File system ACLs applied
- [ ] Session timeout configured
- [ ] Rate limiting enabled
- [ ] Audit logging verified
- [ ] All tests passing
- [ ] Backup restore tested

### Windows Service Setup

```powershell
# Create Windows service for orchestrator
sc.exe create OmniscanOrchestrator `
  binPath= "C:\Python39\python.exe -m omniscan_orchestrator.rest_server" `
  start= auto `
  DisplayName= "Omniscan Orchestrator"

# Start service
sc.exe start OmniscanOrchestrator

# Check status
sc.exe query OmniscanOrchestrator
```

### Monitoring Setup

```python
# Add health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": db_connection.is_alive(),
        "grpc": grpc_client.is_connected(),
        "uptime_seconds": time.time() - start_time
    }
```

---

## Troubleshooting

### Common Issues

#### 1. gRPC Connection Refused
```python
# Check hardware server is running
curl http://localhost:50051

# Verify firewall allows port 50051
netsh advfirewall firewall show rule name=all | findstr 50051
```

#### 2. Database Lock
```python
# Check for active connections
SELECT * FROM pragma_database_list;

# Close all connections
db.close()
```

#### 3. Certificate Validation Failed
```bash
# Check certificate validity
openssl x509 -in cert.pem -text -noout

# Verify SAN matches device UUID
openssl x509 -in cert.pem -text | grep "URI"
```

#### 4. Startup Compatibility Failure
```
❌ INCOMPATIBILITY: incompatible protocol version
   Client: 2.0.0, Server: 1.0.0
   
Solution: Update orchestrator or hardware server to match versions
```

---

## Future Enhancements

### Pending Production Implementation

1. **Database Encryption**: Replace placeholders with SQLCipher or cryptography library
2. **Backup Encryption**: Integrate encryption with backup system
3. **Windows Credential Manager**: Replace file-based key storage
4. **Certificate Revocation**: Implement CRL checking
5. **Multi-Factor Authentication**: Add MFA for administrators

### Planned Features

1. **Certificate Renewal**: Auto-renew before expiration
2. **Hardware Token Support**: Store keys on smartcards/HSM
3. **Centralized Certificate Management**: Database-backed tracking
4. **Advanced Monitoring**: Prometheus metrics export
5. **Cloud Integration**: Secure upload to cloud storage

---

## Summary

The Omniscan Orchestrator implementation provides:

✅ **Complete database schema** with privacy-first design  
✅ **REST API** with authentication and real-time updates  
✅ **gRPC client** with comprehensive service coverage  
✅ **CLI tools** for development and maintenance  
✅ **Security features** including RBAC and audit trails  
✅ **Compatibility checking** to prevent version mismatches  
✅ **Enable button workflow** for safety-critical operations  
✅ **Automated backups** with retention policies  

**Next Steps**: Replace encryption placeholders, complete production deployment checklist, and implement pending enhancements.
