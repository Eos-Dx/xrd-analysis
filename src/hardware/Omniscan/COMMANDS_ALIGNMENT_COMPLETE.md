# COMMANDS_SPEC.md Implementation Summary

## Overview

This document summarizes the implementation completed to align both `omniscan-hw-server` (Rust) and `omniscan-orchestrator` (Python) with the requirements in `COMMANDS_SPEC.md`.

## ✅ Completed: hw-server (Rust)

### 1. Enhanced Command Logging (SPEC line 12)

**File**: `src/grpc/services.rs`

**Changes**:
- Updated `AcquisitionService::log_command()` signature to accept `service` and `command` parameters (line 178)
- Command type now uses `Service.Command` format (e.g., `"Acquisition.StartExposure"`)
- Added `orchestrator_id` extraction from mTLS `device_uuid`
- Enhanced `command_data` JSON with:
  - `operator_id` (from `ctx.user`)
  - `orchestrator_id` (from mTLS device UUID)
  - `service` (e.g., "Acquisition")
  - `command` (e.g., "StartExposure")

**Example**:
```rust
self.log_command(&ctx, "Acquisition", "Start Exposure", CommandResult::Success, elapsed_ms).await;
```

Results in command_logs entry:
```json
{
  "command_type": "Acquisition.StartExposure",
  "command_data": {
    "operator_id": "user@example.com",
    "orchestrator_id": "device-uuid-123",
    "reason": "Patient measurement",
    "command_id": "abc-123-def",
    "service": "Acquisition",
    "command": "StartExposure"
  }
}
```

### 2. Measurement Records Logging (SPEC line 104-129)

**File**: `src/grpc/services.rs` (lines 257-279)

**Implementation**:
- Added `measurement_records` insert in `StartExposure` handler
- Records:
  - `measurement_id` (== `command_id` from CommandContext)
  - `timestamp_start`, `timestamp_end`
  - `exposure_time_ms`
  - `operator` (from `ctx.user`)
  - `status` (Started → Completed/Failed/Stopped/Aborted)
  - `detector_config`, `motion_position` (JSON)
  - `data_files`, `data_size_bytes`, `checksum`

**Key Feature**: measurement_id == command_id linkage per spec

**Status Tracking**:
- Starts with `MeasurementStatus::Started`
- Updated to `Completed` when exposure finishes successfully
- Can be `Failed`, `Stopped`, or `Aborted` based on outcome

### 3. Database Schema (Already Complete)

**File**: `src/audit/mod.rs` (lines 232-251, 254-267)

Tables already implemented:
- ✅ `command_logs` - Full command audit trail
- ✅ `measurement_records` - Measurement tracking with UUID linkage
- ✅ `calibration_records` - Calibration QC and PONI data
- ✅ `interlock_events` - Safety interlock state changes
- ✅ `sessions` - Server session tracking

## ✅ Completed: orchestrator (Python)

### 1. Command Logs Table

**File**: `src/omniscan_orchestrator/database.py` (lines 304-318)

**New Table**:
```sql
CREATE TABLE IF NOT EXISTS command_logs (
    id TEXT PRIMARY KEY,                  -- Orchestrator command ID
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    command_type TEXT NOT NULL,            -- Service.Command format
    command_data TEXT NOT NULL,            -- JSON
    hw_server_command_id TEXT,             -- Links to hw-server
    user_context TEXT,
    result TEXT NOT NULL,                  -- "success" or "failure"
    execution_time_ms INTEGER NOT NULL,
    error_message TEXT
)
```

**Purpose**:
- Mirrors hw-server command logging structure
- Tracks orchestrator-side execution time
- Links to hw-server commands via `hw_server_command_id`
- Enables full command traceability: UI → Orchestrator → HW-Server

### 2. Command Logging Methods

**File**: `src/omniscan_orchestrator/database.py` (lines 1004-1085)

**Added Methods**:

```python
def log_command(
    self,
    command_id: str,
    session_id: str, 
    timestamp: datetime,
    command_type: str,              # "Service.Command" format
    command_data: dict,
    hw_server_command_id: Optional[str],
    user_context: Optional[str],
    result: str,
    execution_time_ms: int,
    error_message: Optional[str] = None
)
```

```python
def get_recent_commands(
    self,
    limit: int = 100,
    session_id: Optional[str] = None
) -> List[dict]
```

### 3. Existing Measurement Infrastructure (Already Complete)

**File**: `src/omniscan_orchestrator/database.py` (lines 202-233)

Already implemented:
- ✅ `measurements` table with `patient_id` mapping
- ✅ `measurement_id` (UUID) linkage to hw-server
- ✅ Full measurement context tracking
- ✅ `record_measurement()` method

**Critical Feature**: Patient PII stays in orchestrator only; hw-server only receives UUID

## 📋 Remaining Work (TODO)

### A. hw-server

#### 1. Update All Other Services
The following services still need `log_command()` method updates:

- **MotionService** (line 820)
  - Update signature to accept `service, command`
  - Update all calls: `move_to`, `move_relative`, `home`, `stop`, `set_velocity`, `get_position`
  
- **DeviceControlService** (line 1114)
  - Update signature
  - Update calls: `power_device`, `get_detector_health`, `get_motion_health`, `get_device_state`
  
- **DeviceInitializationService** (line 1364)
  - Update signature
  - Update calls: `initialize_detector`, `initialize_motion`, `power_off_detector`, `power_off_motion`

#### 2. Background Measurement Logging
- Add `measurement_records` insert in `StartBackgroundMeasurement` handler (line 769)
- Set `status` = "Background"
- Otherwise identical to regular measurements

#### 3. Calibration Records Logging
- Find `CalibrateDetector` handler
- Add `calibration_records` insert with:
  - `calibration_id` (UUID)
  - QC results JSON
  - PONI file content
  - Operator, timestamp, status
- Use `audit_logger.log_calibration_record()`

### B. orchestrator

#### 1. Enhance gRPC Client
**File**: `src/omniscan_orchestrator/grpc_client.py`

Add command logging to all gRPC method wrappers:

```python
def start_measurement(self, exposure_time_ms: int, user: str = "unknown"):
    start_time = time.time()
    command_id = str(uuid.uuid4())
    
    try:
        ctx = self._create_context(user, "Start measurement")
        # Store ctx.command_id for hw_server_command_id
        hw_command_id = ctx.command_id
        
        response = self.acquisition.StartExposure(...)
        
        # Log successful command
        self.db.log_command(
            command_id=command_id,
            session_id=self.session_id,
            timestamp=datetime.utcnow(),
            command_type="Acquisition.StartExposure",
            command_data={"exposure_time_ms": exposure_time_ms, "user": user},
            hw_server_command_id=hw_command_id,
            user_context=user,
            result="success",
            execution_time_ms=int((time.time() - start_time) * 1000)
        )
        
        return {"status": "started"}
    except grpc.RpcError as e:
        # Log failed command
        self.db.log_command(..., result="failure", error_message=str(e))
        return {"error": str(e)}
```

#### 2. Link measurement_id to command_id
**File**: `src/omniscan_orchestrator/rest_server.py`

When calling `db.record_measurement()`:
- Use `command_id` from gRPC `CommandContext` as `measurement_id`
- This creates the linkage: patient_id ← measurement_id == command_id → hw-server logs

## 📊 Compliance Matrix

| Requirement (COMMANDS_SPEC.md) | hw-server | orchestrator | Notes |
|-------------------------------|-----------|--------------|-------|
| Service.Command naming (line 12) | ✅ Partial | ❌ TODO | AcquisitionService done, others need update |
| operator_id logging (line 12) | ✅ Yes | ✅ Yes | From ctx.user |
| orchestrator_id logging (line 12) | ✅ Yes | N/A | From mTLS device_uuid |
| command_logs table | ✅ Yes | ✅ Yes | Both implemented |
| measurement_records (line 104) | ✅ Partial | ✅ Yes | StartExposure done, background TODO |
| measurement_id == command_id (line 123) | ✅ Yes | ⚠️ Verify | Need to verify in rest_server.py |
| calibration_records (line 169) | ❌ TODO | ✅ Yes | hw-server needs implementation |
| Patient PII isolation | ✅ Yes | ✅ Yes | Server never receives patient_id |

## 🔧 Testing Plan

### hw-server
1. Compile: `cargo build`
2. Run tests: `cargo test`
3. Start server and verify:
   - `command_logs` table shows `Service.Command` format
   - `measurement_records` table populated on StartExposure
   - `measurement_id` matches `command_id`

### orchestrator
1. Run database migration (creates `command_logs` table)
2. Test measurement workflow:
   - Create patient
   - Start measurement
   - Verify `measurements` table has entry with correct `measurement_id`
   - Verify `command_logs` table has entry linked to hw-server command
3. Query logs:
   ```python
   db.get_recent_commands(limit=10)
   ```

## 📝 Implementation Files

### Modified Files

**hw-server**:
1. `src/grpc/services.rs` - Enhanced logging, measurement records
2. `IMPLEMENTATION_ALIGNMENT.md` - Tracking document (new)

**orchestrator**:
1. `src/omniscan_orchestrator/database.py` - Added command_logs table and methods

### Generated Documentation:
1. `COMMANDS_ALIGNMENT_COMPLETE.md` - This file
2. `IMPLEMENTATION_ALIGNMENT.md` - Detailed tracking

## 🚀 Next Steps

1. **Complete hw-server logging updates**:
   - Update MotionService, DeviceControlService, DeviceInitializationService
   - Add background measurement logging
   - Add calibration records logging

2. **Complete orchestrator integration**:
   - Enhance gRPC client with command logging
   - Verify measurement_id linkage in REST endpoints
   - Add command log retrieval endpoints

3. **Integration testing**:
   - Full workflow: UI → Orchestrator → HW-Server
   - Verify log correlation via command_id
   - Test patient privacy (PII never leaves orchestrator)

4. **Documentation**:
   - Update API docs with new logging behavior
   - Document command correlation for debugging
   - Add examples for log queries

## 🔗 References

- **COMMANDS_SPEC.md** - Lines 3, 12, 104-109, 129, 169
- **hw-server**: `src/grpc/services.rs`, `src/audit/mod.rs`
- **orchestrator**: `src/omniscan_orchestrator/database.py`, `grpc_client.py`
- **Database schemas**: Both projects maintain SQLite audit databases

## ✅ Status: Phase 1 Complete

**What's Done**:
- ✅ Enhanced command logging format (hw-server AcquisitionService)
- ✅ Measurement records logging (hw-server StartExposure)
- ✅ Command logs table (orchestrator)
- ✅ Command logging methods (orchestrator)
- ✅ Documentation (IMPLEMENTATION_ALIGNMENT.md, this file)

**What's Next**: Complete remaining services and integrate gRPC client logging
