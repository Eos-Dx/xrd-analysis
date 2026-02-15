# Orchestrator Alignment with COMMANDS_SPEC.md

## Executive Summary

**Status**: ⚠️ **Partially Aligned** - Infrastructure in place, needs integration

The orchestrator has most required infrastructure but **does NOT** actively log gRPC commands to its `command_logs` table yet.

## Detailed Analysis

### ✅ What's Aligned

#### 1. CommandContext Usage (SPEC Line 7-11)
**Status**: ✅ **FULLY COMPLIANT**

- **Location**: `grpc_client.py` line 76-85
- **Implementation**:
  ```python
  def _create_context(self, user: str, reason: str = "") -> hub_pb2.CommandContext:
      return hub_pb2.CommandContext(
          command_id=str(uuid.uuid4()),  # ✅ UUID
          user=user,                      # ✅ operator_id
          reason=reason,                  # ✅ free text
          timestamp=...                   # ✅ UTC timestamp
      )
  ```
- **Used in**: All gRPC command methods (StartExposure, Stop, MoveTo, InitializeDetector, etc.)
- **compliance**: 100% - Every command sends proper CommandContext

#### 2. Database Schema (SPEC Line 108-109)
**Status**: ✅ **IMPLEMENTED**

- **command_logs table**: ✅ Exists (`database.py` line 306-318)
  - Has all required fields: id, session_id, timestamp, command_type, command_data, hw_server_command_id, user_context, result, execution_time_ms, error_message
  - Mirrors hw-server structure per spec

- **measurements table**: ✅ Exists (`database.py` line 203-233)
  - Maps `measurement_id` ↔ `patient_id` ✅
  - Tracks timestamps, status, operator ✅
  - Stores calibration linkage ✅

#### 3. Logging Methods (SPEC Line 108)
**Status**: ✅ **IMPLEMENTED**

- **Location**: `database.py` lines 1004-1085
- **Methods**:
  - `log_command()` - Logs orchestrator-side command execution ✅
  - `get_recent_commands()` - Retrieves command history ✅
- **Fields logged**: command_id, service.command format, hw_server_command_id, operator, timing, result

#### 4. Patient Privacy (SPEC Line 109)
**Status**: ✅ **ENFORCED**

- Patient PII stored only in orchestrator ✅
- Only measurement_id (UUID) sent to hw-server ✅
- `patient_id` never leaves orchestrator ✅

### ❌ What's Missing

#### 1. Active Command Logging (CRITICAL)
**Status**: ❌ **NOT IMPLEMENTED**

**Problem**: `grpc_client.py` methods do NOT call `database.log_command()`

**Current behavior**:
```python
def start_measurement(self, exposure_time_ms: int, user: str = "unknown"):
    ctx = self._create_context(user, "Start measurement")
    request = hub_pb2.StartExposureRequest(ctx=ctx, ...)
    self.acquisition.StartExposure(request)
    return {"status": "started"}
    # ❌ MISSING: No call to self.db.log_command()
```

**Required behavior** (per SPEC line 108):
```python
def start_measurement(self, exposure_time_ms: int, user: str = "unknown"):
    start_time = time.time()
    ctx = self._create_context(user, "Start measurement")
    hw_command_id = ctx.command_id  # For linking
    
    try:
        request = hub_pb2.StartExposureRequest(ctx=ctx, ...)
        self.acquisition.StartExposure(request)
        
        # ✅ Log successful command
        self.db.log_command(
            command_id=str(uuid.uuid4()),  # Orchestrator command ID
            session_id=self.session_id,
            timestamp=datetime.utcnow(),
            command_type="Acquisition.StartExposure",  # Service.Command format
            command_data={"exposure_time_ms": exposure_time_ms, "user": user},
            hw_server_command_id=hw_command_id,  # Links to hw-server
            user_context=user,
            result="success",
            execution_time_ms=int((time.time() - start_time) * 1000)
        )
        
        return {"status": "started"}
    except grpc.RpcError as e:
        # ✅ Log failed command
        self.db.log_command(..., result="failure", error_message=str(e))
        return {"error": str(e)}
```

**Impact**: Cannot correlate orchestrator-side and hw-server-side command execution for audit trail

#### 2. Database Instance in gRPC Client
**Status**: ❌ **NOT AVAILABLE**

**Problem**: `OmniscanGrpcClient` class doesn't have access to `OrchestratorDatabase` instance

**Current**:
```python
class OmniscanGrpcClient:
    def __init__(self, server_address: str = "localhost:50051", ...):
        self.channel = ...
        # ❌ No self.db = database
```

**Required**:
```python
class OmniscanGrpcClient:
    def __init__(
        self, 
        server_address: str = "localhost:50051",
        database: Optional[OrchestratorDatabase] = None,  # ✅ Add
        session_id: Optional[str] = None,  # ✅ Add
        ...
    ):
        self.channel = ...
        self.db = database  # ✅ Store reference
        self.session_id = session_id or str(uuid.uuid4())  # ✅ Session tracking
```

#### 3. measurement_id Linkage (SPEC Line 123, 129)
**Status**: ⚠️ **PARTIALLY IMPLEMENTED**

**Problem**: `rest_server.py` may not be using `ctx.command_id` as `measurement_id`

**Required flow**:
1. UI calls orchestrator REST API to start measurement
2. Orchestrator creates CommandContext with command_id
3. Orchestrator calls hw-server StartExposure with command_id
4. **Orchestrator records measurement with measurement_id = command_id** ← VERIFY THIS
5. hw-server logs with measurement_id = command_id

**Verification needed**: Check `rest_server.py` to ensure it uses the command_id from gRPC as measurement_id when calling `db.record_measurement()`

### 📊 Compliance Matrix

| Requirement (COMMANDS_SPEC.md) | Status | Notes |
|-------------------------------|--------|-------|
| **Line 7**: CommandContext with command_id, user, reason, timestamp | ✅ Full | All commands use `_create_context()` |
| **Line 12**: Orchestrator logs every command | ❌ Missing | Methods exist but not called |
| **Line 108**: command_logs table | ✅ Full | Table exists with correct schema |
| **Line 108**: Mirror hw-server timing | ❌ Missing | No active logging |
| **Line 109**: measurement_records table | ✅ Full | Table exists |
| **Line 109**: measurement_id ↔ patient_id mapping | ✅ Full | Implemented |
| **Line 123**: measurement_id == command_id | ⚠️ Verify | Need to check rest_server.py |
| **Line 129**: Orchestrator logs measurements | ⚠️ Verify | Table exists, usage TBD |

### 🔧 Required Changes

#### Change 1: Add Database to gRPC Client

**File**: `src/omniscan_orchestrator/grpc_client.py`

```python
from .database import OrchestratorDatabase
import time

class OmniscanGrpcClient:
    def __init__(
        self,
        server_address: str = "localhost:50051",
        database: Optional[OrchestratorDatabase] = None,
        session_id: Optional[str] = None,
        client_cert_path: Optional[str] = None,
        client_key_path: Optional[str] = None,
        ca_cert_path: Optional[str] = None,
    ):
        self.server_address = server_address
        self.db = database  # Add
        self.session_id = session_id or str(uuid.uuid4())  # Add
        
        # ... rest of initialization
```

#### Change 2: Add Logging to All Command Methods

**Template for all methods**:

```python
def <command_method>(self, ..., user: str = "unknown") -> Dict[str, Any]:
    start_time = time.time()
    ctx = self._create_context(user, "<Description>")
    hw_command_id = ctx.command_id
    orch_command_id = str(uuid.uuid4())  # Separate orchestrator command ID
    
    try:
        # Execute gRPC call
        request = hub_pb2.<CommandRequest>(ctx=ctx, ...)
        response = self.<service>.<Command>(request)
        
        # Log success if database available
        if self.db:
            self.db.log_command(
                command_id=orch_command_id,
                session_id=self.session_id,
                timestamp=datetime.utcnow(),
                command_type="<Service>.<Command>",  # e.g., "Acquisition.StartExposure"
                command_data={...},  # Command parameters as JSON
                hw_server_command_id=hw_command_id,
                user_context=user,
                result="success",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        
        return {...}  # Success response
        
    except grpc.RpcError as e:
        # Log failure if database available
        if self.db:
            self.db.log_command(
                command_id=orch_command_id,
                session_id=self.session_id,
                timestamp=datetime.utcnow(),
                command_type="<Service>.<Command>",
                command_data={...},
                hw_server_command_id=hw_command_id,
                user_context=user,
                result="failure",
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e)
            )
        
        return {"error": str(e)}
```

**Apply to all methods**:
- `start_measurement()` → "Acquisition.StartExposure"
- `stop_measurement()` → "Acquisition.Stop"
- `calibrate_detector()` → "Acquisition.CalibrateDetector"
- `initialize_detector()` → "DeviceInitialization.InitializeDetector"
- `initialize_motion()` → "DeviceInitialization.InitializeMotion"
- `move_to()`, `move_relative()`, `home()` → "Motion.*"
- etc.

#### Change 3: Update REST Server Initialization

**File**: `src/omniscan_orchestrator/rest_server.py`

```python
# Initialize database
db = OrchestratorDatabase(db_path)

# Initialize gRPC client WITH database
grpc_client = OmniscanGrpcClient(
    server_address=config.hw_server_address,
    database=db,  # ✅ Pass database
    session_id=session_id,  # ✅ Pass session
    ...
)
```

#### Change 4: Verify measurement_id Usage

**File**: `src/omniscan_orchestrator/rest_server.py`

Find where `db.record_measurement()` is called and ensure:

```python
# When starting measurement
ctx_command_id = ... # Get from gRPC response or track it
measurement_record = MeasurementRecord(
    measurement_id=ctx_command_id,  # ✅ Use command_id from gRPC
    patient_id=patient_id,
    ...
)
db.record_measurement(measurement_record)
```

## Testing Checklist

After implementing changes:

- [ ] Start orchestrator and verify `command_logs` table gets populated
- [ ] Execute StartExposure and verify:
  - [ ] Orchestrator `command_logs` has entry with Service.Command format
  - [ ] Entry has `hw_server_command_id` linking to hw-server
  - [ ] Entry has orchestrator-side execution_time_ms
- [ ] Verify measurement_id linkage:
  - [ ] Patient record in orchestrator `measurements` table
  - [ ] measurement_id matches command_id
  - [ ] hw-server `measurement_records` has same measurement_id
- [ ] Check failed commands are logged with error_message
- [ ] Query `db.get_recent_commands()` returns expected results

## Summary

**Current State**: 
- ✅ Infrastructure: 100% complete (tables, methods, CommandContext)
- ❌ Integration: 0% complete (no active logging)

**Work Needed**:
1. Pass database instance to gRPC client (5 min)
2. Add logging wrapper to ~15 command methods (2-3 hours)
3. Verify measurement_id linkage in REST server (30 min)
4. Test end-to-end (1 hour)

**Total effort**: ~4 hours to achieve full compliance

**Priority**: HIGH - Required for FDA/IEC 62304 audit trail compliance per COMMANDS_SPEC.md line 12
