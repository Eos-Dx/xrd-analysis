# Start Button Flow: UI → Orchestrator → Hardware Server

## Overview
When the **"Begin Scan"** button is clicked in the UI, it triggers a chain of events to initialize the Detector and Motion Control systems through the Omniscan orchestrator.

---

## 1. UI Layer (React)
**File:** `omniscan-ui/src/pages/MeasurementPage.tsx`

```typescript
const handleBeginScan = () => {
  if (selectedScan) {
    setIsScanning(true);
    // Currently simulates scan with 5s timeout
    setTimeout(() => {
      setIsScanning(false);
    }, 5000);
  }
};
```

**Current State:** This is a **LOCAL SIMULATION** - it doesn't communicate with the backend yet.

**What should happen:** Call API endpoint to start measurement.

---

## 2. API Client (TypeScript)
**File:** `omniscan-ui/src/services/api.ts`

The UI should call this method:

```typescript
async startMeasurement(params: MeasurementParams): Promise<ApiResponse<{ runId: string }>> {
  return this.request('/measurements/start', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}
```

**Endpoint:** `POST /api/measurements/start`

**Parameters:**
```typescript
interface MeasurementParams {
  patient_id: string;           // From patient database
  exposure_duration: number;    // Duration in milliseconds
  // Optional: scan_type, position, etc.
}
```

---

## 3. REST Server (Python - FastAPI)
**File:** `omniscan-orchestrator/src/omniscan_orchestrator/rest_server.py`

### Endpoint: POST `/api/measurements/start` (Line 351)

```python
@app.post("/api/measurements/start", response_model=MeasurementStartResponse)
async def start_measurement(
    request: MeasurementStartRequest,
    user: Dict = Depends(get_current_user)
):
```

**What it does:**
1. ✅ Verifies patient exists in local database
2. ✅ Generates UUID for measurement (privacy protection)
3. ✅ Stores measurement record locally (links UUID → patient)
4. ✅ Calls hardware server via gRPC with UUID only (NO patient PII)
5. ✅ Broadcasts measurement_start event to WebSocket clients
6. ✅ Returns measurement UUID to client

---

## 4. Hardware Initialization Flow

### Before Starting Measurement - Safety Checks

The measurement won't start until the **Enable Button** is activated on the physical GPIO panel. Here's the sequence:

```
UI: Click "Begin Scan"
  ↓
Orchestrator: start_measurement() 
  ↓
Check: Is Enable Button ACTIVE? (20-second window after physical button press)
  ├─ NO → Return error: "Click ENABLE button in GPIO panel"
  └─ YES → Continue
  ↓
Call: grpc_client.start_exposure_with_uuid()
```

---

## 5. gRPC Client - Hardware Communication
**File:** `omniscan-orchestrator/src/omniscan_orchestrator/grpc_client.py`

### Method: `start_exposure_with_uuid()` (Line 718)

```python
async def start_exposure_with_uuid(
    self,
    measurement_id: str,           # UUID only
    operator_id: str,              # User ID (no name)
    exposure_time_ms: int
) -> Dict[str, Any]:
```

**Preconditions Checked:**
- ✅ Enable button ACTIVE (within 20s window)
- ✅ Door CLOSED
- ✅ Radiation SAFE (beam blocked)

**gRPC Call:**
```python
request = hub_pb2.StartExposureRequest(
    ctx=ctx,
    exposure_time_ms=exposure_time_ms,
    max_timeout_ms=exposure_time_ms + 5000,
)
self.acquisition.StartExposure(request)
```

---

## 6. Separate: Detector & Motion Initialization

**IMPORTANT:** These are different from starting a measurement. They are SETUP operations.

### Detector Initialization
**Endpoint:** `POST /api/hardware/detector/init`
**File:** `rest_server.py`, Line 564

```python
@app.post("/api/hardware/{device}/init")
async def initialize_device(device: str, user: Dict = Depends(get_current_user)):
```

**gRPC Call:**
```python
result = grpc_client.initialize_detector(user=user["user_id"])
```

**Safety Requirements:**
- ✅ Key switch ON
- ✅ Enable button ACTIVE (20s window)
- ✅ Radiation SAFE
- ✅ Cooling OK
- ✅ Power OK

**Implementation:** `grpc_client.py`, Line 240
```python
def initialize_detector(self, user: str = "unknown") -> Dict[str, Any]:
    # 1. Check enable button
    button_check = self.check_enable_button()
    if not button_check.get("active", False):
        return {"error": "Enable/Activation button not active..."}
    
    # 2. Verify interlocks
    gpio_state = self.get_gpio_state()
    
    # 3. Call gRPC
    ctx = self._create_context(user, "Initialize detector for imaging")
    request = hub_pb2.InitializeDetectorRequest(ctx=ctx)
    response = self.device_init.InitializeDetector(request)
    
    return {
        "status": "initialized",
        "powered": response.powered,
        "initialized": response.initialized,
        "detector_status": hub_pb2.DetectorStatus.Name(response.status),
        "temperature": response.temperature,
    }
```

### Motion Initialization
**Endpoint:** `POST /api/hardware/motion/init`
**gRPC Call:**
```python
result = grpc_client.initialize_motion(user=user["user_id"])
```

**Implementation:** `grpc_client.py`, Line 294
```python
def initialize_motion(self, user: str = "unknown") -> Dict[str, Any]:
    # Same safety checks as detector
    # Calls: self.device_init.InitializeMotion(request)
    return {
        "status": "initialized",
        "powered": response.powered,
        "initialized": response.initialized,
        "is_homed": response.is_homed,
        "motion_status": hub_pb2.MotionStatus.Name(response.status),
    }
```

---

## 7. Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ UI LAYER (React)                                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ MeasurementPage.tsx                                         │ │
│ │ handleBeginScan() → api.startMeasurement()                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ API CLIENT LAYER (TypeScript)                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ api.ts                                                      │ │
│ │ POST /api/measurements/start                               │ │
│ │ Body: { patient_id, exposure_duration }                    │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR REST SERVER (Python/FastAPI)                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ rest_server.py : start_measurement()                       │ │
│ │                                                             │ │
│ │ 1. Validate patient exists                                │ │
│ │ 2. Generate measurement UUID                              │ │
│ │ 3. Store measurement record (links UUID → patient)        │ │
│ │ 4. Call gRPC to start exposure                            │ │
│ │ 5. Broadcast WebSocket event                              │ │
│ │ 6. Return measurement_id to client                        │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ gRPC CLIENT (Python)                                            │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ grpc_client.py : start_exposure_with_uuid()               │ │
│ │                                                             │ │
│ │ Safety Checks:                                            │ │
│ │ ✓ Enable button active (20s window)                       │ │
│ │ ✓ Door closed                                             │ │
│ │ ✓ Radiation safe (beam blocked)                           │ │
│ │                                                             │ │
│ │ Creates: StartExposureRequest(uuid, exposure_time_ms)     │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ HARDWARE SERVER (Rust gRPC)                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Acquisition.StartExposure()                               │ │
│ │                                                             │ │
│ │ - Verifies all safety interlocks                          │ │
│ │ - Initializes detector if needed                          │ │
│ │ - Starts exposure                                         │ │
│ │ - Manages radiation emission                              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Separate Initialization Sequence

If Detector and Motion need to be initialized BEFORE measurement starts:

```
UI: Click "Initialize Detector"
  ↓
API: POST /api/hardware/detector/init
  ↓
REST Server:
  ├─ Check enable button active
  ├─ Check all safety interlocks
  └─ Call grpc_client.initialize_detector()
  ↓
gRPC Client:
  └─ Send InitializeDetectorRequest to hardware server
  ↓
Hardware Server:
  ├─ Power on detector subsystem
  ├─ Perform initialization sequence
  └─ Return detector status
```

Same flow for Motion Control:
```
API: POST /api/hardware/motion/init
  → grpc_client.initialize_motion()
  → hardware server InitializeMotion()
```

---

## 9. Key Safety Requirements

### Enable Button / Activation Button
- **Purpose:** Physical confirmation by operator
- **Duration:** 20-second window after pressing
- **Required for:** 
  - Detector initialization
  - Motion initialization
  - Starting measurements
- **Location:** GPIO panel (hardware control board)
- **Check Method:** `grpc_client.check_enable_button()`

### Interlocks (All checked via GPIO state)
| Interlock | Status | Required For |
|-----------|--------|--------------|
| Key switch | ON | All device ops |
| Radiation safe | TRUE | Start measurement |
| Door closed | TRUE | Start measurement |
| Cooling OK | TRUE | Device initialization |
| Power OK | TRUE | Device initialization |
| Emergency stop | Not triggered | All operations |

---

## 10. WebSocket Real-Time Updates

When measurement starts, clients receive:

```python
await broadcast_event({
    "type": "measurement_start",
    "data": {
        "measurement_id": "<UUID>",
        "patient_name": "...",
        "timestamp": "...",
        "exposure_duration_ms": 5000,
        "countdown_start": True
    }
})
```

When hardware initializes, clients receive:

```python
await broadcast_event({
    "type": "hardware_init",
    "data": {
        "device": "detector",  # or "motion"
        "status": "initialized",
        "user": "operator_id",
        "detail": {...}
    }
})
```

---

## 11. Error Responses

### Enable Button Not Active
```json
{
  "error": "Enable button not active",
  "message": "Click ENABLE button in GPIO panel to initialize detector",
  "instructions": "You have 20 seconds after clicking the button to complete initialization"
}
```

**HTTP Status:** 412 (Precondition Failed)

### Radiation Not Safe
```json
{
  "error": "Radiation is NOT SAFE (beam not blocked) - block beam before initializing detector"
}
```

---

## 12. Implementation Checklist

To implement the full flow in the UI:

- [ ] Update `MeasurementPage.tsx` to call `api.startMeasurement()` instead of local simulation
- [ ] Pass `MeasurementParams` with patient_id and exposure_duration
- [ ] Handle API response and display measurement_id
- [ ] Listen to WebSocket for `measurement_start` and `hardware_init` events
- [ ] Display safety status updates from WebSocket
- [ ] Handle error responses (missing enable button, interlocks not satisfied, etc.)
- [ ] Add UI for detector/motion initialization endpoints (if needed)
- [ ] Implement countdown timer when measurement starts

---

## 13. Key Files Reference

| Layer | File | Key Functions |
|-------|------|----------------|
| UI | `omniscan-ui/src/pages/MeasurementPage.tsx` | `handleBeginScan()` |
| API Client | `omniscan-ui/src/services/api.ts` | `startMeasurement()`, `initializeDevice()` |
| REST Server | `omniscan-orchestrator/src/omniscan_orchestrator/rest_server.py` | `/api/measurements/start`, `/api/hardware/{device}/init` |
| gRPC Client | `omniscan-orchestrator/src/omniscan_orchestrator/grpc_client.py` | `start_exposure_with_uuid()`, `initialize_detector()`, `initialize_motion()` |
| Models | `omniscan-orchestrator/src/omniscan_orchestrator/models.py` | `MeasurementStartRequest`, `MeasurementStartResponse` |
| Database | `omniscan-orchestrator/src/omniscan_orchestrator/database.py` | `record_measurement()` |

---

## 14. Privacy Protection Design

**Key principle:** Patient PII NEVER reaches the hardware server.

```
UI (has patient name) 
  ↓
Orchestrator:
  - Receives: patient_id, exposure_duration
  - Generates: measurement_uuid (no PII)
  - Stores locally: uuid ↔ patient mapping
  ↓
Hardware Server (receives ONLY):
  - measurement_uuid
  - operator_id (user ID, no name)
  - exposure_time_ms

Result:
- Patient name never appears in hardware logs
- Hardware audit trail is UUID-based only
- Compliance with medical data privacy regulations
```
