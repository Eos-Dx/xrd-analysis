# API Data Structure Alignment Issues

## Overview
This document identifies misalignments between:
- **UI Types** (`omniscan-ui/src/types/index.ts`)
- **Orchestrator API** (`omniscan-orchestrator/src/omniscan_orchestrator/rest_server.py`)
- **Hardware Server gRPC API** (`omniscan-hw-server/API_DOCUMENTATION.md`)

---

## Critical Issues

### 1. **MISSING ENDPOINT: `/api/state`**

**Problem:** UI expects a `/api/state` endpoint that doesn't exist in the orchestrator.

**UI Code:**
```typescript
// src/services/api.ts:115-116
async getSystemState(): Promise<ApiResponse<SystemStateResponse>> {
  return this.request<SystemStateResponse>('/state');
}
```

**UI Type Expected:**
```typescript
interface SystemStateResponse {
  system_state: SystemState;
  devices: CompactDevices;
  interlocks: CompactInterlocks;
  timestamp: string;
}
```

**Status:** ❌ **ENDPOINT MISSING** - The orchestrator only has `/api/health` but NOT `/api/state`

**Impact:** UI will fail to poll system state, causing broken dashboard

**Fix Required:** Add `/api/state` endpoint to orchestrator that returns compact state info

---

### 2. **System State Enum Mismatch**

**Problem:** UI has extra states not defined in the hardware server API.

**UI Definition:**
```typescript
type SystemState = 
  | 'IDLE' 
  | 'PENDING_ARMED' 
  | 'RUNNING' 
  | 'STOPPING' 
  | 'SAFE' 
  | 'CALIBRATION'   // ❌ NOT in hardware API
  | 'MAINTENANCE'   // ❌ NOT in hardware API
  | 'LOCKED';       // ❌ NOT in hardware API
```

**Hardware Server API (from API_DOCUMENTATION.md line 802-807):**
```
ServerState enum:
- IDLE
- PENDING_ARMED
- RUNNING
- STOPPING
- SAFE
```

**Impact:** UI may display states that hardware never sends

**Fix:** Remove `CALIBRATION`, `MAINTENANCE`, `LOCKED` from UI types OR document if orchestrator adds these

---

### 3. **Detector Status Mismatch**

**Problem:** UI uses different detector status values than hardware API.

**UI Definition:**
```typescript
type DetectorDeviceStatus = 'OFF' | 'IDLE' | 'INIT' | 'EXPOSING' | 'READING' | 'ERROR';
```

**Hardware API (API_DOCUMENTATION.md line 669):**
```protobuf
DetectorStatus status = 3;  // OFF, IDLE, EXPOSING, READING, ERROR
```

**Issues:**
- UI has `'INIT'` status ❌ (Hardware uses `initialized: bool` flag instead)
- Hardware has `DETECTOR_IDLE`, `DETECTOR_OFF` etc. with `DETECTOR_` prefix

**Fix:** Align UI types with hardware enum values

---

### 4. **Motion Status Mismatch**

**Problem:** Motion status values don't fully align.

**UI Definition:**
```typescript
type MotionDeviceStatus = 'OFF' | 'IDLE' | 'INIT' | 'MOVING' | 'HOMING' | 'ERROR' | 'LIMIT_HIT';
```

**Hardware API (API_DOCUMENTATION.md line 688):**
```protobuf
MotionStatus status = 4;  // OFF, IDLE, MOVING, HOMING, ERROR, LIMIT_HIT
```

**Issues:**
- UI has `'INIT'` status ❌ (Hardware uses `initialized: bool` flag instead)
- Hardware likely has `MOTION_` prefix (e.g., `MOTION_IDLE`)

**Fix:** Remove `'INIT'` from status enum, use separate `initialized` boolean

---

### 5. **Interlock Structure Differences**

**Problem:** Different required fields between UI and orchestrator.

**UI Type:**
```typescript
interface CompactInterlocks {
  overall_safe: boolean;       // ✅ Required
  key_switch: boolean;         // ✅ Required
  enable_button: boolean;      // ✅ Required
  door_closed?: boolean;       // ❓ Optional
  emergency_stop?: boolean;    // ❓ Optional
  radiation_safe?: boolean;    // ❓ Optional
  cooling_ok?: boolean;        // ❓ Optional
  power_ok?: boolean;          // ❓ Optional
}
```

**Orchestrator Model (models.py line 101-109):**
```python
class SafetyInterlocks(BaseModel):
    overall_safe: bool          # ✅ Required
    key_switch: bool            # ✅ Required
    enable_button: bool         # ✅ Required
    door_closed: bool           # ❌ REQUIRED (not optional!)
    emergency_stop: bool        # ❌ REQUIRED (not optional!)
    radiation_safe: bool        # ❌ REQUIRED (not optional!)
    cooling_ok: bool            # ❌ REQUIRED (not optional!)
    power_ok: bool              # ❌ REQUIRED (not optional!)
```

**Hardware API (API_DOCUMENTATION.md line 453-463):**
All interlock fields are required booleans.

**Impact:** UI may break if it assumes optional fields but orchestrator requires them

**Fix:** Make all interlock fields required in UI types

---

### 6. **Device Status Structure Mismatch**

**Problem:** UI expects different structure than orchestrator provides.

**UI Expected for `/api/state`:**
```typescript
interface CompactDeviceStatus {
  powered: boolean;
  status: string;  // Just powered + status
}

interface CompactDevices {
  pdu: CompactDeviceStatus;
  gpio: CompactDeviceStatus;
  detector: CompactDeviceStatus;
  motion: CompactDeviceStatus;
}
```

**Orchestrator `/api/health` Returns:**
```python
# From rest_server.py lines 320-344
pdu: {
  powered: bool,
  status: str,
  outputs: dict  # ❌ Extra field UI doesn't expect
}
gpio: {
  powered: bool,
  status: str
}
detector: {
  powered: bool,
  status: str,
  temperature: Optional[float],  # ❌ Extra field UI doesn't expect
  voltage: Optional[float]        # ❌ Extra field UI doesn't expect
}
motion: {
  powered: bool,
  status: str,
  is_homed: Optional[bool],      # ❌ Extra field UI doesn't expect
  position: Optional[float]       # ❌ Extra field UI doesn't expect
}
```

**Impact:** 
- `/api/state` endpoint is missing entirely
- `/api/health` has extra fields that UI's "compact" types don't account for

**Fix:** Either:
1. Create `/api/state` with truly compact data, OR
2. Update UI types to handle `/api/health` response structure

---

### 7. **Detailed Status Response Mismatches**

**Problem:** UI detailed types don't match orchestrator's `/api/health` response.

**UI Type:**
```typescript
interface DetailedDetectorStatus {
  powered: boolean;
  status: DetectorDeviceStatus;
  initialized: boolean;          // ✅ Good
  uptime_seconds: number;        // ❌ Orchestrator doesn't send this
  temperature: number | null;
  voltage: number | null;
  total_exposures: number;       // ❌ Orchestrator doesn't send this
  last_exposure_time_ms: number | null;  // ❌ Orchestrator doesn't send this
}
```

**Orchestrator Returns (rest_server.py lines 329-334):**
```python
detector: {
  powered: bool,
  status: str,
  temperature: Optional[float],  # ✅ Matches
  voltage: Optional[float]       # ✅ Matches
  # Missing: initialized, uptime_seconds, total_exposures, last_exposure_time_ms
}
```

**Hardware API Has (API_DOCUMENTATION.md line 368-376):**
```protobuf
message DetectorHealth {
  bool powered = 1;
  float temperature = 2;
  float voltage = 3;
  DetectorStatus status = 4;
  uint32 last_exposure_time = 5;     // ✅ Available in hardware
  uint64 total_exposures = 6;        // ✅ Available in hardware
  uint64 uptime_seconds = 7;         // ✅ Available in hardware
}
```

**Impact:** Orchestrator isn't querying all available hardware data

**Fix:** Orchestrator should call `GetDetectorHealth` to get full data, not just basic state

---

### 8. **Motion Status Fields Missing**

**Problem:** Similar to detector, motion status is incomplete.

**UI Type:**
```typescript
interface DetailedMotionStatus {
  powered: boolean;
  status: MotionDeviceStatus;
  initialized: boolean;          // ❌ Orchestrator doesn't send
  uptime_seconds: number;        // ❌ Orchestrator doesn't send
  is_homed: boolean;
  position: number | null;
  target_position: number | null;  // ❌ Orchestrator doesn't send
  total_moves: number;            // ❌ Orchestrator doesn't send
}
```

**Hardware API Has (API_DOCUMENTATION.md line 400-408):**
```protobuf
message MotionHealth {
  bool powered = 1;
  MotionStatus status = 2;
  optional double position = 3;
  optional double target_position = 4;  // ✅ Available
  bool is_homed = 5;
  uint64 total_moves = 6;              // ✅ Available
  uint64 uptime_seconds = 7;           // ✅ Available
}
```

**Fix:** Orchestrator should call `GetMotionHealth` for complete data

---

### 9. **GPIO Detailed Status Structure Mismatch**

**Problem:** UI expects nested structure that orchestrator may not provide.

**UI Type:**
```typescript
interface DetailedGpioStatus {
  powered: boolean;
  status: GpioDeviceStatus;
  uptime_seconds: number;
  inputs: {
    key_switch: boolean;
    activation_button: boolean;
    activation_remaining_secs: number | null;
    emergency_stop: boolean;
    door_closed: boolean;
    radiation_safe: boolean;
    cooling_ok: boolean;
    power_ok: boolean;
  };
  outputs: {
    main_led: string;
    radiation_led: string;
  };
  interlocks: {
    overall_safe: boolean;
    emergency_stop: boolean;
    door_closed: boolean;
    radiation_safe: boolean;
    cooling_ok: boolean;
    power_ok: boolean;
  };
}
```

**Hardware API Has (API_DOCUMENTATION.md line 646-654):**
```protobuf
message GpioStateResponse {
  bool powered = 1;
  bool key_switch_on = 2;
  bool activation_button_active = 3;
  optional uint32 activation_remaining_secs = 4;
  InterlockStatus interlocks = 5;
  string main_led = 6;
  string radiation_led = 7;
}
```

**Issues:**
- Hardware doesn't have nested `inputs` and `outputs` objects
- Fields are flat at top level
- Hardware doesn't have `uptime_seconds` for GPIO

**Fix:** Update UI types to match flat structure from hardware

---

### 10. **Calibration Response Structure**

**Problem:** UI expects different fields than what's available.

**UI Type:**
```typescript
interface CalibrationStatus {
  id: string;                    // ❌ Hardware doesn't track calibration ID
  timestamp: Date;
  valid: boolean;
  expiresAt: Date;              // ❌ Field name mismatch (expires_at)
  distanceCheck: boolean;       // ❌ Field name mismatch (distance_check)
  snrThreshold: number;         // ❌ Field name mismatch (snr_threshold)
  parameters: Record<string, number>;  // ❌ Not in hardware API
}
```

**Orchestrator Model (models.py line 91-97):**
```python
class CalibrationStatus(BaseModel):
    id: Optional[str]
    timestamp: Optional[str]
    valid: bool
    expires_at: Optional[str]         # ✅ Snake case
    distance_check: Optional[bool]    # ✅ Snake case
    snr_threshold: Optional[float]    # ✅ Snake case
```

**Issues:**
- UI uses camelCase, orchestrator uses snake_case
- UI expects `Date` objects, orchestrator sends ISO strings
- UI has extra `parameters` field

**Fix:** Update UI to use snake_case OR configure JSON serializer to convert

---

### 11. **Measurement Response Structure**

**Problem:** Field name and type mismatches.

**UI Type:**
```typescript
interface MeasurementParams {
  sampleId: string;           // ❌ camelCase
  exposureDuration: number;   // ❌ camelCase, in seconds
  beamIntensity: number;      // ❌ Not in API
  operatorId: string;         // ❌ camelCase
}
```

**Orchestrator Model (models.py line 49-53):**
```python
class MeasurementStartRequest(BaseModel):
    patient_id: str              # ❌ UI doesn't have this
    sample_id: str               # ✅ But snake_case
    exposure_duration: int       # ✅ But snake_case, in milliseconds
    notes: Optional[str]         # ❌ UI doesn't have this
```

**Issues:**
- Case convention mismatch (camelCase vs snake_case)
- UI has `beamIntensity` which hardware doesn't support
- Orchestrator has `patient_id` which UI doesn't send
- Unit mismatch: UI says "seconds", API expects milliseconds

**Fix:** Align field names and ensure UI sends milliseconds

---

### 12. **WebSocket Event Type Mismatch**

**Problem:** UI expects different event type than orchestrator sends.

**UI WebSocket Handler (websocket.ts line 3):**
```typescript
type EventType = 
  | 'system_health' 
  | 'measurement_update' 
  | 'safety_alert' 
  | 'calibration_status' 
  | 'gpio_update' 
  | 'interlock_change'      // ❌ Not sent by orchestrator
  | 'hardware_init' 
  | 'hardware_stop' 
  | 'measurement_start' 
  | 'measurement_stop' 
  | 'state_change';
```

**Orchestrator Sends (rest_server.py line 78):**
```python
event_data = {
    "type": "state_change",      # ✅ Matches
    "component": component,
    "change_type": change_type,  # Uses change_type, not separate event type
    # ...
}
```

**Hardware Change Types (from state monitor):**
- `INTERLOCK_CHANGED`
- `KEY_SWITCH_CHANGED`
- `ENABLE_BUTTON_ACTIVATED`
- `ENABLE_BUTTON_DEACTIVATED`

**Impact:** UI expects `interlock_change` event type, but orchestrator sends `state_change` with `change_type: "INTERLOCK_CHANGED"`

**Fix:** Update UI to handle `state_change` events with `change_type` field

---

## Summary of Required Fixes

### Orchestrator Changes (Priority 1)
1. ✅ **Add `/api/state` endpoint** - Returns compact state for polling
2. ✅ **Query full device health** - Call `GetDetectorHealth`, `GetMotionHealth` instead of just state
3. ✅ **Return all interlock fields as required** - Don't make them optional

### UI Type Changes (Priority 1)
4. ✅ **Remove extra system states** - Remove `CALIBRATION`, `MAINTENANCE`, `LOCKED` unless orchestrator adds them
5. ✅ **Fix detector/motion status enums** - Remove `INIT`, align with hardware API
6. ✅ **Make interlock fields required** - Remove optional `?` from critical safety fields
7. ✅ **Fix field name casing** - Use snake_case to match orchestrator OR configure serializer
8. ✅ **Fix measurement units** - `exposureDuration` is milliseconds, not seconds
9. ✅ **Update WebSocket event handling** - Handle `state_change` with `change_type` field

### UI Type Changes (Priority 2)
10. ✅ **Update GPIO status structure** - Use flat fields, not nested `inputs`/`outputs`
11. ✅ **Update detailed device status types** - Add missing fields from hardware API
12. ✅ **Fix calibration types** - Use snake_case, handle ISO string dates

### Documentation
13. ✅ **Document case conversion strategy** - Decide on camelCase ↔ snake_case handling

---

## Recommendations

### Short-term (Quick Fixes)
1. Add `/api/state` endpoint to orchestrator that returns minimal state for UI polling
2. Make all UI interlock fields required (remove `?`)
3. Fix measurement parameter units and field names

### Medium-term (Alignment)
1. Configure JSON serializer to auto-convert between camelCase ↔ snake_case
2. Update orchestrator to query full device health from hardware server
3. Update UI enums to match hardware API exactly

### Long-term (Architecture)
1. Consider generating TypeScript types from Pydantic models automatically
2. Consider using Protocol Buffers + grpc-web directly from UI to hardware server
3. Add integration tests that validate UI ↔ orchestrator ↔ hardware data flow

---

## Testing Checklist

- [ ] UI can successfully call `/api/state` and receive valid data
- [ ] UI can successfully call `/api/health` and receive complete device info
- [ ] All interlock fields are present and non-null in UI
- [ ] Detector and motion status values match hardware enums
- [ ] WebSocket state change events are properly handled
- [ ] Measurement start request sends correct field names and units
- [ ] Calibration status displays correctly with snake_case fields
- [ ] GPIO detailed status renders without errors

---

**Generated:** 2025-10-28  
**Status:** 🔴 CRITICAL - UI will not work without these fixes
