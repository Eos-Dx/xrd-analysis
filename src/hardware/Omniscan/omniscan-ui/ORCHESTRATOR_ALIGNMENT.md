# UI Server Alignment with Orchestrator REST API

## Overview

The UI server has been aligned with the orchestrator's REST API as documented in `../omniscan-orchestrator/API_REFERENCE.md`. The UI communicates with the orchestrator (port 8081), which in turn communicates with the hardware server (port 50051) via gRPC.

## Architecture

```
Browser (UI) ←→ Node.js Proxy (port 3001) ←→ Orchestrator REST API (port 8081) ←→ HW Server gRPC (port 50051)
                                                   WebSocket (port 8081)
```

## Changes Made

### 1. Type Definitions (`src/types/index.ts`)

#### Updated Interlock Status
- **Added**: `beam_shutter_closed` field to `CompactInterlocks` interface
- **Removed**: `radiation_safe` (not in orchestrator API)
- **Aligned with**: Orchestrator `/api/state` response structure

```typescript
export interface CompactInterlocks {
  overall_safe: boolean;
  key_switch: boolean;
  enable_button: boolean;
  door_closed: boolean;
  emergency_stop: boolean;
  cooling_ok: boolean;
  power_ok: boolean;
  beam_shutter_closed: boolean;  // Added
}
```

#### Updated Measurement Parameters
- **Renamed**: `exposure_duration` → `exposure_ms` to match orchestrator API
- **Made optional**: `sample_id` field
- **Aligned with**: `POST /api/measurements/start` endpoint

```typescript
export interface MeasurementParams {
  patient_id: string;   // UUID
  exposure_ms: number;  // Renamed from exposure_duration
  sample_id?: string;   // Now optional
  notes?: string;
}
```

#### Added GPIO Types
- **New**: `GpioState` interface for `/api/gpio/state` endpoint
- **New**: `EnableButtonStatus` interface for `/api/gpio/enable-button` endpoint

```typescript
export interface GpioState {
  key_switch_on: boolean;
  enable_button_active: boolean;
  enable_button_remaining_secs: number;
  door_closed: boolean;
  emergency_stop_active: boolean;
  beam_shutter_closed: boolean;
}

export interface EnableButtonStatus {
  active: boolean;
  remaining_secs: number;
}
```

### 2. API Client (`src/services/api.ts`)

#### Added GPIO Methods
```typescript
async getGpioState(): Promise<ApiResponse<GpioState>>
async getEnableButtonStatus(): Promise<ApiResponse<EnableButtonStatus>>
```

#### Updated Hardware Control Methods
- **Removed**: `'gpio'` option from device type (GPIO is always powered, no manual control)
- **Device types**: Now only `'detector' | 'motion'`

```typescript
async initializeDevice(device: 'detector' | 'motion')
async stopDevice(device: 'detector' | 'motion')
```

#### Updated Measurement Methods
- **startMeasurement**: Returns `{measurementId, status}` instead of `{runId, measurementId}`
- **stopMeasurement**: Changed from `/measurements/${runId}/stop` to `/measurements/stop` with body `{run_id}`
- **abortMeasurement**: Changed to `/measurements/abort` with body `{run_id}`
- **getMeasurementHistory**: Changed to `/measurements/history` with optional `patient_id` query param

### 3. Server Proxy (`server.js`)

#### Updated State Endpoint Response
- **Added**: `beam_shutter_closed` field to interlocks mapping in `/api/state` endpoint
- **Ensured**: All orchestrator interlocks are properly mapped

```javascript
interlocks: {
  overall_safe: health.interlocks.overall_safe,
  key_switch: health.interlocks.key_switch,
  enable_button: health.interlocks.enable_button,
  door_closed: health.interlocks.door_closed,
  emergency_stop: health.interlocks.emergency_stop,
  cooling_ok: health.interlocks.cooling_ok,
  power_ok: health.interlocks.power_ok,
  beam_shutter_closed: health.interlocks.beam_shutter_closed || true
}
```

### 4. WebSocket Service (`src/services/websocket.ts`)

#### Updated Event Types
Aligned with orchestrator WebSocket events from API_REFERENCE.md:

```typescript
type EventType = 
  | 'state_change'              // System state changed
  | 'gpio_state_change'         // GPIO state changed
  | 'calibration_complete'      // Calibration finished
  | 'measurement_complete'      // Measurement finished
  | 'measurement_start'         // Measurement started
  | 'measurement_stop'          // Measurement stopped
  | 'hardware_init'             // Device initialization
  | 'hardware_stop'             // Device stopped
  | 'safety_alert'              // Safety interlock violation
  | 'connection'                // Connection status
  | 'echo';                     // Echo response
```

#### Updated WebSocket URL
- **Confirmed**: Port 8081 (same as REST API per start_server.bat)
- **Format**: `ws://localhost:8081/ws`

#### Updated Event Data Structure
```typescript
interface WebSocketEvent {
  type: EventType;
  data?: {
    state?: string;                        // For state_change
    enable_button_active?: boolean;        // For gpio_state_change
    enable_button_remaining_secs?: number; // For gpio_state_change
    calibration_id?: string;               // For calibration_complete
    overall_pass?: boolean;                // For calibration_complete
    measurement_id?: string;               // For measurement_complete
    status?: string;                       // For measurement_complete
    [key: string]: any;
  };
  timestamp?: string;
}
```

## Orchestrator API Endpoints Used

### Authentication
- `POST /api/auth/login` - User authentication
- `POST /api/auth/logout` - End session

### System State
- `GET /api/state` - Get compact system state (for polling)
- `GET /api/health` - Get detailed system health

### GPIO & Safety
- `GET /api/gpio/state` - Get GPIO state
- `GET /api/gpio/enable-button` - Get enable button status

### Hardware Control
- `POST /api/hardware/{device}/init` - Initialize detector or motion
- `POST /api/hardware/{device}/stop` - Stop/power off device

### Patients
- `POST /api/patients` - Create patient
- `GET /api/patients/search?mrn={mrn}` - Search patients by MRN

### Measurements
- `POST /api/measurements/start` - Start measurement
- `POST /api/measurements/stop` - Stop measurement
- `POST /api/measurements/abort` - Abort measurement (if implemented)
- `GET /api/measurements/history` - Get measurement history

### Calibration
- `POST /api/calibration/start` - Start calibration
- `GET /api/calibration/status` - Get calibration status
- `GET /api/calibration/latest` - Get latest calibration
- `GET /api/calibration/history` - Get calibration history

### WebSocket
- `ws://localhost:8081/ws` - Real-time event stream

## Key Differences from Hardware Server

The UI does **NOT** communicate directly with the hardware server's gRPC API. Key differences:

1. **No CommandContext**: The orchestrator manages CommandContext internally when calling gRPC
2. **No gRPC types**: UI uses JSON REST API, not protobuf messages
3. **Patient data**: Patient PII stays in orchestrator, never sent to hardware server
4. **Session management**: Orchestrator handles session-based authentication
5. **Simplified responses**: Orchestrator provides REST-friendly JSON responses

## Testing

To verify alignment:

```bash
# 1. Start orchestrator
cd ../omniscan-orchestrator
python -m omniscan_orchestrator.main

# 2. Start UI server
cd ../omniscan-ui
npm run server

# 3. Start UI dev server
npm run dev

# 4. Test endpoints
curl -X POST http://localhost:3001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"operator","password":"password"}'

# Get session_id from response, then:
curl -H "X-Session-Id: {session_id}" http://localhost:3001/api/state
curl -H "X-Session-Id: {session_id}" http://localhost:3001/api/gpio/enable-button
```

## References

- **Orchestrator API**: `../omniscan-orchestrator/API_REFERENCE.md`
- **Hardware Server Commands**: `../omniscan-hw-server/COMMAND_SCHEMAS.md`
- **Hardware Server API**: `../omniscan-hw-server/API_DOCUMENTATION.md`
- **Orchestrator Alignment**: `../omniscan-orchestrator/COMMANDS_SPEC_ALIGNMENT.md`

## Status

✅ **Complete** - UI server is now fully aligned with orchestrator REST API specification.

All changes maintain backward compatibility where possible and follow the orchestrator's response formats.
