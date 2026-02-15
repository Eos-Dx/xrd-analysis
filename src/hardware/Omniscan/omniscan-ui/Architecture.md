# Omniscan UI Architecture

## Overview
React-based web UI for the Omniscan Medical XRD Diagnostic System, designed for IEC 62304 Class B medical device compliance.

---

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Zustand** for state management
- **React Router** for navigation
- **Tailwind CSS** for styling
- **date-fns** for date formatting
- **Recharts** for data visualization

---

## Component Structure

```
src/
├── components/
│   ├── auth/           # Login and authentication
│   ├── calibration/    # Calibration workflow with QC reports
│   ├── common/         # Reusable UI components
│   ├── layout/         # App layout and navigation
│   ├── measurement/    # Measurement controls and history
│   └── status/         # System status displays
├── pages/              # Route-level pages
├── services/           # API and WebSocket clients
├── store/              # Zustand state management
├── types/              # TypeScript type definitions
└── App.tsx             # Main application
```

---

## System States

The UI reflects these operational states from the hardware server:

- `IDLE` - Ready for measurements
- `PENDING_ARMED` - Arming sequence in progress
- `RUNNING` - Measurement active
- `STOPPING` - Shutdown in progress
- `SAFE` - Safe mode (fault detected)

**Note:** States like `CALIBRATION`, `MAINTENANCE`, and `LOCKED` were removed as they don't exist in the hardware API.

---

## Device Status Enums

### Detector Status
```typescript
type DetectorDeviceStatus = 'OFF' | 'IDLE' | 'EXPOSING' | 'READING' | 'ERROR';
```

### Motion Status
```typescript
type MotionDeviceStatus = 'OFF' | 'IDLE' | 'MOVING' | 'HOMING' | 'ERROR' | 'LIMIT_HIT';
```

### GPIO Status
```typescript
type GpioDeviceStatus = 'OFF' | 'IDLE' | 'ERROR';
```

**Note:** `INIT` status was removed from all device types - hardware uses separate `initialized: boolean` fields.

---

## Data Structures

### Safety Interlocks
All fields are **required** for safety compliance:

```typescript
interface CompactInterlocks {
  overall_safe: boolean;
  key_switch: boolean;
  enable_button: boolean;
  door_closed: boolean;
  emergency_stop: boolean;
  radiation_safe: boolean;
  cooling_ok: boolean;
  power_ok: boolean;
}
```

### GPIO State (Flat Structure)
```typescript
interface DetailedGpioStatus {
  powered: boolean;
  status: GpioDeviceStatus;
  key_switch_on: boolean;
  activation_button_active: boolean;
  activation_remaining_secs: number | null;
  interlocks: CompactInterlocks;
  main_led: string;
  radiation_led: string;
}
```

### Calibration QC Report
```typescript
interface CalibrationQcReport {
  success: boolean;
  calibration_id: string;
  timestamp: string;
  calibrant_material: string;
  overall_pass: boolean;
  qc_checks: {
    total_intensity: CalibrationQcCheck;
    goodness_of_fit: CalibrationQcCheck;
    snr: CalibrationQcCheck;
    ring_quality: CalibrationQcCheck;
    poni: PoniCalibrationResult;
  };
  formatted_report: string;
}
```

---

## State Management (Zustand)

### Core State
```typescript
interface AppState {
  // Authentication
  user: User | null;
  sessionId: string | null;
  
  // System State
  systemState: SystemState;
  devices: CompactDevices;
  interlocks: CompactInterlocks;
  
  // Calibration
  calibrationStatus: CalibrationStatus;
  calibrationRunning: boolean;
  runningCalibrationId: string | null;
  latestCalibration: CalibrationQcReport | null;
  
  // Measurements
  activeMeasurement: MeasurementState | null;
  
  // WebSocket
  wsConnected: boolean;
  
  // UI State
  notifications: Notification[];
}
```

### Key Actions
- `setUser()` - Set authenticated user
- `setSystemState()` - Update system state
- `setCalibrationRunning()` - Track async calibration
- `setLatestCalibration()` - Store QC report
- `addNotification()` - Display user notifications
- `setWebSocketConnected()` - Track connection status

---

## API Client Architecture

### Base API Client
```typescript
class ApiClient {
  private baseUrl: string;
  private sessionId: string | null;
  
  async request<T>(endpoint: string, options?: RequestInit): Promise<ApiResponse<T>>;
}
```

### Response Normalization
The API client normalizes responses to handle both:
1. Direct orchestrator responses: `{success, field1, field2}`
2. Wrapped responses: `{success, data: {field1, field2}}`

### Authentication
- Session-based authentication with `x-session-id` header
- Automatic session management
- Session timeout handling

---

## WebSocket Architecture

### Connection
```typescript
class WebSocketService {
  private ws: WebSocket | null;
  private listeners: Map<EventType, Set<EventHandler>>;
  
  connect(url: string): void;
  subscribe(eventType: EventType, handler: EventHandler): void;
  unsubscribe(eventType: EventType, handler: EventHandler): void;
}
```

### Event Types
```typescript
type EventType = 
  | 'system_health'
  | 'measurement_update'
  | 'safety_alert'
  | 'calibration_status'
  | 'calibration_start'      // Async calibration started
  | 'calibration_complete'   // Async calibration finished
  | 'gpio_update'
  | 'hardware_init'
  | 'hardware_stop'
  | 'measurement_start'
  | 'measurement_stop'
  | 'state_change'
  | 'safety_status'
  | 'connection'
  | 'echo';
```

### State Change Events
Hardware server sends `state_change` events with `change_type`:
- `INTERLOCK_CHANGED`
- `KEY_SWITCH_CHANGED`
- `ENABLE_BUTTON_ACTIVATED`
- `ENABLE_BUTTON_DEACTIVATED`

---

## Async Calibration Architecture

### Flow
1. User clicks "Start Daily Calibration"
2. `POST /api/calibration/start` returns immediately with `calibration_id`
3. Orchestrator runs calibration in background with 13s timeout protection
4. WebSocket broadcasts `calibration_start` event
5. UI shows progress indicator
6. User can navigate away
7. WebSocket broadcasts `calibration_complete` event
8. UI fetches full report and displays notification

### Timeout Protection
- Background task uses `asyncio.wait_for()` with 13-second timeout
- Prevents hanging if hardware server becomes unresponsive
- Automatically broadcasts error on timeout
- Task cleanup allows new calibrations to start

### WebSocket Events

**calibration_start:**
```json
{
  "type": "calibration_start",
  "data": {
    "calibration_id": "UUID",
    "user": "OP001",
    "status": "running"
  },
  "timestamp": "2025-01-03T19:00:00.000Z"
}
```

**calibration_complete:**
```json
{
  "type": "calibration_complete",
  "data": {
    "calibration_id": "UUID",
    "success": true,
    "overall_pass": true,
    "error": null,
    "user": "OP001"
  },
  "timestamp": "2025-01-03T19:00:05.123Z"
}
```

---

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout

### System State
- `GET /api/state` - Compact system state for polling
- `GET /api/health` - Detailed health information

### Patients
- `POST /api/patients` - Register new patient
- `GET /api/patients/search?mrn=...` - Search by MRN
- `GET /api/patients/{patient_id}` - Load by ID

### Measurements
- `POST /api/measurements/start` - Start measurement (returns immediately)
- `POST /api/measurements/stop` - Stop active measurement
- `GET /api/measurements?limit=50` - Get measurement history

### Calibration
- `POST /api/calibration/start` - Start async calibration (returns immediately)
- `GET /api/calibration/latest` - Get most recent calibration report
- `GET /api/calibration/history?hours=24&limit=50` - Get calibration history
- `GET /api/calibration/running` - Check if calibration is running

### Hardware Control
- `POST /api/hardware/{device}/init` - Initialize device
- `POST /api/hardware/{device}/stop` - Stop device

---

## Configuration

### API Proxy (vite.config.ts)
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8081',
    changeOrigin: true,
  },
  '/ws': {
    target: 'ws://localhost:8081',
    ws: true,
  },
}
```

### Port Configuration
| Component | Port | URL |
|-----------|------|-----|
| **UI (Vite Dev Server)** | 3000 | http://localhost:3000 |
| **Orchestrator REST API** | 8081 | http://localhost:8081 |
| **Hardware Server gRPC** | 50051 | localhost:50051 |

---

## Data Flow Diagrams

### Measurement Flow
```
┌─────────────┐
│    UI       │ POST /api/measurements/start
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Orchestrator│ gRPC: StartMeasurement
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Hardware   │ Run measurement
│   Server    │
└──────┬──────┘
       │ WebSocket: measurement_update
       ▼
┌─────────────┐
│    UI       │ Display progress
└─────────────┘
```

### Calibration Flow
```
┌─────────────┐
│    UI       │ POST /api/calibration/start
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Orchestrator│ Create background task
└──────┬──────┘
       │ Return immediately
       ▼
┌─────────────┐
│    UI       │ Show progress indicator
└─────────────┘
       │
       │ WebSocket: calibration_start
       ▼
┌─────────────┐
│ Orchestrator│ Run calibration (async)
└──────┬──────┘
       │ gRPC: CalibrateDetector
       ▼
┌─────────────┐
│  Hardware   │ Perform calibration
│   Server    │
└──────┬──────┘
       │ Calibration data
       ▼
┌─────────────┐
│ Orchestrator│ Process QC checks
│             │ Run PONI calibration
│             │ Generate report
└──────┬──────┘
       │ WebSocket: calibration_complete
       ▼
┌─────────────┐
│    UI       │ Fetch report & display
└─────────────┘
```

---

## Security Architecture

- **Session Management**: Session IDs with automatic timeout
- **Role-Based Access Control**: Different views for Operators, Engineers, Administrators
- **TLS Encryption**: All API traffic encrypted in production
- **WebSocket Authentication**: Uses same session context as REST API
- **Audit Trail**: Complete logging of user actions
- **Data Integrity**: Immediate persistence, offline buffering support

---

## Error Handling

### API Client
- Network error handling with retry logic
- Graceful degradation for missing endpoints
- User-friendly error messages

### WebSocket
- Automatic reconnection on disconnect
- Connection status monitoring
- Event queue for missed messages

### UI Components
- Loading states for async operations
- Error boundaries for component failures
- User notifications for important events

---

## Performance Considerations

- **Lazy Loading**: Code splitting for routes
- **Memoization**: React.memo for expensive components
- **Debouncing**: API polling and search inputs
- **WebSocket**: Efficient real-time updates vs polling
- **Batch Updates**: Multiple state changes in single render

---

## Type Safety

All components use TypeScript with strict mode:
- No implicit `any`
- Strict null checks
- Type inference for API responses
- Interface-based contracts between layers

---

## Testing Strategy

### Manual Testing
- Authentication flow
- Measurement workflow
- Calibration workflow
- Real-time updates
- Error handling

### Integration Testing
- UI ↔ Orchestrator communication
- WebSocket event handling
- State synchronization

---

## Compliance & Standards

- **IEC 62304 Class B**: Medical device software lifecycle
- **HIPAA**: Patient data handling
- **FDA Cybersecurity**: Security guidelines compliance
- **Audit Trail**: Complete traceability
- **Data Integrity**: Measurement → operator → calibration linkage

---

**Last Updated:** 2025-11-04
