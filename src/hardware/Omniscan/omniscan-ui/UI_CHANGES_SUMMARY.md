# UI Changes Summary - API Alignment

## Date: 2025-10-28

## Overview
Updated UI types and components to align with the hardware server gRPC API and orchestrator REST API specifications.

---

## Changes Made

### 1. **System State Enum** (`src/types/index.ts`)
**Removed non-existent states:**
- ❌ `CALIBRATION`
- ❌ `MAINTENANCE`  
- ❌ `LOCKED`

**Now aligned with hardware API:**
```typescript
type SystemState = 'IDLE' | 'PENDING_ARMED' | 'RUNNING' | 'STOPPING' | 'SAFE';
```

---

### 2. **Detector Status Enum** (`src/types/index.ts`)
**Removed:**
- ❌ `INIT` (hardware uses separate `initialized: boolean` field)

**Now aligned with hardware API:**
```typescript
type DetectorDeviceStatus = 'OFF' | 'IDLE' | 'EXPOSING' | 'READING' | 'ERROR';
```

---

### 3. **Motion Status Enum** (`src/types/index.ts`)
**Removed:**
- ❌ `INIT` (hardware uses separate `initialized: boolean` field)

**Now aligned with hardware API:**
```typescript
type MotionDeviceStatus = 'OFF' | 'IDLE' | 'MOVING' | 'HOMING' | 'ERROR' | 'LIMIT_HIT';
```

---

### 4. **Interlock Structure** (`src/types/index.ts`)
**Made all fields required for safety:**
```typescript
interface CompactInterlocks {
  overall_safe: boolean;       // ✅ Required
  key_switch: boolean;         // ✅ Required  
  enable_button: boolean;      // ✅ Required
  door_closed: boolean;        // ✅ Now required (was optional)
  emergency_stop: boolean;     // ✅ Now required (was optional)
  radiation_safe: boolean;     // ✅ Now required (was optional)
  cooling_ok: boolean;         // ✅ Now required (was optional)
  power_ok: boolean;           // ✅ Now required (was optional)
}
```

---

### 5. **GPIO Detailed Status** (`src/types/index.ts`)
**Changed from nested structure to flat structure:**

**Before:**
```typescript
interface DetailedGpioStatus {
  inputs: { key_switch, activation_button, ... };
  outputs: { main_led, radiation_led };
}
```

**After (aligned with hardware `GpioStateResponse`):**
```typescript
interface DetailedGpioStatus {
  powered: boolean;
  status: GpioDeviceStatus;
  key_switch_on: boolean;              // ✅ Flat field
  activation_button_active: boolean;   // ✅ Flat field
  activation_remaining_secs: number | null;
  interlocks: { ... };
  main_led: string;                    // ✅ Flat field
  radiation_led: string;               // ✅ Flat field
}
```

---

### 6. **Measurement Units Comment** (`src/types/index.ts`)
**Fixed documentation:**
```typescript
interface MeasurementParams {
  exposureDuration: number; // milliseconds (was incorrectly documented as "seconds")
}
```

---

### 7. **Optional Device Health Fields** (`src/types/index.ts`)
**Made certain fields optional** until orchestrator implements full health queries:

```typescript
interface DetailedDetectorStatus {
  uptime_seconds?: number;           // Optional until orchestrator queries GetDetectorHealth
  total_exposures?: number;          // Optional
  last_exposure_time_ms?: number;    // Optional
}

interface DetailedMotionStatus {
  uptime_seconds?: number;           // Optional until orchestrator queries GetMotionHealth
  target_position?: number;          // Optional
  total_moves?: number;              // Optional
}
```

---

### 8. **WebSocket Event Types** (`src/services/websocket.ts`)
**Added missing event types and state change types:**

```typescript
type EventType = 
  | 'system_health' 
  | 'measurement_update' 
  | 'safety_alert' 
  | 'calibration_status' 
  | 'gpio_update' 
  | 'hardware_init' 
  | 'hardware_stop' 
  | 'measurement_start' 
  | 'measurement_stop' 
  | 'state_change'      // ✅ Includes interlock changes via change_type
  | 'safety_status'     // ✅ Added
  | 'connection'        // ✅ Added
  | 'echo';             // ✅ Added

// State change types from hardware server
type StateChangeType = 
  | 'INTERLOCK_CHANGED'
  | 'KEY_SWITCH_CHANGED'
  | 'ENABLE_BUTTON_ACTIVATED'
  | 'ENABLE_BUTTON_DEACTIVATED'
  | string;
```

**Updated WebSocketEvent interface:**
```typescript
interface WebSocketEvent {
  type: EventType;
  data?: unknown;
  timestamp: string;
  component?: string;              // ✅ For state_change events
  change_type?: StateChangeType;   // ✅ For state_change events
  interlocks?: any;                // ✅ For interlock updates
  key_switch_on?: boolean;
  activation_button_active?: boolean;
  activation_remaining_secs?: number;
}
```

---

### 9. **DetailedDiagnosticsPanel Component** (`src/components/hardware/DetailedDiagnosticsPanel.tsx`)
**Updated GPIO section to use flat structure:**

**Before:**
```tsx
<Badge>{healthData.gpio.inputs.key_switch ? 'ON' : 'OFF'}</Badge>
<span>{healthData.gpio.outputs.main_led}</span>
```

**After:**
```tsx
<Badge>{healthData.gpio.key_switch_on ? 'ON' : 'OFF'}</Badge>
<span>{healthData.gpio.main_led}</span>
```

**Added activation timer display:**
```tsx
{healthData.gpio.activation_remaining_secs > 0 && (
  <Badge variant="warning">
    {healthData.gpio.activation_remaining_secs}s remaining
  </Badge>
)}
```

**Removed GPIO uptime** (not provided by hardware)

---

### 10. **HardwareStatusPanel Component** (`src/components/hardware/HardwareStatusPanel.tsx`)
**Removed `INIT` status handling:**

```typescript
// Before
if (status === 'INIT' || status === 'HOMING') return 'yellow';

// After  
if (status === 'HOMING') return 'yellow';
```

---

### 11. **SystemStatusHeader Component** (`src/components/status/SystemStatusHeader.tsx`)
**Removed references to non-existent states:**

**Before:**
```typescript
case 'CALIBRATION': return 'Calibration';
case 'MAINTENANCE': return 'Maintenance';
case 'LOCKED': return 'Calibration Required';
```

**After:**
```typescript
// Removed - these states don't exist in hardware API
```

**Updated state color mapping:**
```typescript
case 'PENDING_ARMED': return 'warning';
case 'STOPPING': return 'warning';
```

---

## Remaining Work (For Orchestrator)

The following must be implemented in the orchestrator to fully support the UI:

### Critical (P1)
1. ✅ **Add `/api/state` endpoint** - UI expects this for lightweight polling
2. ✅ **Query full device health** - Call `GetDetectorHealth` and `GetMotionHealth` from hardware
3. ✅ **Return complete interlock data** - Ensure all interlock fields are populated

### Important (P2)
4. ⚠️  **Field naming convention** - Decide on camelCase ↔ snake_case conversion strategy
5. ⚠️  **GPIO state response** - Return flat structure matching hardware API
6. ⚠️  **Add device health fields** - Return `uptime_seconds`, `total_exposures`, `total_moves` etc.

---

## Testing Checklist

After orchestrator implements the required endpoints:

- [ ] UI can call `/api/state` without errors
- [ ] All interlock fields are present (no null/undefined)
- [ ] Detector status values match hardware enum
- [ ] Motion status values match hardware enum
- [ ] GPIO state has flat structure (not nested)
- [ ] WebSocket `state_change` events work correctly
- [ ] Activation button timer displays correctly
- [ ] Detailed diagnostics panel renders without errors
- [ ] No console errors related to missing/undefined fields

---

## Files Modified

1. `src/types/index.ts` - Core type definitions
2. `src/services/websocket.ts` - WebSocket event types
3. `src/components/hardware/DetailedDiagnosticsPanel.tsx` - GPIO display
4. `src/components/hardware/HardwareStatusPanel.tsx` - Device status badges
5. `src/components/status/SystemStatusHeader.tsx` - System state display

---

## Breaking Changes

⚠️  **Components using the following will break until orchestrator is updated:**

1. Any code expecting `CALIBRATION`, `MAINTENANCE`, or `LOCKED` states
2. Any code expecting `INIT` detector/motion status
3. Any code accessing `gpio.inputs.*` or `gpio.outputs.*` (now flat)
4. Any code assuming interlock fields are optional

---

## Notes

- The UI is now aligned with the **hardware gRPC API** specification
- Some fields are optional (`uptime_seconds`, `total_exposures`) until orchestrator queries full health
- All critical safety fields (interlocks) are now required
- WebSocket events now handle hardware server `state_change` events with `change_type` field

---

## Build Status

✅ **TypeScript compilation successful** (vite v5.4.21, 365 modules transformed, built in 1.68s)

All TypeScript errors have been resolved. The UI now compiles without errors.

---

**Status:** ✅ UI changes complete and building successfully, waiting for orchestrator implementation
