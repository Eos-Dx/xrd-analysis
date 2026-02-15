# UI Update Guide: Radiation Safe Status Fix

**Date:** 2025-11-23  
**Issue:** Hardware server and orchestrator have been fixed to return correct `radiation_safe` status  
**Impact:** UI will now receive accurate real-time radiation safety status (no more FAULT displays)

---

## What Changed on the Backend

### Hardware Server Changes
The hardware server now uses **real-time GPIO interlock status** (10ms watchdog updates) for all gRPC services:
- ✅ `Health.GetAggregateHealth()` - Fixed
- ✅ `Safety.GetInterlockStatus()` - Fixed  
- ✅ `Safety.CheckSafetyToOperate()` - Fixed
- ✅ `StateMonitor.GetGpioState()` - Already correct

**Key point:** All services now return consistent `radiation_safe` field that reflects the **actual sensor state** from GPIO hardware.

### Orchestrator Status
The orchestrator correctly passes through the `radiation_safe` field from the hardware server to the UI in these endpoints:
- ✅ `/api/health` - Returns full health including `interlocks.radiation_safe`
- ✅ `/api/state` - Returns compact state including `interlocks.radiation_safe`

---

## What the UI Needs to Do

### ✅ **GOOD NEWS: Your UI is Already Correct!**

The UI code is **already properly configured** to handle `radiation_safe`. Here's what you have:

#### 1. **Type Definition** ✅
**File:** `src/types/index.ts` (line 25)
```typescript
export interface CompactInterlocks {
  overall_safe: boolean;
  key_switch: boolean;
  enable_button: boolean;
  door_closed: boolean;
  emergency_stop: boolean;
  cooling_ok: boolean;
  power_ok: boolean;
  radiation_safe: boolean;  // ✅ Correct field name!
}
```

#### 2. **SafetyInterlockPanel Component** ✅
**File:** `src/components/status/SafetyInterlockPanel.tsx` (line 15)
```typescript
const interlockItems = [
  { key: 'key_switch' as const, label: 'Key Switch', required: true },
  { key: 'enable_button' as const, label: 'Activation Button', required: false },
  { key: 'door_closed' as const, label: 'Door Closed', required: true },
  { key: 'emergency_stop' as const, label: 'Emergency Stop', required: true },
  { key: 'radiation_safe' as const, label: 'Radiation Safe', required: true }, // ✅ Correct!
  { key: 'cooling_ok' as const, label: 'Cooling OK', required: true },
  { key: 'power_ok' as const, label: 'Power OK', required: true },
] as const;
```

The component correctly:
- Maps `radiation_safe` boolean to "Radiation Safe" label
- Shows Badge with `'success'` (OK) when `true`, `'error'` (FAULT) when `false`
- Displays as a **required** safety interlock

#### 3. **API Service** ✅
**File:** `src/services/api.ts` (line 134)
```typescript
async getSystemState(): Promise<ApiResponse<SystemStateResponse>> {
  const response = await this.request<any>('/state');
  
  // Transform orchestrator format to UI format
  if (response.success && response.data) {
    const data = response.data;
    const transformed: SystemStateResponse = {
      system_state: data.state || data.system_state,
      devices: data.devices || {},
      interlocks: data.interlocks || {}, // ✅ Passes through all interlock fields
      timestamp: data.timestamp || new Date().toISOString()
    };
    return { success: true, data: transformed };
  }
  
  return response;
}
```

---

## What You Need to Test

### 1. **Restart the Orchestrator**
The orchestrator needs to be restarted to pick up the hardware server's fixes:

```bash
# In the orchestrator terminal
# Stop current process (Ctrl+C)
# Restart with:
python -m omniscan_orchestrator
```

### 2. **Restart Your Dev Server** (if needed)
If you have the UI dev server running:

```bash
# In the UI terminal
npm run dev
# or
yarn dev
```

### 3. **Verify in the UI**

Open your browser and check:

#### A. **System Status Header**
**Component:** `SystemStatusHeader.tsx`
- Should show "Safety: All OK" (green badge) when all interlocks are satisfied
- Should show "Safety: Fault" (red badge) if any interlock fails

#### B. **Safety Interlock Panel**
**Component:** `SafetyInterlockPanel.tsx`
- Should display 7 interlock items including "Radiation Safe"
- **"Radiation Safe"** should show:
  - ✅ **Badge: "OK" (green)** when beam is closed/blocked (radiation_safe = true)
  - ❌ **Badge: "FAULT" (red)** when beam is open (radiation_safe = false)
- Overall status banner at top should be green when all safe

#### C. **Detailed Diagnostics Panel** (if you have one)
**Component:** `DetailedDiagnosticsPanel.tsx`
- Should show complete GPIO state
- Verify `radiation_safe` displays correctly

### 4. **Test Real-Time Updates**

If you have WebSocket connections set up:

1. **Open the UI**
2. **Toggle the radiation safe state** in the hardware server (via GPIO)
3. **Verify the UI updates** within ~10ms-100ms to reflect the change
4. Check that:
   - Badge color changes (green ↔ red)
   - Overall safety status updates
   - No "FAULT" appears when radiation is actually safe

---

## Expected Behavior

### When Beam is Closed (Radiation Safe = true)
```
┌─────────────────────────────────────┐
│ Safety Interlocks                   │
├─────────────────────────────────────┤
│ ● System Safe - Ready for Operation │ ← Green banner
├─────────────────────────────────────┤
│ Key Switch          [OK]            │ ← Green badge
│ Activation Button   [N/A]           │
│ Door Closed         [OK]            │
│ Emergency Stop      [OK]            │
│ Radiation Safe      [OK]   ✅       │ ← Green badge (FIXED!)
│ Cooling OK          [OK]            │
│ Power OK            [OK]            │
└─────────────────────────────────────┘
```

### When Beam is Open (Radiation Safe = false)
```
┌─────────────────────────────────────┐
│ Safety Interlocks                   │
├─────────────────────────────────────┤
│ ● System Not Safe - Check Interlocks│ ← Red banner
├─────────────────────────────────────┤
│ Key Switch          [OK]            │
│ Activation Button   [N/A]           │
│ Door Closed         [OK]            │
│ Emergency Stop      [OK]            │
│ Radiation Safe      [FAULT]  ❌     │ ← Red badge
│ Cooling OK          [OK]            │
│ Power OK            [OK]            │
└─────────────────────────────────────┘
```

---

## Troubleshooting

### Issue 1: Still seeing FAULT for Radiation Safe

**Check:**
1. Is the orchestrator restarted?
   ```bash
   # Check orchestrator logs for "Connected to hardware server"
   ```

2. Is the hardware server running with updated code?
   ```bash
   # In hw-server directory
   cargo build
   ./start-server.bat
   ```

3. Check browser console for API response:
   ```javascript
   // In browser console
   fetch('/api/state')
     .then(r => r.json())
     .then(d => console.log('Interlocks:', d.interlocks))
   ```

4. Expected response:
   ```json
   {
     "interlocks": {
       "overall_safe": true,
       "key_switch": true,
       "enable_button": false,
       "door_closed": true,
       "emergency_stop": true,
       "radiation_safe": true,  // ← Should be true when safe
       "cooling_ok": true,
       "power_ok": true
     }
   }
   ```

### Issue 2: Field is undefined/null

**Check:**
```typescript
// In browser console, check the store state:
useStore.getState().systemState?.interlocks?.radiation_safe
// Should return boolean (true/false), not undefined
```

If `undefined`, the orchestrator might not be passing the field. Check orchestrator logs.

### Issue 3: WebSocket not updating

**Check:**
1. WebSocket connection status in browser DevTools → Network → WS
2. Look for `state_change` events with component `"GPIO"`
3. Verify event handler updates the store's `systemState`

---

## Summary

✅ **Your UI code is already correct!**  
✅ **No UI changes needed**  
✅ **Just restart orchestrator to get the fix**  
✅ **Test to verify the status displays correctly**

The `radiation_safe` field will now:
- Always be present (boolean, never undefined)
- Update in real-time (10ms latency from hardware)
- Show correct status (OK when safe, FAULT when unsafe)
- Display as "Radiation Safe" with green/red badge

---

## Contact

If you see any issues after restarting the orchestrator:
1. Check orchestrator logs for hardware server connection errors
2. Check browser console for API response structure
3. Verify `/api/state` or `/api/health` returns `interlocks.radiation_safe` as boolean
