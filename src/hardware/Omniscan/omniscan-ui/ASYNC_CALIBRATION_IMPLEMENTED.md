# Async Calibration Implementation - Complete

## Overview
Successfully implemented async calibration with WebSocket notifications in both orchestrator and UI. Calibration now runs in the background without blocking the user interface.

---

## Backend Changes (Orchestrator) ✅

### 1. Global State
**File:** `omniscan-orchestrator/src/omniscan_orchestrator/rest_server.py`

Added:
```python
# Calibration background task
active_calibration_task: Optional[asyncio.Task] = None
current_calibration_id: Optional[str] = None
```

### 2. Async Calibration Task
Created `run_calibration_async()` function that:
- Broadcasts `calibration_start` event via WebSocket
- Runs `grpc_client.calibrate_detector()` in thread with **timeout protection**
  - Uses `asyncio.wait_for()` with 13-second timeout (1.3x typical integration time)
  - Prevents hanging forever if hardware server crashes/is unresponsive
  - Automatically broadcasts error if timeout occurs
- Persists results to database
- Broadcasts `calibration_complete` event with success/error status

### 3. Modified `/api/calibration/start` Endpoint
- Returns immediately with `{success, calibration_id}`
- Prevents starting multiple calibrations
- Starts background task using `asyncio.create_task()`

### 4. New `/api/calibration/running` Endpoint
- Returns current calibration status
- Response: `{running: bool, calibration_id?: string}`

---

## Frontend Changes (UI) ✅

### 1. WebSocket Service
**File:** `src/services/websocket.ts`

Added event types:
- `calibration_start`
- `calibration_complete`

### 2. Store Updates
**File:** `src/store/useStore.ts`

Added state:
```typescript
calibrationRunning: boolean
runningCalibrationId: string | null
latestCalibration: CalibrationQcReport | null
```

Added actions:
- `setCalibrationRunning(running, id)`
- `setLatestCalibration(calibration)`

### 3. App.tsx - WebSocket Handlers
**File:** `src/App.tsx`

Added subscriptions:
- `calibration_start`: Sets running state, shows notification
- `calibration_complete`: 
  - Clears running state
  - Fetches full report on success
  - Shows success/failure notification

### 4. API Client
**File:** `src/services/api.ts`

Updated:
- `startCalibration()`: Returns `{calibration_id}` immediately
- Added `getRunningCalibration()`: Check running status

### 5. CalibrationPanel
**File:** `src/components/calibration/CalibrationPanel.tsx`

Changes:
- Removed local `isCalibrating` state (now uses store)
- Replaced `qcReport` with `latestCalibration` from store
- Added running indicator with spinner and progress message
- Button shows "Calibration Running..." when active
- User can navigate away during calibration

---

## User Experience

### Before (Blocking)
1. User clicks "Start Calibration"
2. UI freezes for 5-10 seconds
3. User cannot navigate or interact
4. Report appears when done

### After (Async)
1. User clicks "Start Calibration"
2. API returns immediately
3. Blue progress indicator appears: "Calibration in Progress"
4. User can navigate to other pages
5. Notification appears when complete: "Calibration completed successfully!"
6. Report auto-shows on calibration page

---

## WebSocket Events

### `calibration_start`
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

### `calibration_complete`
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

## Testing

### Manual Testing
```bash
# Terminal 1: Start orchestrator
cd omniscan-orchestrator
python -m omniscan_orchestrator.rest_server

# Terminal 2: WebSocket monitor
wscat -c ws://localhost:8080/ws

# Terminal 3: Start UI
cd omniscan-ui
npm run dev

# In browser:
# 1. Login
# 2. Go to Calibration page
# 3. Click "Start Daily Calibration"
# 4. Observe:
#    - API returns immediately
#    - Progress indicator appears
#    - Can navigate to other pages
#    - WebSocket events in terminal 2
#    - Notification when complete
```

### Expected Behavior
- ✅ Calibration starts immediately (< 100ms)
- ✅ UI remains responsive
- ✅ WebSocket events broadcast to all clients
- ✅ Report appears when complete
- ✅ Prevents starting multiple calibrations
- ✅ Notifications show success/failure

---

## Benefits Achieved

1. **Non-blocking UI**: User can navigate freely during calibration
2. **Real-time updates**: All connected clients see progress
3. **Better UX**: Clear progress indication with spinner
4. **Multi-user support**: All clients notified when calibration completes
5. **Error handling**: Failures broadcast immediately
6. **Consistent pattern**: Matches measurements workflow
7. **Timeout protection**: Won't hang forever if hardware server fails (13s timeout)

## Timeout Protection

**Problem**: If hardware server crashes or becomes unresponsive during calibration, the async task could hang forever.

**Solution**: `asyncio.wait_for()` with 13-second timeout (1.3x typical integration time)

```python
result = await asyncio.wait_for(
    asyncio.to_thread(
        grpc_client.calibrate_detector,
        user=user_id
    ),
    timeout=13.0  # 1.3x typical ~10s calibration
)
```

**Behavior on timeout:**
1. Task raises `asyncio.TimeoutError`
2. Exception caught and converted to error message
3. `calibration_complete` event broadcast with error
4. UI shows notification: "Calibration timed out - hardware server may be unresponsive"
5. Task cleans up and allows new calibration to start

---

## Files Modified

### Orchestrator (3 files)
1. `src/omniscan_orchestrator/rest_server.py`
   - Added async calibration task
   - Modified `/api/calibration/start`
   - Added `/api/calibration/running`

### UI (5 files)
1. `src/services/websocket.ts` - Added event types
2. `src/store/useStore.ts` - Added calibration state
3. `src/App.tsx` - Added WebSocket handlers
4. `src/services/api.ts` - Updated API methods
5. `src/components/calibration/CalibrationPanel.tsx` - UI updates

---

## Next Steps (Optional Enhancements)

1. Add progress percentage if hardware server supports it
2. Add "Cancel Calibration" button
3. Show estimated time remaining
4. Add calibration queue if multiple users try to start
5. Persist running state to DB for crash recovery
