# Async Calibration with WebSocket Notifications

## Problem
Currently, `POST /api/calibration/start` is **blocking** - it calls `grpc_client.calibrate_detector()` synchronously and waits several seconds for completion. This freezes the UI during calibration.

## Solution
Convert calibration to an **async operation** with WebSocket notifications.

---

## Backend Changes (Orchestrator)

### 1. Modify `/api/calibration/start` endpoint

**File:** `omniscan-orchestrator/src/omniscan_orchestrator/rest_server.py`

```python
# Global state for tracking active calibration
active_calibration_task: Optional[asyncio.Task] = None
current_calibration_id: Optional[str] = None

async def run_calibration_async(calibration_id: str, user_id: str):
    """Background task to run calibration and broadcast results."""
    global current_calibration_id
    
    try:
        # Broadcast start event
        await broadcast_event({
            "type": "calibration_start",
            "data": {
                "calibration_id": calibration_id,
                "user": user_id,
                "status": "running"
            }
        })
        
        # Run calibration (blocking gRPC call in thread to avoid blocking event loop)
        result = await asyncio.to_thread(
            grpc_client.calibrate_detector,
            user=user_id
        )
        
        # Persist to database
        if not result.get("error") and result.get("success"):
            try:
                qc = result.get("qc_checks", {})
                poni = qc.get("poni", {})
                db_cal = DBCalibrationRecord(
                    calibration_id=calibration_id,
                    timestamp=datetime.fromisoformat(result.get("timestamp", datetime.utcnow().isoformat())),
                    operator_id=user_id,
                    calibrant_material=result.get("calibrant_material"),
                    overall_pass=bool(result.get("overall_pass", False)),
                    # ... other fields ...
                    formatted_report=result.get("formatted_report")
                )
                db.record_calibration(db_cal)
            except Exception as e:
                print(f"Calibration persistence error: {e}")
        
        # Broadcast completion event
        await broadcast_event({
            "type": "calibration_complete",
            "data": {
                "calibration_id": calibration_id,
                "success": result.get("success", False),
                "overall_pass": result.get("overall_pass"),
                "error": result.get("error"),
                "user": user_id
            }
        })
        
    except Exception as e:
        # Broadcast error
        await broadcast_event({
            "type": "calibration_complete",
            "data": {
                "calibration_id": calibration_id,
                "success": False,
                "error": str(e),
                "user": user_id
            }
        })
    finally:
        current_calibration_id = None


@app.post("/api/calibration/start", response_model=CalibrationStartResponse)
async def start_calibration(user: Dict = Depends(get_current_user)):
    """Start daily calibration procedure (async - returns immediately)."""
    global active_calibration_task, current_calibration_id
    
    # Check if calibration already running
    if active_calibration_task and not active_calibration_task.done():
        return CalibrationStartResponse(
            success=False,
            error="Calibration already in progress"
        )
    
    calibration_id = str(uuid.uuid4())
    current_calibration_id = calibration_id
    
    # Start calibration in background
    active_calibration_task = asyncio.create_task(
        run_calibration_async(calibration_id, user["user_id"])
    )
    
    return CalibrationStartResponse(
        success=True,
        calibration_id=calibration_id
    )
```

### 2. Add calibration status endpoint

```python
@app.get("/api/calibration/running")
async def get_running_calibration(user: Dict = Depends(get_current_user)):
    """Check if calibration is currently running."""
    global active_calibration_task, current_calibration_id
    
    is_running = (
        active_calibration_task is not None 
        and not active_calibration_task.done()
    )
    
    return {
        "running": is_running,
        "calibration_id": current_calibration_id if is_running else None
    }
```

### 3. Update REST_API.md

Add to WebSocket section:
```markdown
- `calibration_start` - Calibration initiated
- `calibration_complete` - Calibration finished (success or error)
```

---

## Frontend Changes (UI)

### 1. Update WebSocket handler

**File:** `src/services/websocket.ts`

Add message handler:
```typescript
case 'calibration_start':
  store.setCalibrationRunning(true, data.calibration_id);
  break;

case 'calibration_complete':
  store.setCalibrationRunning(false, null);
  if (data.success) {
    // Fetch full report
    api.getLatestCalibration().then(result => {
      if (result.success && result.data) {
        store.setLatestCalibration(result.data);
      }
    });
  }
  break;
```

### 2. Update store

**File:** `src/store/useStore.ts`

Add state:
```typescript
interface AppState {
  // ... existing ...
  calibrationRunning: boolean;
  runningCalibrationId: string | null;
}

// Actions
setCalibrationRunning: (running: boolean, id: string | null) => {
  set({ calibrationRunning: running, runningCalibrationId: id });
},
```

### 3. Update CalibrationPanel

**File:** `src/components/calibration/CalibrationPanel.tsx`

```typescript
const { calibrationRunning, runningCalibrationId } = useStore();

const handleStartCalibration = async () => {
  try {
    const result = await api.startCalibration();
    
    if (result.success && result.data?.calibration_id) {
      addNotification('Calibration started - running in background', 'info');
      // UI state updated via WebSocket
    } else {
      addNotification(result.error || 'Failed to start calibration', 'error');
    }
  } catch (error) {
    addNotification('Network error starting calibration', 'error');
  }
};

// UI shows progress indicator when calibrationRunning === true
{calibrationRunning && (
  <div className="bg-blue-50 border border-blue-200 rounded p-4">
    <div className="flex items-center gap-3">
      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600" />
      <div>
        <p className="text-sm font-medium text-blue-900">
          Calibration in Progress
        </p>
        <p className="text-xs text-blue-700">
          ID: {runningCalibrationId}
        </p>
      </div>
    </div>
  </div>
)}
```

### 4. Update API client

**File:** `src/services/api.ts`

```typescript
async startCalibration(): Promise<ApiResponse<{ calibration_id: string }>> {
  // Now returns immediately with just calibration_id
  // WebSocket will notify on completion
  const response = await this.request<any>('/calibration/start', { 
    method: 'POST' 
  });
  
  if (response.success && (response.calibration_id || response.data?.calibration_id)) {
    const calibrationId = response.calibration_id || response.data?.calibration_id;
    return { success: true, data: { calibration_id: calibrationId } };
  }
  
  return response;
}

async getRunningCalibration(): Promise<ApiResponse<{ running: boolean; calibration_id?: string }>> {
  return this.request('/calibration/running');
}
```

---

## Benefits

1. **Non-blocking UI** - User can navigate away during calibration
2. **Progress indication** - UI shows calibration is running
3. **Real-time updates** - All connected clients see status
4. **Error handling** - Failures broadcast to all clients
5. **Consistent pattern** - Matches measurement flow

## Migration Path

1. ✅ Deploy backend changes first (backward compatible - just returns faster)
2. ✅ Update UI to handle WebSocket events
3. ✅ Test with multiple concurrent clients

## Testing

```bash
# Terminal 1: Start calibration
curl -X POST http://localhost:8080/api/calibration/start \
  -H "x-session-id: $SID"

# Terminal 2: WebSocket client
wscat -c ws://localhost:8080/ws

# Should see:
# {"type": "calibration_start", "data": {...}, "timestamp": "..."}
# ... (wait ~5 seconds) ...
# {"type": "calibration_complete", "data": {...}, "timestamp": "..."}
```
