# Dashboard Hardware Initialization Implementation

## Overview
Added two "Init" buttons to the Dashboard that allow operators to initialize the Detector and Motion Control systems directly from the dashboard interface.

---

## Changes Made

### 1. Updated Dashboard Component
**File:** `omniscan-ui/src/pages/Dashboard.tsx`

**New Features:**
- Added "Init Detector" button
- Added "Init Motion" button
- Both buttons are located in a blue information panel with clear instructions
- Buttons are disabled during initialization to prevent multiple simultaneous requests
- Loading indicator shown during initialization

**Key Code:**
```typescript
const handleInitializeDevice = async (device: 'detector' | 'motion') => {
  setIsInitializing(true);
  try {
    const response = await api.initializeDevice(device);
    
    if (response.success) {
      addNotification(
        `${device.charAt(0).toUpperCase() + device.slice(1)} initialized successfully`,
        'success'
      );
    } else {
      addNotification(
        response.error || `Failed to initialize ${device}`,
        'error'
      );
    }
  } catch (error) {
    addNotification(
      `Error initializing ${device}: ${error instanceof Error ? error.message : 'Unknown error'}`,
      'error'
    );
  } finally {
    setIsInitializing(false);
  }
};
```

---

## User Flow

### Step-by-Step Process

1. **Operator opens Dashboard** (default landing page after login)

2. **Review Current State**
   - Safety interlocks display on the left panel
   - Hardware status shows on the right panel
   - Yellow info banner explains measurements are in Measurement tab

3. **Initialize Hardware**
   - Operator **physically clicks the ENABLE button** on the GPIO control panel
   - Timer starts (20-second window)
   - Operator then clicks "Init Detector" or "Init Motion" button on UI

4. **Backend Processing**
   ```
   UI: Click "Init Detector"
     ↓
   API Client: POST /api/hardware/detector/init
     ↓
   REST Server (orchestrator):
     - Check if enable button is active (within 20s window)
     - Verify all safety interlocks
     - Call gRPC to initialize detector
     ↓
   Hardware Server:
     - Power up detector
     - Run initialization sequence
     - Return status
     ↓
   UI: Show success/error notification
   ```

5. **Result**
   - Success: Green notification appears ("Detector initialized successfully")
   - Error: Red notification shows the issue (e.g., "Enable button not active")

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Dashboard                                                   │
│ System overview and equipment status                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │ Safety Interlocks    │  │ Hardware Status      │        │
│  │                      │  │                      │        │
│  │ - Overall Safe       │  │ - PDU Status         │        │
│  │ - Key Switch: ON     │  │ - GPIO Status        │        │
│  │ - Door: CLOSED       │  │ - Detector Status    │        │
│  │ - etc.               │  │ - Motion Status      │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│ 🔧 Hardware Initialization                                  │
│                                                              │
│ Click ENABLE button on GPIO panel, then click below to     │
│ initialize devices (20-second window).                     │
│                                        ┌──────────────────┐│
│                                        │ Init Detector    ││
│                                        ├──────────────────┤│
│                                        │ Init Motion      ││
│                                        └──────────────────┘│
├─────────────────────────────────────────────────────────────┤
│ ℹ️ Measurements can only be started from the Measurement   │
│    tab. Ensure all hardware is initialized and calibrated  │
│    first.                                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Safety Considerations

### Enable Button Requirement
- **Required:** Physical confirmation via GPIO panel's ENABLE/ACTIVATION button
- **Duration:** 20-second window after pressing
- **Why:** Prevents accidental initialization without operator awareness
- **Error if missed:** "Enable button not active" (HTTP 412)

### Safety Interlocks Checked
The orchestrator verifies before initialization:

| Interlock | Status | Purpose |
|-----------|--------|---------|
| Key switch | ON | System armed |
| Radiation safe | TRUE | Beam physically blocked |
| Cooling OK | TRUE | Thermal systems ready |
| Power OK | TRUE | Power distribution stable |
| Emergency stop | Not triggered | Safety circuit intact |

### If Interlocks Fail
- Red error notification displays the specific issue
- Examples:
  - "Key switch must be ON for detector initialization"
  - "Radiation is NOT SAFE (beam not blocked) - block beam before initializing detector"
  - "Cooling is NOT OK - check cooling system before initializing detector"

---

## API Integration

### Endpoint: POST /api/hardware/{device}/init

**Parameters:**
- `device`: "detector" or "motion"
- Header: `x-session-id` (automatically added by API client)

**Response (Success):**
```json
{
  "success": true,
  "device": "detector",
  "status": "initialized",
  "detail": {
    "status": "initialized",
    "powered": true,
    "initialized": true,
    "detector_status": "IDLE",
    "temperature": 22.5
  }
}
```

**Response (Enable Button Not Active):**
```json
{
  "error": "Enable button not active",
  "message": "Click ENABLE button in GPIO panel to initialize detector",
  "instructions": "You have 20 seconds after clicking the button to complete initialization"
}
```
HTTP Status: `412 Precondition Failed`

**Response (Other Errors):**
```json
{
  "error": "Radiation is NOT SAFE (beam not blocked)..."
}
```
HTTP Status: `500 Internal Server Error`

---

## Notification System

Notifications are managed through Zustand store:

```typescript
// Success notification
addNotification('Detector initialized successfully', 'success');

// Error notification  
addNotification('Enable button not active', 'error');

// Auto-dismisses after 5 seconds
```

**Notification Types:**
- `success` - Green background, button worked
- `error` - Red background, something went wrong
- `warning` - Yellow background, caution needed
- `info` - Blue background, informational

---

## Implementation Details

### Component Structure
```
Dashboard (page component)
├── useState(isInitializing) - Prevent multiple requests
├── useStore(systemState) - Current system health
├── useStore(addNotification) - Show user feedback
├── handleInitializeDevice() - Calls API
├── SafetyInterlockPanel - Shows interlock status
├── HardwareStatusPanel - Shows device status
├── Init Buttons Panel
│   ├── Instructions
│   ├── "Init Detector" button
│   └── "Init Motion" button
└── Info Banner - Reminders about workflow
```

### Button States
- **Default:** Blue, clickable
- **Disabled (during init):** Gray, shows loading spinner
- **Hover:** Darker blue
- **Active:** Connected to API call

---

## Orchestrator Integration

### gRPC Client Methods

**For Detector:**
```python
def initialize_detector(self, user: str = "unknown") -> Dict[str, Any]:
    # 1. Check enable button active (20s window)
    # 2. Verify all safety interlocks
    # 3. Call: self.device_init.InitializeDetector(request)
    # 4. Return status
```

**For Motion:**
```python
def initialize_motion(self, user: str = "unknown") -> Dict[str, Any]:
    # Same process as detector
    # Calls: self.device_init.InitializeMotion(request)
```

### REST Server Endpoint

**File:** `omniscan-orchestrator/src/omniscan_orchestrator/rest_server.py`

```python
@app.post("/api/hardware/{device}/init")
async def initialize_device(device: str, user: Dict = Depends(get_current_user)):
    # Validate device type
    # Check enable button
    # Initialize device
    # Broadcast to WebSocket clients
    # Return result
```

---

## Workflow Example

### Scenario: Operator initializes Detector

1. **Operator is on Dashboard page**
   - Sees safety status (all green)
   - Sees hardware status (detector OFF)

2. **Operator initiates init**
   - Walks to GPIO panel
   - Presses physical ENABLE button (20s timer starts)
   - Quickly returns to computer

3. **Operator clicks "Init Detector"**
   - Button becomes gray with loading spinner
   - Backend checks: enable button is active ✓
   - Backend verifies interlocks ✓
   - Backend sends gRPC to hardware server
   - Hardware server powers up detector
   - Takes 2-3 seconds

4. **Result displayed**
   - Button returns to normal blue
   - Green notification: "Detector initialized successfully"
   - Hardware status updates to show detector IDLE

5. **Next step**
   - Detector ready for calibration or measurement
   - Operator can now repeat for Motion system
   - Both systems must be initialized before measurements

---

## Error Handling

### Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Enable button not active | Didn't press GPIO button | Press ENABLE button on GPIO panel, then retry within 20s |
| Radiation is NOT SAFE | Beam block not engaged | Verify beam block is physically installed and closed |
| Cooling is NOT OK | Cooling system issue | Check cooling pump, verify no blockages |
| Power is NOT OK | PDU issue | Check power distribution unit power switch |
| Key switch must be ON | Physical key switch OFF | Turn key switch ON on GPIO panel |

---

## Testing

### Unit Tests Can Verify:
- [ ] Button renders with correct text
- [ ] Button disabled during initialization
- [ ] Loading spinner shows during API call
- [ ] Success notification on 200 response
- [ ] Error notification on failed response
- [ ] Error notification on network error
- [ ] Both detector and motion buttons work independently

### Integration Tests Can Verify:
- [ ] API call includes session header
- [ ] API endpoint returns correct response format
- [ ] Orchestrator validates enable button
- [ ] Orchestrator checks all interlocks
- [ ] Orchestrator calls correct gRPC method

---

## Future Enhancements

1. **Progress indication** - Show real-time progress during initialization
2. **Auto-init** - Option to initialize both detector and motion together
3. **Retry logic** - Auto-retry if enable button expires mid-init
4. **Calibration check** - Verify calibration is valid before allowing measurements
5. **Status subscriptions** - Update device status via WebSocket in real-time
6. **Historical logs** - Show recent init operations and results

---

## Summary

The Dashboard now provides operators with a clear, safe interface to initialize hardware devices with:
- ✅ Simple one-click interface for each device
- ✅ Clear instructions about ENABLE button requirement
- ✅ Real-time feedback via notifications
- ✅ Safety interlocks verification at orchestrator layer
- ✅ Disabled state during operations to prevent conflicts
- ✅ Integration with existing authentication and notification systems
