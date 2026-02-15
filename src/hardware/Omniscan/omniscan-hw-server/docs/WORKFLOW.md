# Operational Workflow Guide

## System Startup

### Step 1: Power On System
```
1. Turn on main power
2. Computer boots automatically
3. GPIO hardware initializes (auto-powered)
4. Hardware Server starts automatically (if configured)
```

**Initial State:**
- ✅ GPIO: Powered & Reading Interlocks
- ⚫ Key Switch: OFF
- ⚫ Detector: Not Initialized
- ⚫ Motion: Not Initialized
- 🔒 Safety State: LOCKED

---

## Device Initialization

### Step 2: Turn Key Switch ON
```
Operator: Physically turns key switch to ON position
```

**GPIO Detects:**
- Key Switch: OFF → ON

**Server Response:**
- Safety State: LOCKED → IDLE
- Activation Button: Now meaningful (previously ignored)
- Device Init: Now allowed (with activation button)

**New State:**
- ✅ GPIO: Powered & Reading Interlocks
- ✅ Key Switch: ON
- ⚫ Detector: Not Initialized
- ⚫ Motion: Not Initialized
- 🟢 Safety State: IDLE

---

### Step 3: Activate Activation Button
```
Operator/Orchestrator: Click "ACTIVATE" button in GPIO Control Panel
```

**Terminology Note**: This button is called "Activation Button" throughout the system (also referred to as "Enable Button" in some contexts).

**Result:**
- Activation Button: ACTIVE for 20 seconds
- Countdown displayed: "✅ ACTIVE (20s)" → "✅ ACTIVE (19s)" → ...
- Non-safe operations: ALLOWED

**State:**
- ✅ GPIO: Powered & Reading Interlocks
- ✅ Key Switch: ON
- ✅ Activation Button: ACTIVE (20s countdown)
- ⚫ Detector: Not Initialized
- ⚫ Motion: Not Initialized
- 🟢 Safety State: IDLE

---

### Step 4a: Initialize Detector (via Orchestrator)

**Orchestrator sends gRPC request:**
```protobuf
InitializeDetector {
  ctx: CommandContext {
    command_id: "uuid-123"
    user: "engineer@company.com"
    reason: "Start X-ray imaging session"
    timestamp: ...
  }
}
```

**Server Checks:**
1. ✅ Key Switch: ON?
2. ✅ Activation Button: ACTIVE?
3. ✅ All Interlocks: SAFE?
4. ✅ Safety State: IDLE?

**If ALL checks pass:**
```
→ Power ON detector (same source as computer)
→ Initialize detector hardware
→ Set detector state: INITIALIZING → IDLE
→ Return success to orchestrator
```

**If ANY check fails:**
```
→ Return error with reason
→ Log failure in audit system
→ Maintain current state
```

**New State:**
- ✅ GPIO: Powered & Reading Interlocks
- ✅ Key Switch: ON
- ✅ Activation Button: ACTIVE (remaining time)
- ✅ Detector: INITIALIZED & IDLE
- ⚫ Motion: Not Initialized
- 🟢 Safety State: IDLE

---

### Step 4b: Initialize Motion (via Orchestrator)

**Orchestrator sends gRPC request:**
```protobuf
InitializeMotion {
  ctx: CommandContext {
    command_id: "uuid-456"
    user: "engineer@company.com"
    reason: "Prepare motion system"
    timestamp: ...
  }
}
```

**Server Checks:**
1. ✅ Key Switch: ON?
2. ✅ Activation Button: ACTIVE?
3. ✅ All Interlocks: SAFE?
4. ✅ Safety State: IDLE?

**If ALL checks pass:**
```
→ Power ON motion controller
→ Home motion axes (find reference positions)
→ Set motion state: HOMING → IDLE
→ Return success to orchestrator
```

**Final State:**
- ✅ GPIO: Powered & Reading Interlocks
- ✅ Key Switch: ON
- ⚫ Activation Button: EXPIRED (after 20s)
- ✅ Detector: INITIALIZED & IDLE
- ✅ Motion: INITIALIZED & HOMED
- 🟢 Safety State: IDLE
- 🎯 **System Ready for Operations**

---

## Normal Operations

### Step 5: Execute Measurement

**Prerequisites:**
- ✅ Detector: Initialized
- ✅ Motion: Initialized & Homed
- ✅ Key Switch: ON
- ✅ All Interlocks: SAFE
- ✅ Activation Button: ACTIVE (harmful operation - activates X-rays)

**Orchestrator sends:**
```protobuf
StartExposure {
  ctx: CommandContext { ... }
  exposure_time_ms: 1000
}
```

**Server Executes:**
```
Safety Check → IDLE → PENDING_ARMED
    ↓
Start Detector Exposure
    ↓
PENDING_ARMED → RUNNING
    ↓
[X-ray exposure in progress]
    ↓
Exposure Complete
    ↓
RUNNING → STOPPING → IDLE
```

---

## State Monitoring

### Orchestrator Subscription

**Step 1: Subscribe to Updates**
```protobuf
SubscribeToStateUpdates {}
```

**Server Response:**
```
→ Returns stream of StateChangeEvent
→ Events contain: "CHANGED" + component type
→ NO detailed state information
```

**Step 2: Query Specific State (with certificate)**
```protobuf
GetGpioState {}          // Returns full GPIO state
GetDetectorState {}      // Returns full detector state
GetMotionState {}        // Returns full motion state
GetFullServerState {}    // Returns everything
```

**Security:**
- State change notifications: Open (anyone can subscribe)
- State details: Authenticated (certificate required)

---

## Error Scenarios

### Activation Button Expired During Init

**Scenario:**
```
T=0s:  Click ACTIVATE button
T=10s: Orchestrator sends InitializeDetector
T=25s: Orchestrator sends InitializeMotion  ← TOO LATE
```

**Result:**
```
InitializeMotion → ERROR: "Activation button not active"
→ Orchestrator must click ACTIVATE again
→ Retry InitializeMotion within 20 seconds
```

---

### Interlock Violation During Operation

**Scenario:**
```
Measurement in progress
→ Door opened (interlock violation)
```

**Server Response:**
```
1. Detect interlock fault (GPIO)
2. Trigger emergency stop
3. Abort current operation
4. Safety State: RUNNING → SAFE
5. Log event to audit system
6. Notify orchestrator: "INTERLOCK_VIOLATION"
```

**Recovery:**
```
1. Close door (restore interlock)
2. Verify all interlocks safe
3. Reset system if needed
4. Re-initialize devices
5. Resume operations
```

---

### Key Switch Turned OFF

**Scenario:**
```
System operational
→ Key switch turned OFF
```

**Server Response:**
```
1. Detect key switch OFF (GPIO)
2. Initiate safe shutdown
3. Stop all operations
4. Power off detector (controlled)
5. Disable motion (safe stop)
6. Safety State: * → LOCKED
7. Notify orchestrator: "KEY_SWITCH_OFF"
```

**State After:**
- ✅ GPIO: Still powered (hardware)
- ⚫ Key Switch: OFF
- ⚫ Detector: Powered down
- ⚫ Motion: Disabled
- 🔒 Safety State: LOCKED

---

## Shutdown Sequence

### Normal Shutdown

**Step 1: Stop Operations**
```
If measurement in progress:
  → Complete current exposure
  → Save data
```

**Step 2: Power Down Devices**
```
Orchestrator → PowerOffDetector
Orchestrator → PowerOffMotion
```

**Step 3: Turn Key OFF**
```
Operator → Turns key switch OFF
Server → Transitions to LOCKED state
```

**Step 4: Exit Application**
```
Close Server GUI
→ gRPC server continues (background)
→ Or Ctrl+C to stop everything
```

---

## Quick Reference

### State Transitions

```
LOCKED ──[Key ON]──► IDLE
IDLE ──[StartMeasurement + Enable]──► PENDING_ARMED
PENDING_ARMED ──[Interlocks OK]──► RUNNING
RUNNING ──[Complete]──► STOPPING
STOPPING ──[Cleanup]──► IDLE
ANY ──[E-Stop / Error]──► SAFE
SAFE ──[Recovery]──► IDLE
```

### Required Conditions

| Operation | Key ON | Activation Button | Interlocks | Init Status | Reason |
|-----------|--------|-------------------|------------|-------------|--------|
| Init Detector | ✅ | ✅ (20s) | ✅ | N/A | **Harmful**: Powers ON X-ray detector |
| Init Motion | ✅ | ✅ (20s) | ✅ | N/A | **Harmful**: Activates motion system |
| Start Exposure | ✅ | ✅ (20s) | ✅ | ✅ Both* | **Harmful**: Activates X-rays |
| Move Motion | ✅ | ✅ (20s) | ✅ | ✅ Motion | **Harmful**: Physical movement |
| Stop Exposure | ✅ | ⚫ | ✅ | - | Safe: Stops operation |
| Get States | ✅ | ⚫ | - | - | Safe: Read-only |
| Power Off | ✅ | ⚫ | - | - | Safe: Shutdown |

*Calibration exposures bypass the "valid calibration required" check (bootstrap exception).

### Timeouts

- Activation Button: **20 seconds**
- Measurement Timeout: Configurable (default: 300s)
- gRPC Request Timeout: 30s
- Interlock Poll Rate: 100ms
