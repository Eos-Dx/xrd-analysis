# Omniscan Hardware Server Architecture

## Overview

The Omniscan Hardware Server is a medical device control system that manages X-ray detector and motion hardware through a safety-controlled interface. It provides gRPC services for the orchestrator to control and monitor hardware state.

## System Startup Sequence

### 1. Computer Power-On
```
Computer Boots → GPIO Hardware Auto-Starts → Server Application Starts
```

- **GPIO Hardware**: Physical GPIO card powers on with computer
- **Interlocks**: Immediately readable (E-Stop, Door, Radiation, Cooling, Power)
- **Server**: Loads configuration, starts gRPC services

### 2. Initial State
```
Key Switch: OFF
GPIO: POWERED & READING INTERLOCKS
Detector: NOT INITIALIZED (No Power)
Motion: NOT INITIALIZED (No Power)
Safety State: LOCKED
```

### 3. Key Switch ON → Initialization Allowed
```
User: Turns physical key switch
→ GPIO detects key switch ON
→ Safety State: LOCKED → IDLE
→ Orchestrator can now request device initialization
```

### 4. Device Initialization Workflow

#### Prerequisites:
- ✅ Key Switch: ON
- ✅ All Interlocks: SAFE
- ✅ Activation Button: ACTIVE (20-second window)

#### Initialization Sequence:
```
Orchestrator → [gRPC: InitializeDetector] → Hardware Server
                                              ↓
                                         Check Prerequisites
                                              ↓
                                         Power ON Detector
                                              ↓
                                         Initialize Detector
                                              ↓
                                         Report Status
                                              ↓
                                         Orchestrator ← [Status Update]

Orchestrator → [gRPC: InitializeMotion] → Hardware Server
                                            ↓
                                       Check Prerequisites
                                            ↓
                                       Power ON Motion
                                            ↓
                                       Home Motion Axes
                                            ↓
                                       Report Status
                                            ↓
                                       Orchestrator ← [Status Update]
```

## Power Management

### Current Implementation (Demo)
All devices powered from same source as computer.

### Future Implementation (Production)
```
Computer Power → PDU (Power Distribution Unit)
                  ↓
                  ├→ Detector (Controlled Power)
                  ├→ Motion Controller (Controlled Power)
                  └→ Aux Equipment (Controlled Power)
```

## State Monitoring & Updates

### Push Model (State Change Notifications)

The server implements a **publish-subscribe** pattern for state updates:

```
Hardware Server                    Orchestrator
     │                                  │
     │◄─────[Subscribe: StateUpdates]──┤
     │                                  │
     │──────[Ack: Subscribed]──────────►│
     │                                  │
   [State                               │
   Changes]                             │
     │                                  │
     │──────[Notification: CHANGED]────►│
     │      (No details)                │
     │                                  │
     │◄─────[GetCurrentState]───────────┤
     │      (with certificate)          │
     │                                  │
     │──────[Full State Details]───────►│
     │                                  │
```

### State Change Events

Server **notifies** orchestrator when:
- GPIO state changes (interlocks, key switch, activation button)
- Detector state changes (power, temperature, status)
- Motion state changes (power, position, homing)
- Safety state transitions

### Security Model

⚠️ **Implementation Status**: mTLS framework complete, enforcement not active in demo mode.

1. **Notification**: Anyone can receive "something changed" event
2. **State Query**: Certificate validation framework implemented (production deployment required)

## API Endpoints

### Device Control (gRPC)

#### Detector
```protobuf
service DeviceControl {
  rpc InitializeDetector(InitRequest) returns (StatusResponse);
  rpc PowerOffDetector(Empty) returns (StatusResponse);
  rpc GetDetectorState(Empty) returns (DetectorState);
}
```

#### Motion
```protobuf
service DeviceControl {
  rpc InitializeMotion(InitRequest) returns (StatusResponse);
  rpc HomeMotion(HomeRequest) returns (StatusResponse);
  rpc PowerOffMotion(Empty) returns (StatusResponse);
  rpc GetMotionState(Empty) returns (MotionState);
}
```

#### State Monitoring
```protobuf
service StateMonitor {
  rpc SubscribeToStateUpdates(Empty) returns (stream StateChangeEvent);
  rpc GetFullServerState(Empty) returns (ServerState);
  rpc GetGpioState(Empty) returns (GpioState);
}
```

## Activation Button Logic

### Purpose
Physical confirmation required for **potentially harmful** operations that arrive from the orchestrator.

### Harmful Operations (Require Enable Button)
- **Initialize Detector**: Powers ON X-ray detector
- **Initialize Motion**: Activates motion system (can cause physical movement)
- **Start Exposure**: Activates X-rays (radiation hazard)
- **Move Motion**: Causes physical movement (collision hazard)

### Safe Operations (No Enable Button Required)
- **Read States**: Query GPIO/Detector/Motion state (read-only)
- **Stop Operations**: Stop exposure, stop motion (safety operation)
- **Power Off**: Controlled shutdown (safety operation)
- **Get Interlocks**: Read safety status (read-only)

### Behavior
- **Click**: Activates for 20 seconds
- **Countdown**: Live timer displayed in GUI
- **Auto-Expire**: Automatically deactivates after 20 seconds
- **Scope**: Each harmful operation must occur within the 20-second window

### Implementation
```rust
// Activate (in GPIO)
gpio.activate_enable_button().await?;

// Check status
let is_active = gpio.is_activation_button_active().await;

// Get remaining time
let remaining_secs = gpio.get_activation_button_remaining_time().await;
```

## Safety Interlocks

### Physical Interlocks (GPIO)
- **Emergency Stop**: Must be released (not pressed)
- **Door**: Must be closed
- **Radiation Monitor**: Must indicate safe
- **Cooling System**: Must be operational
- **Power Supply**: Must be stable

### Software Interlocks
- **Key Switch**: Must be ON
- **Activation Button**: Must be active (for init operations)
- **Safety State**: Must be appropriate for operation

### Interlock Check Flow
```
Operation Requested
    ↓
Check Physical Interlocks (GPIO)
    ↓
Check Software Interlocks
    ↓
Check Safety State Machine
    ↓
Proceed or Reject
```

## Configuration

### server.toml
```toml
[device]
name = "GPIO"              # "GPIO" = Demo mode
gui_mode = true            # Enable GUI
interlocks_armed = true    # Auto-arm interlocks

[workflow]
enable_button_timeout = 20  # Activation button duration (seconds)

[certificates]
enable_mtls = false        # Mutual TLS for orchestrator
device_uuid = "DEMO-001"   # Device identifier
```

## FDA/IEC 62304 Compliance

### Audit Logging

**Storage Model**:
- **Audit Database** (`audit.db`): Append-only command logs, state transitions (immutable)
- **Operational Database** (`operational.db`): Measurement records, calibration records (mutable status)

All operations are logged:
- User actions
- State transitions
- Safety checks
- Command execution
- Failures and errors

### Traceability
Each command includes:
- `command_id`: Unique identifier
- `user`: Who initiated
- `timestamp`: When occurred
- `reason`: Why executed
- Result and execution time

### Safety State Machine

**Complete State Set**: LOCKED, INITIALIZED, IDLE, WARMING_UP, CALIBRATED, PENDING_ARMED, RUNNING, STOPPING, SAFE, CALIBRATION, MAINTENANCE

**Common Operational Transitions** (simplified view):
- LOCKED → IDLE (key switch ON)
- IDLE → PENDING_ARMED (enable button + interlocks)
- PENDING_ARMED → RUNNING (operation started)
- RUNNING → STOPPING → SAFE (controlled shutdown)

See WARP.md for complete state definitions.

## Development vs Production

### Demo Mode
- `device.name = "GPIO"`
- All devices simulated
- GUI with full control
- Auto-armed interlocks

### Production Mode
- `device.name = "OMNIScan XRD System"`
- Real hardware interfaces
- GUI monitoring only
- Physical interlocks required

## Error Handling

### Hardware Failures
- Detector communication error → Abort, return to SAFE
- Motion limit reached → Stop, maintain position
- Interlock violation → Emergency stop, log event

### Timeout Handling
- Activation button expires → Reject initialization requests
- Command timeout → Return error, maintain safety state
- Watchdog timeout → Trigger safety shutdown

## Future Enhancements

1. **PDU Integration**: Controlled power distribution with sequencing
   - Power-on sequence: Cooling → Detector → Motion → Aux
   - Power-off sequence: Reverse order with safe shutdown
   - Failure handling: Power off all previous steps on any failure
2. **mTLS Authentication**: Enable certificate enforcement in production
3. **Remote Monitoring**: Telemetry and diagnostics
4. **Predictive Maintenance**: Health monitoring and alerts
5. **Multi-Device Support**: Multiple detectors/motion systems
