# Hardware Devices Documentation

Complete documentation for all hardware devices controlled by the OMNIScan Hardware Server.

---

## Table of Contents

1. [Overview](#overview)
2. [X-ray Detectors](#x-ray-detectors)
3. [Motion Control System](#motion-control-system)
4. [GPIO & Safety Interlocks](#gpio--safety-interlocks)
5. [Power Distribution Unit (PDU)](#power-distribution-unit-pdu)

---

## Overview

The hardware server provides exclusive control over four main device types:

- **X-ray Detectors** 🔬 - Capture X-ray diffraction patterns
- **Motion Controllers** 🎯 - Precise XY-stage positioning
- **GPIO PCIe Device** 🔒 - Safety interlock system and I/O
- **PDU** ⚡ - Power distribution and control

All devices implement trait-based abstractions for modularity and testability.

---

## X-ray Detectors

### Purpose
Capture X-ray diffraction patterns from patient samples for diagnostic analysis.

### Supported Implementations
- **DemoDetector** - Simulation for development and testing
- **TestDetector** - Unit testing mock
- **Future**: Bruker BIS (TCP), AdvaCam (C DLL)

### Architecture

```
src/devices/detectors/
├── mod.rs           # DetectorDevice trait definition
├── demo.rs          # Demo implementation
└── test.rs          # Test implementation
```

### DetectorDevice Trait

```rust
#[async_trait]
pub trait DetectorDevice: Send + Sync {
    // Power control
    async fn power_on(&self) -> Result<(), DetectorError>;
    async fn power_off(&self) -> Result<(), DetectorError>;
    async fn is_powered(&self) -> bool;
    
    // Status monitoring
    async fn get_status(&self) -> DetectorStatus;
    async fn get_health(&self) -> DetectorHealth;
    
    // Exposure operations
    async fn start_exposure(&self, exposure_time_ms: u32) -> Result<(), DetectorError>;
    async fn stop_exposure(&self) -> Result<(), DetectorError>;
    async fn get_last_result(&self) -> Option<ExposureResult>;
    
    // Calibration
    async fn calibrate(&self) -> Result<(), DetectorError>;
}
```

### Detector Status States

```rust
pub enum DetectorStatus {
    Off,           // Powered off
    Init,          // Initializing (not yet ready)
    Idle,          // Ready for exposure
    Exposing,      // Actively collecting X-ray data
    Reading,       // Reading data from sensor
    Error(String), // Error state
}
```

### State Transitions

```
Off → Init (power_on)
Init → Idle (initialization complete)
Idle → Exposing (start_exposure)
Exposing → Reading (exposure time complete)
Reading → Idle (readout complete)
Any → Error (hardware failure)
* → Off (power_off)
```

### Key Operations

#### Power Management
```rust
// Power on detector
detector.power_on().await?;

// Power off detector
detector.power_off().await?;

// Check power status
let powered = detector.is_powered().await;
```

#### Exposure Control
```rust
// Start 5-second exposure
detector.start_exposure(5000).await?;

// Stop ongoing exposure
detector.stop_exposure().await?;

// Get last exposure result
if let Some(result) = detector.get_last_result().await {
    println!("Exposure: {}ms", result.exposure_time_ms);
    println!("Data size: {} bytes", result.data_size);
}
```

#### Health Monitoring
```rust
let health = detector.get_health().await;
println!("Temperature: {:.1}°C", health.temperature);
println!("Total exposures: {}", health.total_exposures);
```

### Demo Implementation Features

**Power Management**:
- Voltage: 0V (off) → 12V (on)
- Temperature: 22.5°C (ambient) → 25°C (powered)
- Initialization: Instant (production: 2-5 seconds)

**Exposure Simulation**:
- Real-time delays matching exposure_time_ms
- Reading phase: 100ms fixed
- Temperature rise: +0.5°C per exposure
- Data size: exposure_time_ms * 1024 bytes

**Calibration**:
- Duration: 5 seconds (production: 300 seconds)
- 10% random failure rate (for testing)

### Safety Requirements

**Power-On Prerequisites**:
- ✅ Key switch must be ON
- ✅ Activation button must be ACTIVE (20-second window)
- ✅ All safety interlocks satisfied

**Exposure Prerequisites**:
- ✅ Detector powered and IDLE
- ✅ Valid calibration (within 24 hours) - **Exception**: Calibration exposures bypass this check
- ✅ Safety state allows measurement
- ✅ Activation button required for measurements

### gRPC Integration

```protobuf
service Acquisition {
  rpc StartExposure(StartExposureRequest) returns (Empty);
  rpc Stop(StopRequest) returns (Empty);
  rpc GetLastExposureResult(Empty) returns (GetExposureResultResponse);
  rpc CalibrateDetector(CalibrateDetectorRequest) returns (Empty);
}

service DeviceControl {
  rpc PowerDevice(PowerDeviceRequest) returns (Empty);
  rpc GetDetectorHealth(Empty) returns (DetectorHealth);
}
```

---

## Motion Control System

### Purpose
Manage precision XY stage positioning for sample alignment and measurement workflows.

### Supported Implementations
- **XYDemoMotion** - Two-axis XY stage simulation
- **DemoMotion** - Single-axis demo
- **TestMotion** - Unit testing mock
- **Future**: Thorlabs Stage, Newport Stage

### Architecture

```
src/devices/motions/
├── mod.rs           # MotionControlDevice trait
├── xy_demo.rs       # XY stage demo implementation
├── demo.rs          # Single-axis demo
└── test.rs          # Test implementation
```

### MotionControlDevice Trait

```rust
#[async_trait]
pub trait MotionControlDevice: Send + Sync {
    // Power control
    async fn power_on(&self) -> Result<(), MotionError>;
    async fn power_off(&self) -> Result<(), MotionError>;
    async fn is_powered(&self) -> bool;
    
    // Status monitoring
    async fn get_status(&self) -> MotionStatus;
    async fn get_health(&self) -> MotionHealth;
    
    // Positioning operations
    async fn home(&self) -> Result<(), MotionError>;
    async fn move_to(&self, position_mm: f64) -> Result<(), MotionError>;
    async fn move_relative(&self, distance_mm: f64) -> Result<(), MotionError>;
    async fn stop_motion(&self) -> Result<(), MotionError>;
    
    // Configuration
    async fn get_position(&self) -> Option<f64>;
    async fn get_limits(&self) -> MotionLimits;
    async fn set_velocity(&self, velocity_mm_s: f64) -> Result<(), MotionError>;
}
```

### Motion Status States

```rust
pub enum MotionStatus {
    Off,           // Powered off
    Init,          // Initializing (not yet ready)
    Idle,          // Ready for movement
    Moving,        // Actively moving to target
    Homing,        // Finding reference position
    Error(String), // Error state
    LimitHit,      // Hardware limit switch triggered
}
```

### State Transitions

```
Off → Init (power_on)
Init → Homing (auto-home if configured)
Homing → Idle (homing complete)
Idle → Moving (move_to/move_relative)
Moving → Idle (movement complete)
Moving → LimitHit (limit reached)
Any → Error (hardware failure)
* → Off (power_off)
```

### Key Operations

#### Homing (REQUIRED before positioning)
```rust
// Home all axes to establish reference positions
motion.home().await?;

// For XY stage
xy_motion.home_xy().await?;
```

**Purpose**: Homing is **required** before:
- Calibration measurements
- Absolute positioning (move_to)
- Any diagnostic measurements

**Duration**:
- Demo: 2 seconds (both axes)
- Production: 5-20 seconds

#### Absolute Positioning
```rust
// Move to 50mm position
motion.move_to(50.0).await?;

// For XY stage
xy_motion.move_to_xy(50.0, 50.0).await?;
```

**Requirements**:
- Must be homed first
- Position within limits
- No motion in progress
- Safety interlocks satisfied

#### Relative Movement
```rust
// Move 10mm forward
motion.move_relative(10.0).await?;

// Move 5mm backward
motion.move_relative(-5.0).await?;
```

#### Velocity Control
```rust
// Set slower velocity for fine positioning
motion.set_velocity(5.0).await?;
```

### XY Demo Implementation

**Configuration**:
```toml
[motion]
max_speed = 50.0          # mm/s
acceleration = 100.0      # mm/s²

[motion.x_axis]
min_position = 0.0
max_position = 100.0
max_velocity = 50.0

[motion.y_axis]
min_position = 0.0
max_position = 100.0
max_velocity = 50.0
```

**Movement Simulation**:
- Diagonal movement (both axes simultaneously)
- Trapezoidal velocity profile
- Realistic acceleration/deceleration
- Progress updates via notifications

### Safety Requirements

**Power-On Prerequisites**:
- ✅ Key switch must be ON
- ✅ Activation button must be ACTIVE (20-second window)
- ✅ All safety interlocks satisfied
- ✅ No obstacles in motion path

**Movement Prerequisites**:
- ✅ Controller powered
- ✅ **Must be homed** (establishes coordinate system)
- ✅ Target position within limits
- ✅ No ongoing motion
- ✅ Safety interlocks satisfied

**Emergency Stop**:
- All motion stops immediately
- Position may be inaccurate
- **Re-homing required** before further movements

### gRPC Integration

```protobuf
service Motion {
  rpc MoveTo(MoveToRequest) returns (Empty);
  rpc MoveRelative(MoveRelativeRequest) returns (Empty);
  rpc Home(HomeRequest) returns (Empty);
  rpc Stop(StopRequest) returns (Empty);
  rpc SetVelocity(SetVelocityRequest) returns (Empty);
  rpc GetPosition(Empty) returns (GetPositionResponse);
}

service DeviceControl {
  rpc PowerDevice(PowerDeviceRequest) returns (Empty);
  rpc GetMotionHealth(Empty) returns (MotionHealth);
}
```

---

## GPIO & Safety Interlocks

### Purpose
Hardware-level safety interlock system providing physical safety controls and I/O management.

### Interface
PCIe expansion card with digital I/O (production) or demo simulation (development).

### Pin Assignments

**Terminology Note**: The radiation safety system consists of:
- **Beam-Stop Position Sensor** (GPIO Pin 3): Mechanical beam block position feedback
- **Beam Intensity Watchdog** (Future): X-ray flux monitoring (not yet implemented)

#### Input Pins (Interlocks & Controls)
| Pin | Name | Logic | Purpose |
|-----|------|-------|---------|
| 1 | Emergency Stop | HIGH = safe | E-stop button NC contacts |
| 2 | Door Closed | HIGH = safe | Door reed switch NO |
| 3 | Beam-Stop Position | HIGH = safe | Beam-stop position sensor (closed=safe) |
| 4 | Cooling OK | HIGH = safe | Temperature switch NC |
| 5 | Power OK | HIGH = safe | PDU power good signal |
| 6 | Beam-Stop Control | OUTPUT | Beam-stop actuator command |
| 7 | Key Switch | HIGH = ON | Key switch position |
| 8 | Activation Button | HIGH = pressed | Momentary push button |
| 9 | Reserved | - | Future expansion |

#### Output Pins (Indicators & Control)
| Pin | Name | Function |
|-----|------|----------|
| 10 | Radiation Warning LED | Multi-color: Red/Yellow/Green/Off |
| 11 | Door Status LED | Green = closed, Red = open |
| 12 | Key Switch LED | Green = ON, Red = OFF |
| 13 | Power Status LED | Green = OK, Red = fault |
| 14 | Cooling Status LED | Green = OK, Red = fault |

### GPIO State Structure

```rust
pub struct DemoGpioState {
    pub powered: bool,
    pub pins: HashMap<u8, GpioPin>,
    
    // Interlock states (updated by watchdog)
    pub emergency_stop_ok: bool,
    pub door_closed_ok: bool,
    pub radiation_safe_input: bool,
    pub cooling_ok: bool,
    pub power_ok: bool,
    
    // Control outputs
    pub beam_stop_output: bool,
    
    // LED states
    pub radiation_led: LedColor,
    pub door_led: bool,
    pub key_led: bool,
    pub power_led: bool,
    pub cooling_led: bool,
    
    // Activation button
    pub activation_button_active: bool,
    pub activation_button_expires_at: Option<Instant>,
}
```

### Key Switch Monitoring

The key switch controls system access:

**States**:
- **OFF (LOCKED)**: No operations allowed, system in LOCKED state
- **ON (OPERATE)**: Operations enabled, system transitions to IDLE

**Monitoring**:
- Polled every 100ms by GPIO watchdog
- State changes trigger safety state machine transitions
- All changes logged to audit system

**Key Switch Behavior**:
```rust
// Read current state
let key_on = gpio.get_key_switch_state().await?;

// Demo mode: Set state (simulates physical switch)
gpio.set_key_switch_sync(true);  // Turn ON
```

### Activation Button (Enable Button)

**Purpose**: Physical confirmation for potentially harmful operations.

**Behavior**:
- Click activates for **20 seconds**
- Countdown displayed in GUI
- Auto-expires after timeout
- Required for: detector/motion init, exposures, motion commands

**API**:
```rust
// Activate for 20 seconds
gpio.activate_enable_button().await?;

// Check if active
let is_active = gpio.is_activation_button_active().await;

// Get remaining time
let remaining_secs = gpio.get_activation_button_remaining_time().await;
```

### Interlock System

**Safety Logic**:
```rust
overall_safe = emergency_stop_ok 
            && door_closed_ok 
            && radiation_safe_input 
            && cooling_ok 
            && power_ok;
```

All five interlocks must be true for system to be safe.

**Interlock Monitoring**:
- Real-time monitoring (10ms polling in production, 100ms in demo)
- Immediate response to violations
- Automatic transition to SAFE state on fault
- Complete audit trail of all events

### LED Control

**Radiation Warning LED** (Multi-color):
- **Red**: Unsafe - beam open
- **Orange**: System not ready
- **Green**: Safe and ready

**Status LEDs** (On/Off):
- **Door LED**: Green = closed, Off/Red = open
- **Key LED**: Green = ON, Off/Red = OFF
- **Power LED**: Green = OK, Off/Red = fault
- **Cooling LED**: Green = OK, Off/Red = fault

**Auto-Update Logic**: LEDs automatically reflect interlock states.

### Watchdog Architecture

**1. GPIO Input Polling Watchdog**
- Polls input pins at 100ms intervals
- Updates interlock state fields
- Detects key switch changes
- Handles activation button edge detection

**2. Beam-Stop Watchdog**
- Monitors beam-stop actuator output
- Simulates 150-200ms mechanical delay
- Updates radiation_safe_input sensor
- Enforces activation button requirement for beam removal

**3. Activation Button Timer**
- Manages 20-second enable window
- Triggered by rising edge on Pin 8
- Counts down and auto-expires
- Emits notifications on state changes

### Safety Requirements

**Beam-Stop Control**:
- ⚠️ **CRITICAL**: Beam-stop can only be OPENED when activation button is active
- Closing beam-stop is always allowed (safety operation)
- Prevents accidental radiation exposure

### GUI Integration

**Demo Mode**:
- Full control of all GPIO states
- Simulate sensor inputs
- Control LED outputs
- Test activation button timing

**Production Mode**:
- Visualization only (read-only)
- Displays real hardware state
- No control buttons (hardware is source of truth)

### gRPC Integration

```protobuf
service Safety {
  rpc GetInterlockStatus(Empty) returns (InterlockStatus);
  rpc ResetInterlocks(CommandContext) returns (Empty);
  rpc CheckSafetyToOperate(Empty) returns (InterlockStatus);
}

service StateMonitor {
  rpc GetGpioState(Empty) returns (GpioStateResponse);
}
```

---

## Power Distribution Unit (PDU)

### Purpose
Master power control for the entire hardware system, ensuring safe power sequencing.

### Current Implementation

**Demo Mode**: Simple binary ON/OFF state simulation
**Production Future**: Controlled power distribution with per-outlet management

### PDU State Structure

```rust
pub struct DemoPdu {
    pub state: Arc<RwLock<DemoPduState>>,
    notifier: Arc<RwLock<Option<StateNotificationSender>>>,
}

pub struct DemoPduState {
    pub powered: bool,
    pub last_changed: Instant,
}
```

### Key Operations

#### Power Control
```rust
// Power ON
pdu.set_powered_sync(true);

// Power OFF
pdu.set_powered_sync(false);

// Check status
let powered = pdu.is_powered_sync();
```

**Effects**:
- Updates powered state
- Updates last_changed timestamp
- Emits POWER_CHANGED notification
- Logs power state change

**Thread Safety**: Uses `try_write()` with `blocking_write()` fallback - safe from any thread including GUI.

#### Notification Configuration
```rust
// Configure state change broadcaster
let (tx, _rx) = tokio::sync::broadcast::channel(100);
pdu.set_notifier_sync(tx);
```

### State Monitoring

**Notifications**:
```rust
StateChangeNotification {
    component: "PDU",
    change_type: "POWER_CHANGED",
    timestamp: <current_time>
}
```

**Subscribers** (orchestrator, monitoring services) receive notifications and can react accordingly.

### Integration with GPIO

The PDU controls overall system power, which affects GPIO interlocks:

```
PDU State         GPIO Interlock
─────────────────────────────────
OFF      →        power_ok = false
ON       →        power_ok = true
```

**Power-On Sequence**:
1. PDU must be ON
2. GPIO power_ok must be true
3. Key switch must be ON
4. All other interlocks satisfied

**Emergency Power-Off**:
1. PDU powered OFF
2. All devices lose power
3. System enters safe state
4. Re-initialization required

### GUI Integration

```rust
// Power status display
let powered = pdu.is_powered_sync();
ui.horizontal(|ui| {
    ui.label("Main Power:");
    if powered {
        ui.colored_label(egui::Color32::GREEN, "ON");
    } else {
        ui.colored_label(egui::Color32::RED, "OFF");
    }
});

// Power control button
let button_text = if powered { "Power OFF" } else { "Power ON" };
if ui.button(button_text).clicked() {
    pdu.set_powered_sync(!powered);
}
```

### Future Enhancements

**Multi-Outlet PDU**:
- Per-outlet power control (detector, motion, aux equipment)
- Current/voltage monitoring per outlet
- Power consumption tracking
- Over-current protection
- Outlet grouping and dependencies
- Power budget management

**Network PDU Integration**:
- SNMP or HTTP API communication
- Remote power control
- UPS integration
- Real-time power monitoring

---

## Common Patterns

### Error Handling

All devices follow consistent error handling:

```rust
pub enum DeviceError {
    NotPowered,
    AlreadyInProgress,
    InvalidParameter(String),
    HardwareError(String),
    SafetyViolation(String),
}
```

### Health Monitoring

All devices expose health information:

```rust
pub struct DeviceHealth {
    pub powered: bool,
    pub status: DeviceStatus,
    pub uptime: Duration,
    pub total_operations: u64,
    // Device-specific fields...
}
```

### State Notifications

All devices can emit state change notifications:

```rust
service_state.notify_state_change("DETECTOR", "POWER_CHANGED");
service_state.notify_state_change("MOTION", "POSITION_CHANGED");
service_state.notify_state_change("GPIO", "KEY_SWITCH_CHANGED");
service_state.notify_state_change("PDU", "POWER_CHANGED");
```

---

## Related Documentation

- **CALIBRATION.md** - Detector calibration procedures
- **WORKFLOW.md** - Operational workflows
- **ARCHITECTURE.md** - System architecture
- **COMMANDS_SPEC.md** - Complete gRPC command catalog
