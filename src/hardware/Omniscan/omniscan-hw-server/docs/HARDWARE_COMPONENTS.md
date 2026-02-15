# OMNIScan Hardware Components

## Overview

This document details the physical hardware devices and software services controlled by the OMNIScan Hardware Server.

---

## 🔌 Physical Hardware Devices

### 1. **X-ray Detectors** 🔬

**Purpose**: Capture X-ray diffraction patterns from patient samples for diagnostic analysis.

**Planned Implementations**:
- **Bruker BIS Detector**
  - Interface: TCP/IP network protocol
  - Communication: Async TCP socket with custom protocol
  - Status: Planned integration
  
- **AdvoCam Detector**
  - Interface: C DLL (Foreign Function Interface)
  - Communication: Native function calls via FFI
  - Status: Planned integration

**Software Interface**: `DetectorDevice` trait in `src/devices/detectors/`

**Key Operations**:
- Start/stop exposure
- Configure exposure parameters
- Retrieve diffraction data
- Status monitoring

---

### 2. **Motion Devices** 🎯

**Purpose**: Precise XY-stage positioning for sample placement and scanning.

**Planned Implementations**:
- **Thorlabs Motor Controllers**
  - Interface: C DLL (Foreign Function Interface)
  - Axes: X and Y (2D positioning)
  - Status: Planned integration

**Software Interface**: `MotionControlDevice` trait in `src/devices/motions/`

**Key Operations**:
- Absolute positioning (`MoveTo`)
- Relative movement (`MoveRelative`)
- Homing sequence
- Velocity control
- Position feedback
- Emergency stop

---

### 3. **PDU (Power Distribution Unit)** ⚡

**Purpose**: Centralized power control for all hardware components ensuring safe power sequencing.

**Interface**: TCP/IP with authentication

**Key Operations**:
- Power on individual components
- Power off individual components
- Query power status
- Emergency power cutoff

**Safety Features**:
- Authenticated access control
- Power sequencing (safe startup/shutdown order)
- Status monitoring and fault detection

---

### 4. **GPIO PCIe Device** 🔒

**Purpose**: Hardware-level safety interlock system providing physical safety controls.

**Interface**: PCIe expansion card with digital I/O

**Software Interface**: `GpioDevice` trait in `src/devices/gpio/`

**Signals Monitored**:

1. **Key Switch**
   - Function: Operator mode selection
   - States: OFF / OPERATE / MAINTENANCE
   - Safety: Required in OPERATE for measurements

2. **Emergency Stop (E-Stop)**
   - Function: Hardware-level immediate shutdown
   - Type: Hardware interlock (breaks power)
   - Action: Immediately disables beam and motion

3. **Door Interlock Sensors**
   - Function: Safety door state monitoring
   - Action: Beam disabled when door open
   - Type: Fail-safe (normally open)

4. **Physical Enable Button**
   - Function: Beam authorization confirmation
   - Requirement: Must be held during exposure
   - Safety: Prevents unintended exposures

5. **Beam Watchdog**
   - Function: Continuous X-ray intensity monitoring
   - Action: Triggers safety shutdown if beam fault detected
   - Monitoring: Real-time intensity validation

**Safety Philosophy**: Multiple independent hardware interlocks ensure beam cannot operate in unsafe conditions.

---

### 5. **Optional Future Devices** 🔮

#### Temperature Tracker
- **Purpose**: Internal machine temperature monitoring
- **Function**: Prevent overheating, thermal safety
- **Interface**: Analog sensor via GPIO or network
- **Status**: Future enhancement

#### Vacuum Pump Controls
- **Purpose**: Sample chamber vacuum system management
- **Function**: Vacuum pressure control and monitoring
- **Interface**: TCP/IP or serial protocol
- **Status**: Future enhancement

---

## 📡 Software Services (gRPC)

### 1. **Acquisition Service** 🎯

**Purpose**: Control X-ray exposure and data acquisition.

**Key RPCs**:
- `StartExposure`: Initiate X-ray exposure with parameters
- `Stop`: Controlled shutdown of active exposure
- `Abort`: Emergency abort of exposure
- `GetState`: Query current system state
- `CalibrateDetector`: Execute calibration routine
- `GetLastExposureResult`: Retrieve acquisition data
- `SubscribeRunEvents`: Real-time event streaming

**Safety Integration**: Validates all interlocks before starting exposure.

---

### 2. **Motion Service** 🎛️

**Purpose**: Control sample positioning and scanning.

**Key RPCs**:
- `MoveTo`: Absolute positioning command
- `MoveRelative`: Relative movement command
- `Home`: Execute homing sequence
- `Stop`: Stop motion immediately
- `SetVelocity`: Configure motion speed
- `GetPosition`: Query current position

**Safety Integration**: Motion disabled during exposures and when interlocks violated.

---

### 3. **Health Service** ❤️

**Purpose**: System health monitoring and readiness checks.

**Key RPCs**:
- `Liveness`: Basic server health check
- `Readiness`: Comprehensive operational readiness
- `GetAggregateHealth`: Detailed health status of all components

**Monitoring**: Detector health, motion status, power state, interlock status.

---

### 4. **Safety Service** 🛡️

**Purpose**: Safety interlock monitoring and validation.

**Key RPCs**:
- `GetInterlockStatus`: Query all interlock states
- `ResetInterlocks`: Clear latched faults (requires authorization)
- `CheckSafetyToOperate`: Validate system safety for operation

**Critical Service**: All operations must pass safety checks before execution.

---

### 5. **Beam Monitoring Service** 📊

**Purpose**: Real-time beam position and intensity tracking.

**Key RPCs**:
- `GetBeamPosition`: Query current beam position
- `GetBeamIntensity`: Query current beam intensity
- `SubscribeBeamTracking`: Real-time streaming of beam parameters

**Application**: Quality control, beam stability monitoring, safety validation.

---

### 6. **Calibrant Quality Service** 🎓

**Purpose**: Calibration validation and quality assessment.

**Key RPCs**:
- `ValidateCalibrant`: Assess calibrant quality
- `GetCalibrationStatus`: Query calibration validity and age

**Compliance**: Enforces 24-hour calibration rule per FDA requirements.

---

## 🔄 Hardware-Software Integration

### Device Abstraction Layer

All hardware devices implement trait-based abstractions:

```rust
// src/devices/detectors/mod.rs
pub trait DetectorDevice {
    async fn start_exposure(&mut self, params: ExposureParams) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    async fn get_status(&self) -> Result<DetectorStatus>;
}

// src/devices/motions/mod.rs
pub trait MotionControlDevice {
    async fn move_to(&mut self, position: Position) -> Result<()>;
    async fn home(&mut self) -> Result<()>;
    async fn get_position(&self) -> Result<Position>;
}

// src/devices/gpio/mod.rs
pub trait GpioDevice {
    fn read_interlocks(&self) -> Result<InterlockStatus>;
    fn read_enable_button(&self) -> Result<bool>;
}
```

### Communication Patterns

- **Synchronous**: GPIO interlock reads (real-time safety)
- **Asynchronous**: Detector operations, motion commands
- **Streaming**: Beam monitoring, event subscriptions
- **Request-Response**: Status queries, health checks

---

## 🛡️ Safety Architecture

### Multi-Layer Safety

1. **Hardware Layer**: Physical interlocks (GPIO)
2. **Software Layer**: State machine validation
3. **Protocol Layer**: gRPC command context validation
4. **Audit Layer**: Comprehensive logging

### Safety State Machine Integration

All hardware operations must pass through the safety state machine:
- LOCKED → IDLE → PENDING_ARMED → RUNNING → STOPPING → SAFE

### Interlock Monitoring

Continuous monitoring of all safety signals with immediate response to violations.

---

## 📋 Development Status

| Component | Status | Priority | Integration |
|-----------|--------|----------|-------------|
| Detectors (DEMO) | ✅ Complete | - | Implemented |
| Motion (DEMO) | ✅ Complete | - | Implemented |
| GPIO (DEMO) | ✅ Complete | - | Implemented |
| Bruker BIS | 🚧 Planned | High | Phase 1 |
| AdvoCam | 🚧 Planned | Medium | Phase 1 |
| Thorlabs Motion | 🚧 Planned | High | Phase 1 |
| PDU Control | 🚧 Planned | High | Phase 1 |
| GPIO PCIe | 🚧 Planned | Critical | Phase 1 |
| Beam Monitoring | 🚧 Planned | High | Phase 2 |
| Calibrant Quality | 🚧 Planned | Medium | Phase 2 |
| Temperature | 📋 Future | Low | Phase 4 |
| Vacuum Control | 📋 Future | Low | Phase 4 |

**Legend**: ✅ Complete | 🚧 Planned | 📋 Future

---

## 🔗 References

- **Main README**: [README.md](README.md)
- **Requirements**: [../info/OMNIScan_Requirements_v2_Tagged.md](../info/OMNIScan_Requirements_v2_Tagged.md)
- **mTLS Setup**: [MTLS_SETUP.md](MTLS_SETUP.md)
- **Global Architecture**: [../GLOBAL_README.md](../GLOBAL_README.md)

---

*Document Version: 1.0*  
*Last Updated: 2024-10-24*
