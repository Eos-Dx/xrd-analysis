# Equipment Command Schemas

This directory contains declarative TOML schemas describing the logic, safety requirements, and execution flow for all equipment commands in the OMNIScan medical device system.

## Purpose

Command schemas provide a **single source of truth** for:
- ✅ Safety requirements (states, interlocks, enable button, calibration)
- ✅ Parameter validation (types, ranges, constraints)
- ✅ Execution logic (step-by-step device interactions)
- ✅ Audit logging requirements
- ✅ FDA risk classification
- ✅ Error handling and rollback procedures

## Schema Structure

Each command schema is a TOML file with the following structure:

```toml
command_id = "command_name"
description = "Human-readable description"
service = "ServiceName"  # Which gRPC service
audit_level = "Critical"  # Low | Standard | High | Critical
risk_level = "High"       # Low | Medium | High | Critical

[safety_requirements]
allowed_states = ["Idle", "Calibrated"]
requires_enable_button = true
requires_calibration = true
requires_key_switch = true
enable_timeout_seconds = 20
max_execution_time_seconds = 300

[[safety_requirements.interlock_checks]]
name = "door_closed"
required_value = true
error_message = "Safety door must be closed"

[[parameters]]
name = "exposure_time_ms"
required = true
description = "Exposure time in milliseconds"

[parameters.param_type]
type = "Float"
unit = "milliseconds"
min = 100.0
max = 300000.0

[execution]
rollback_on_failure = true

[[execution.steps]]
type = "StateTransition"
to_state = "PendingArmed"
reason = "Awaiting physical enable button"

[[execution.steps]]
type = "CallDevice"
device = "detector"
method = "start_exposure"
parameters = ["exposure_time_ms"]

[[execution.error_cleanup]]
type = "StateTransition"
to_state = "Safe"
reason = "Operation failed"
```

## Command Inventory

### Acquisition Service (X-ray Measurements)

| Command | Description | Requires Enable | Requires Calibration | Beam Control | Risk Level |
|---------|-------------|-----------------|---------------------|--------------|------------|
| `start_exposure` | Begin X-ray exposure measurement | ✅ Yes | ✅ Yes | Opens/Closes | **High** |
| `stop` | Controlled stop of measurement | ❌ No | ❌ No | Closes | Medium |
| `abort` | Emergency abort (hardware-triggered) | ❌ No | ❌ No | Closes | **Critical** |
| `calibrate_detector` | Perform daily calibration with QC | ✅ Yes | ❌ No | Opens/Closes | **High** |
| `start_background_measurement` | Acquire dark frame (beam closed) | ❌ No | ❌ No | Verifies Closed | Low |

### Motion Service (XY Stage Control)

| Command | Description | Requires Enable | Risk Level |
|---------|-------------|-----------------|------------|
| `move_to` | Move to absolute position (mm) | ❌ No | Medium |
| `move_relative` | Move relative distance (mm) | ❌ No | Medium |
| `home` | Home XY stage to reference position | ❌ No | Medium |
| `motion_stop` | Stop all motion immediately | ❌ No | Low |
| `set_velocity` | Configure motion speed (mm/s) | ❌ No | Low |

### Device Control Service

| Command | Description | Requires Enable | Risk Level |
|---------|-------------|-----------------|------------|
| `power_device` | Power on/off detector, motion, GPIO | ❌ No | Medium |
| `initialize_detector` | Initialize detector hardware | ✅ Yes | Medium |
| `initialize_motion` | Initialize motion controller | ✅ Yes | Medium |
| `power_off_detector` | Safely power down detector | ❌ No | Low |
| `power_off_motion` | Safely power down motion | ❌ No | Low |

### Safety Service

| Command | Description | Risk Level |
|---------|-------------|------------|
| `reset_interlocks` | Clear interlock violations after safety restored | **High** |

## Execution Step Types

### Device Interaction
- **`CallDevice`**: Call a method on detector, motion, GPIO, or PDU
- **`CheckInterlock`**: Verify an interlock signal value
- **`SetGpioOutput`**: Set a GPIO output pin state

### State Management
- **`StateTransition`**: Transition safety state machine
- **`WaitForCondition`**: Poll for a condition to become true
- **`Delay`**: Wait for a fixed duration

### Logging & Notifications
- **`LogAudit`**: Write to FDA audit trail
- **`NotifyStateChange`**: Broadcast state change event

### Control Flow
- **`ConditionalBranch`**: Execute different steps based on condition

## Safety Requirements

### Allowed States
Commands specify which `SafetyState` values permit execution:
- `Locked` - System locked, requires calibration
- `Initialized` - Key switch ON, awaiting login
- `Idle` - Ready for operations
- `WarmingUp` - X-ray source warming up
- `Calibrated` - Valid calibration within 24h
- `PendingArmed` - Awaiting physical enable button
- `Running` - Active measurement
- `Stopping` - Controlled shutdown
- `Safe` - Emergency safe state
- `Calibration` - Calibration in progress
- `Maintenance` - Maintenance mode

### Interlock Checks
Commands specify required interlock states:
- `key_switch` - Key switch in ON position
- `enable_button` - Physical enable button pressed
- `emergency_stop` - E-stop NOT pressed (false = safe)
- `door_closed` - Safety door closed (true)
- `beam_watchdog` - Beam intensity safe (true)
- `radiation_safe` - Beam shutter closed (true)

### Enable Button Requirement
High-risk commands require physical confirmation:
- `start_exposure` - ✅ Requires enable button (20s timeout)
- `calibrate_detector` - ✅ Requires enable button (20s timeout)
- `initialize_detector` - ✅ Requires enable button (20s timeout)
- `initialize_motion` - ✅ Requires enable button (20s timeout)

### Beam-Stop (Shutter) Control
Commands that control X-ray beam access:
- `start_exposure` - Opens beam-stop after enable button → closes after measurement completes
- `calibrate_detector` - Opens beam-stop after enable button → closes after calibration completes
- `stop` - Closes beam-stop when stopping measurement
- `abort` - Immediately closes beam-stop for safety (hardware-triggered)
- `start_background_measurement` - Verifies beam-stop is closed (`radiation_safe` = true)

**Beam-Stop States:**
- **Open** (`radiation_safe = false`): X-rays active, Radiation LED RED
- **Closed** (`radiation_safe = true`): X-rays blocked, Radiation LED GREEN

**Safety Requirement:**
- ⚠️ **CRITICAL**: Beam-stop can only be OPENED when activation button is active
- Closing beam-stop (safe operation) is always allowed without activation button
- This prevents accidental radiation exposure in demo/test environments
- GUI enforces this by disabling manual "Remove Beam-Stop" control unless activation is active

### Blocking Operations
Commands that wait for completion before returning:
- **Measurements**: `start_exposure`, `calibrate_detector`, `start_background_measurement`
- **Motion**: `move_to`, `move_relative`, `home` (waits for motion idle)
- **Initialization**: `initialize_detector`, `initialize_motion` (waits for device ready)
- **Non-blocking**: `motion_stop`, `set_velocity`, `power_device`, power-off commands

## Parameter Types

Schemas define parameters with built-in validation:

### Float
```toml
[parameters.param_type]
type = "Float"
unit = "milliseconds"
min = 100.0
max = 300000.0
```

### Integer
```toml
[parameters.param_type]
type = "Integer"
min = 1000
max = 600000
```

### Boolean
```toml
[parameters.param_type]
type = "Boolean"
```

### String
```toml
[parameters.param_type]
type = "String"
max_length = 256
pattern = "^[A-Za-z0-9_-]+$"  # Optional regex
```

### Enum
```toml
[parameters.param_type]
type = "Enum"
allowed_values = ["detector", "motion", "gpio"]
```

## Risk Classification

Commands are classified by FDA risk level:

### High Risk (Radiation/Patient Exposure)
- `start_exposure` - X-ray beam exposure
- `calibrate_detector` - Calibration with beam open

### Critical Risk (Emergency/Safety)
- `abort` - Emergency termination
- `reset_interlocks` - Safety system reset

### Medium Risk (Device State Changes)
- `move_to`, `move_relative`, `home` - Motion control
- `power_device` - Device power changes
- `initialize_detector`, `initialize_motion` - Device initialization

### Low Risk (Queries/Configuration)
- `motion_stop`, `set_velocity` - Motion configuration
- `start_background_measurement` - Dark frame (beam closed)
- `power_off_detector`, `power_off_motion` - Safe power down

## Audit Levels

Commands specify audit logging granularity:

### Critical (FDA-Critical Operations)
- `start_exposure` - Full execution trace
- `calibrate_detector` - QC results and PONI files
- `abort` - Complete abort sequence
- `reset_interlocks` - Safety system changes

### High (Safety-Related)
- `stop` - Measurement termination
- `initialize_detector`, `initialize_motion` - Device initialization
- `power_device` - Power state changes

### Standard (Normal Operations)
- `move_to`, `move_relative`, `set_velocity` - Motion commands
- `start_background_measurement` - Dark frames

### Low (Queries)
- Read-only operations (not typically in schemas)

## Usage in Code

### Loading Schemas
```rust
use omniscan_hw_server::commands::CommandRegistry;

// Load all command schemas from directory
let registry = CommandRegistry::load_from_directory("config/commands")?;

// Get a specific command
let start_exposure_schema = registry.get("start_exposure").unwrap();

// List all commands
let all_commands = registry.list_commands();

// Get commands by service
let acquisition_commands = registry.get_by_service("Acquisition");
```

### Validating Commands
```rust
// Validate schema structure
schema.validate()?;

// Check if command allowed in current state
if !schema.safety_requirements.allowed_states.contains(&current_state) {
    return Err("Command not allowed in current state");
}

// Validate parameters
for param in &schema.parameters {
    if param.required && !provided_params.contains_key(&param.name) {
        return Err("Missing required parameter");
    }
}
```

### Generating Documentation
```rust
// Generate markdown documentation
let docs = registry.generate_documentation();
std::fs::write("docs/COMMAND_REFERENCE.md", docs)?;
```

## Benefits

### For Development
- ✅ **Single source of truth** - All command logic documented in one place
- ✅ **Type safety** - Schema validation catches errors at load time
- ✅ **Reusability** - Same schema used by gRPC, REST, CLI, GUI
- ✅ **Testability** - Validate schemas independently of implementation

### For FDA Compliance
- ✅ **Traceability** - Clear documentation of all command requirements
- ✅ **Risk management** - Explicit risk classification for each command
- ✅ **Audit trail** - Configurable logging levels per command
- ✅ **Safety validation** - Interlock requirements documented

### For Operations
- ✅ **Self-documenting** - Human-readable TOML format
- ✅ **Version control** - Track changes to command logic over time
- ✅ **Easy updates** - Modify schemas without code changes
- ✅ **Validation** - Catch configuration errors at startup

## Extending the System

### Adding a New Command

1. Create a new TOML file in `config/commands/`:
```toml
command_id = "my_new_command"
description = "Description of what it does"
service = "MyService"
audit_level = "Standard"
risk_level = "Medium"

[safety_requirements]
allowed_states = ["Idle"]
requires_enable_button = false
requires_calibration = false
requires_key_switch = true

[execution]
rollback_on_failure = false

[[execution.steps]]
type = "CallDevice"
device = "detector"
method = "my_method"
parameters = []
```

2. Restart server - schemas are loaded at startup
3. Command is now available via gRPC/API

### Adding a New Execution Step Type

1. Update `src/commands/schema.rs`:
```rust
pub enum ExecutionStep {
    // ... existing types ...
    
    MyNewStepType {
        my_parameter: String,
    },
}
```

2. Implement execution logic in command executor
3. Use in schemas:
```toml
[[execution.steps]]
type = "MyNewStepType"
my_parameter = "value"
```

## Files

| File | Command | Service |
|------|---------|---------|
| `start_exposure.toml` | StartExposure | Acquisition |
| `stop.toml` | Stop | Acquisition |
| `abort.toml` | Abort | Acquisition |
| `calibrate_detector.toml` | CalibrateDetector | Acquisition |
| `start_background_measurement.toml` | StartBackgroundMeasurement | Acquisition |
| `move_to.toml` | MoveTo | Motion |
| `move_relative.toml` | MoveRelative | Motion |
| `home.toml` | Home | Motion |
| `motion_stop.toml` | Stop | Motion |
| `set_velocity.toml` | SetVelocity | Motion |
| `power_device.toml` | PowerDevice | DeviceControl |
| `initialize_detector.toml` | InitializeDetector | DeviceInitialization |
| `initialize_motion.toml` | InitializeMotion | DeviceInitialization |
| `power_off_detector.toml` | PowerOffDetector | DeviceInitialization |
| `power_off_motion.toml` | PowerOffMotion | DeviceInitialization |
| `reset_interlocks.toml` | ResetInterlocks | Safety |

---

## Summary Statistics

**Total Commands**: 16 schemas covering all OMNIScan equipment operations

**Safety-Critical**: 4 commands (start_exposure, calibrate_detector, abort, reset_interlocks)

**Requires Enable Button**: 4 commands (radiation + initialization operations)

**Requires Calibration**: 1 command (start_exposure - patient measurements only)

**Beam-Stop Control**: 5 commands (4 open/close, 1 verification)

**Blocking Operations**: 8 commands (wait for completion before returning)

**Hardware-Triggered**: 1 command (abort - triggered by E-stop or key switch)

**All 16 command schemas validated and ready for implementation!** ✅
