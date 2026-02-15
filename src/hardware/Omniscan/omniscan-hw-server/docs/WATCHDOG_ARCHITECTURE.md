# Watchdog Architecture

Comprehensive documentation of all watchdog and monitoring systems in the omniscan-hw-server.

---

## Overview

The system implements multiple independent watchdogs for safety-critical monitoring. This document clarifies the architecture, responsibilities, and implementation status of each watchdog.

---

## Watchdog Summary Table

| Watchdog Name | Monitors | Poll/Check Frequency | Failure Condition | Safety Action | Status |
|---------------|----------|---------------------|-------------------|---------------|--------|
| **GPIO Input Polling Watchdog** | All GPIO input pins | 100ms (demo) / 10ms (prod) | N/A (continuous monitoring) | Updates interlock state | ✅ Implemented |
| **Activation Button Timer** | Activation button timeout | Countdown (1s updates) | 20-second timeout expires | Deactivates button | ✅ Implemented |
| **Beam-Stop Watchdog** | Beam-stop actuator position | 100ms | Actuator output vs sensor mismatch > 200ms | Update radiation_safe flag | ✅ Implemented |
| **Beam Intensity Watchdog** | X-ray beam intensity (future) | TBD (likely 100ms) | Intensity outside ±10% for > X ms | Abort exposure, transition to SAFE | ⏳ Planned |
| **Hardware Watchdog** | Software execution | External hardware WDT | Software stops updating WDT | Hardware cuts beam enable line | ⏳ Planned |

---

## 1. GPIO Input Polling Watchdog

### Purpose
Continuously monitor all physical safety interlock sensors and update software interlock state.

### Implementation
- **Location**: `src/devices/gpio/demo.rs` (lines ~513-517)
- **Thread**: Background tokio task spawned on GPIO initialization
- **Poll Rate**: 100ms (demo mode), 10ms recommended for production

### Monitored Signals
- Emergency Stop (Pin 1)
- Door Closed (Pin 2)
- Beam-Stop Position (Pin 3)
- Cooling OK (Pin 4)
- Power OK (Pin 5)
- Key Switch (Pin 7)
- Activation Button (Pin 8)

### Processing
1. Read all input pins
2. Update internal interlock state fields
3. Detect key switch state changes
4. Handle activation button edge detection (rising edge triggers 20s timer)
5. Emit state change notifications

### Failure Mode
This is not a traditional "watchdog" that detects failures - it's a continuous monitor. If the GPIO hardware or thread fails, interlocks cannot be updated, which will prevent all operations.

---

## 2. Activation Button Timer

### Purpose
Enforce 20-second timeout window for potentially harmful operations requiring physical confirmation.

### Implementation
- **Location**: `src/devices/gpio/demo.rs` (lines ~522-530)
- **Mechanism**: Tokio async timer started on button press (rising edge on Pin 8)
- **Duration**: 20 seconds (configurable via `workflow.enable_button_timeout`)

### Monitored Condition
- Time elapsed since activation button was pressed

### Processing
1. User clicks activation button (GPIO Pin 8 rising edge)
2. Timer starts: `activation_button_expires_at = now() + 20s`
3. `is_activation_button_active()` checks: `now() < expires_at`
4. Auto-expires after 20 seconds
5. Emits notification on state change

### Safety Action
- After expiration, all harmful operations are rejected
- Operations in progress are NOT aborted (button only gates operation initiation)

---

## 3. Beam-Stop Watchdog

### Purpose
Monitor beam-stop actuator and verify mechanical position matches commanded state.

### Implementation
- **Location**: `src/devices/gpio/demo.rs` (lines ~519-524)
- **Thread**: Background tokio task
- **Check Rate**: 100ms

### Monitored Condition
- Beam-Stop Control output (Pin 6) vs Beam-Stop Position sensor (Pin 3)
- Simulates 150-200ms mechanical delay

### Processing
1. Read `beam_stop_output` (commanded state)
2. Wait for mechanical delay (150-200ms)
3. Update `radiation_safe_input` (actual position sensor)
4. If mismatch persists > 200ms: mechanical failure detected

### Safety Action
- Updates `radiation_safe_input` flag
- Safety state machine reads this flag for all exposure operations
- Beam OPEN requires `radiation_safe_input = false`
- Beam CLOSED requires `radiation_safe_input = true`

### Enforcement
- Activation button required to OPEN beam-stop (remove beam block)
- Closing beam-stop is always allowed (safety operation)

---

## 4. Beam Intensity Watchdog (FUTURE)

### Purpose
Monitor X-ray beam intensity during exposures and detect beam faults.

### Implementation Status
⏳ **Planned** - Framework prepared, not yet implemented

### Planned Behavior
- **Monitor**: X-ray detector or separate beam monitor sensor
- **Check Rate**: 100ms (10 Hz)
- **Threshold**: Beam intensity within ±10% of target
- **Tolerance**: Fault persists for > X ms (configurable, e.g., 200ms)

### Processing (Planned)
1. Read beam intensity from sensor
2. Compare to expected intensity for current exposure
3. If outside tolerance band:
   - Start fault timer
   - If fault persists > X ms: Trigger abort
4. If within tolerance: Reset fault timer

### Safety Action (Planned)
- Abort current exposure
- Transition safety state to SAFE
- Log beam fault event with details
- Require system reset before next measurement

### Requirement
- **SYS_OMNI-SERVER-005**: "System shall abort exposure if beam intensity drops below threshold for > X ms"

---

## 5. Hardware Watchdog (FUTURE)

### Purpose
Independent hardware-level monitoring to detect software hangs or crashes.

### Implementation Status
⏳ **Planned** - Requires external hardware watchdog timer (WDT)

### Planned Architecture
- **Hardware**: External WDT IC connected to beam enable line
- **Software**: Periodic "kick" signal from Rust server (e.g., every 500ms)
- **Failure Detection**: If software stops sending kick signal for > 1s, WDT triggers

### Processing (Planned)
1. Software thread sends periodic WDT kick signal
2. Hardware WDT resets timer on each kick
3. If timer expires (no kick received): Hardware action triggered

### Safety Action (Planned)
- Hardware WDT directly cuts beam enable line
- Independent of software state
- Requires manual reset (power cycle or reset button)

### Requirement
- **RISK_OMNI-SERVER-003**: "Independent hardware watchdog resets beam enable line on software failure"

---

## Watchdog Interaction Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Software Watchdogs                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │  GPIO Input      │      │  Activation      │        │
│  │  Polling WD      │      │  Button Timer    │        │
│  │  (100ms)         │      │  (20s countdown) │        │
│  └────────┬─────────┘      └────────┬─────────┘        │
│           │                         │                   │
│           ├─> Updates ────> InterlockStatus             │
│           │                         │                   │
│  ┌────────▼─────────┐      ┌────────▼─────────┐        │
│  │  Beam-Stop       │      │  Beam Intensity  │        │
│  │  Position WD     │      │  WD (Future)     │        │
│  │  (150-200ms)     │      │  (100ms)         │        │
│  └────────┬─────────┘      └────────┬─────────┘        │
│           │                         │                   │
│           └─────> Safety State <────┘                   │
│                   Machine                               │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ Hardware Interface
                        │
┌───────────────────────▼─────────────────────────────────┐
│               Hardware Watchdog (Future)                 │
│                                                          │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │  WDT Kick        │ ───> │  External WDT    │        │
│  │  (500ms)         │      │  Hardware        │        │
│  └──────────────────┘      └────────┬─────────┘        │
│                                     │                   │
│                            Timeout (1s)                 │
│                                     │                   │
│                            ┌────────▼─────────┐        │
│                            │  Cut Beam Enable │        │
│                            │  Line (Hardware) │        │
│                            └──────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Status Summary

### ✅ Implemented (Demo Mode)
1. GPIO Input Polling Watchdog
2. Activation Button Timer
3. Beam-Stop Watchdog

### ⏳ Planned (Production)
4. Beam Intensity Watchdog
5. Hardware Watchdog (external WDT)

---

## Production Deployment Checklist

Before deploying to production, the following watchdog enhancements are required:

- [ ] Reduce GPIO polling rate from 100ms to 10ms for faster interlock response
- [ ] Implement Beam Intensity Watchdog with sensor integration
- [ ] Integrate external hardware WDT for independent safety monitoring
- [ ] Validate all watchdog failure modes with fault injection testing
- [ ] Document watchdog test procedures in validation protocol

---

## Related Documentation

- **ARCHITECTURE.md**: Overall system architecture
- **HARDWARE.md**: GPIO pin assignments and device documentation
- **CALIBRATION.md**: Safety requirements for calibration
- **WORKFLOW.md**: Operational workflows and safety checks

---

*Document Version: 1.0*  
*Last Updated: 2026-01-20*
