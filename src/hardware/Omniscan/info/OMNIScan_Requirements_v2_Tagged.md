# 🩺 OMNIScan System Requirements (v2)
*(Reorganized and tagged for Doorstop: USR_, SYS_, RISK_, VER_)*

---

## 🧱 Subsystem: OMNI-SERVER (Rust Hardware Server)

### USER NEEDS

| ID | Title | Description |
|----|--------|-------------|
| **USR_OMNI-SERVER-001** | Reliable and safe operation | The hardware server must ensure safe beam and motion control under all conditions; interlocks must disable beam on any fault. |
| **USR_OMNI-SERVER-002** | Daily calibration capability | The operator must be able to verify device health via a daily calibration routine. |
| **USR_OMNI-SERVER-003** | Data integrity and traceability | All measurements and logs must remain traceable to device, operator, and calibration data. |

---

### SYSTEM REQUIREMENTS

| ID | Title | Description |
|----|--------|-------------|
| **SYS_OMNI-SERVER-001** | Safety authority | The Rust Hardware Server exclusively authorizes all beam and motion operations. |
| **SYS_OMNI-SERVER-002** | Interlocks and watchdogs | The system shall include key switch, enable button, E-stop, door sensor, beam watchdog, and optional over-temperature input. |
| **SYS_OMNI-SERVER-003** | Operational states | The state machine shall implement {IDLE, PENDING_ARMED, RUNNING, STOPPING, SAFE, CALIBRATION, MAINTENANCE, LOCKED}. |
| **SYS_OMNI-SERVER-004** | Calibration enforcement | Daily calibration must be performed once per 24 h; operation without valid calibration is prohibited. |
| **SYS_OMNI-SERVER-005** | Beam fault handling | The system shall abort exposure if beam intensity drops below threshold for > X ms and remain locked until stable. |
| **SYS_OMNI-SERVER-006** | Data retention | All metadata, logs, and calibration records must be stored indefinitely; archival encryption optional after ≥ 5 years. |
| **SYS_OMNI-SERVER-007** | Audit database | Maintain append-only, encrypted audit log with TPM-signed daily roll; export read-only for audits. |
| **SYS_OMNI-SERVER-008** | gRPC interface | Expose endpoints: StartCalibration, StartMeasurement, Abort, Stop, GetState, etc. |
| **SYS_OMNI-SERVER-009** | Session security | Require mutual TLS + password + key switch; session expires on disconnect. |
| **SYS_OMNI-SERVER-010** | Safety profile updates | Allow updates only through signed USB/cloud package with physical activation. |

---

### RISK CONTROLS (ISO 14971 Alignment)

| ID | Hazard / Failure Mode | Mitigation |
|----|-----------------------|-------------|
| **RISK_OMNI-SERVER-001** | Beam exposure when door open | Door sensor interlock disables beam output. |
| **RISK_OMNI-SERVER-002** | Motion when E-stop pressed | Hardware E-stop line breaks power to actuators. |
| **RISK_OMNI-SERVER-003** | Software watchdog failure | Independent hardware watchdog resets beam enable line. |
| **RISK_OMNI-SERVER-004** | Unauthorized configuration | Require signed update package and key switch activation. |
| **RISK_OMNI-SERVER-005** | Data loss during upload | Local database keeps all raw data until verified upload confirmation. |

---

### VERIFICATION

| ID | Title | Description |
|----|--------|-------------|
| **VER_OMNI-SERVER-001** | Unit test coverage | ≥ 80 % code coverage for all safety-related modules. |
| **VER_OMNI-SERVER-002** | Integration test | Validate gRPC state machine and interlock simulation. |
| **VER_OMNI-SERVER-003** | System qualification | Manual qualification of transitions IDLE ↔ RUNNING ↔ SAFE ↔ STOPPING. |
| **VER_OMNI-SERVER-004** | Calibration procedure test | Verify enforcement of 24-hour calibration rule. |
| **VER_OMNI-SERVER-005** | Fault injection | Simulate beam drop > X ms and confirm transition to SAFE state. |

---

## 🧩 Subsystem: OMNI-ORCH (Python Orchestrator)

*(Reserved for future use — structure prepared for Doorstop integration)*

| Level | Tag Prefix | Example |
|--------|-------------|----------|
| User Needs | `USR_OMNI-ORCH` | `USR_OMNI-ORCH-001` |
| System Reqs | `SYS_OMNI-ORCH` | `SYS_OMNI-ORCH-001` |
| Software Reqs | `SW_OMNI-ORCH` | `SW_OMNI-ORCH-001` |
| Risk | `RISK_OMNI-ORCH` | `RISK_OMNI-ORCH-001` |
| Verification | `VER_OMNI-ORCH` | `VER_OMNI-ORCH-001` |

---

## 💻 Subsystem: OMNI-UI (JavaScript Frontend)

*(Reserved for future use — structure prepared for Doorstop integration)*

| Level | Tag Prefix | Example |
|--------|-------------|----------|
| User Needs | `USR_OMNI-UI` | `USR_OMNI-UI-001` |
| System Reqs | `SYS_OMNI-UI` | `SYS_OMNI-UI-001` |
| Software Reqs | `SW_OMNI-UI` | `SW_OMNI-UI-001` |
| Risk | `RISK_OMNI-UI` | `RISK_OMNI-UI-001` |
| Verification | `VER_OMNI-UI` | `VER_OMNI-UI-001` |

---

## 🧾 Notes

- This structure follows **Doorstop-compatible YAML prefixing** for traceability.
- Each subsystem will be mapped to a separate tree under the common Doorstop root.
- Cross-links (e.g., `SYS_OMNI-ORCH` → `SYS_OMNI-SERVER`) will be created during integration.
