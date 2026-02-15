# Omniscan

A medical X-ray diffraction (XRD) diagnostic platform comprising a Rust-based Hardware Server, a Python Clinic Orchestrator, and a cloud backend for processing, ML inference, and reporting. The system targets IEC 62304 Class B and FDA De Novo compliance with strong safety, security, and traceability.

## Overview
- **Hardware Server** (Windows service, Rust): master authority over beam, motion, and safety; communicates with detectors and I/O; exposes gRPC.
- **Clinic Orchestrator** (Python): local workflow, UI bridge, buffering, and cloud sync with FDA-compliant audit trails, RBAC, and safety-first error handling.
- **Cloud Platform** (AWS): Core backend, processing, ML, and reporting services with message queues and specialized data stores.

## High-level Architecture
- Rust Hardware Server ⇄ Orchestrator via gRPC (mTLS)
- Orchestrator ⇄ Web UI via REST/WebSocket
- Cloud: Ingress → Processing → ML → Reporting queues; object storage for XRD data, metadata DBs for lineage.

## Safety Principles (Hardware Server)
- Interlocks: key switch, enable button, door sensor, E‑stop, beam watchdog, optional over‑temp.
- Software safety: pre-check interlocks; abort on interlock drop; exposure/dose timers; SAFE state enforcement; arm/enable window.
- Safety profile: encrypted/signed config sealed to TPM; engineer-only updates with physical action.
- Auditability: append-only encrypted audit DB with daily TPM-signed rolls; structured logs + Windows Event Log.

## Operational States
{ IDLE, PENDING_ARMED, RUNNING, STOPPING, SAFE, CALIBRATION, MAINTENANCE, LOCKED }

## gRPC API (seed)
- Acquisition: StartMeasurement, Stop, Abort, SubscribeRunEvents
- Health: Liveness, Readiness
- Common fields: command_id, user_context, timestamps, state transitions

## Data & QC
- Raw: one .pony file per run + sidecar JSON (stub schema)
- Metadata JSON: run_id (UUIDv4 from HW server), sample_id, operator, device info, QC, calibrant link, audit refs
- QC checks: SNR threshold; beam intensity watchdog; daily distance check

## Traceability and Compliance
- Standards: IEC 62304 (Class B), ISO 14971, IEC 62366-1, FDA Cybersecurity 2023, ISO 13485 QMS, ISO 27001, HIPAA/GDPR.
- Evidence: requirements ↔ risks ↔ code ↔ tests ↔ logs mapped in a traceability matrix (e.g., Doorstop).
- DHF structure: Plans, Requirements, Design, Code, Verification, Release, Maintenance.

## Requirements Summary
- USR: adjunct diagnostic support; safe operation; daily calibration; full traceability.
- SYS: cloud-connected operations; local buffering; end-to-end lineage; security & compliance; hybrid V&V.
- OMNI: safety authority in HW server; interlocks & watchdogs; daily calibration; audit DB; gRPC interface; session security; safety profile updates.
- ORCH: local coordination, offline mode with upload blocking for diagnosis, session pairing via mTLS + password.
- CLOUD: AWS single-tenant VPC; durable queues; model governance and reporting stubs.
- DATA/QC: formats and QC thresholds as above.
- VER: unit/integration/system tests; clinical validation; Doorstop traceability.

## Development Roadmap
1) Phase 1: Port C# BIS logic to Rust (tokio/tonic); baseline gRPC; functional parity.
2) Phase 2: Harden security (mTLS, rustls, TPM-backed keys), health endpoints.
3) Phase 3: Safety Profile manager, GPIO/interlocks, watchdogs, audit DB.
4) Phase 4: Add AdvoCam/Thorlabs/Power Supply drivers; telemetry; CI simulation/mocks.

## Open-Source Tooling
- CI/CD: GitLab CE or Jenkins
- Requirements: Doorstop (YAML in repo)
- Testing: pytest, Allure, Kiwi TCMS; Rust: cargo test
- Static/SCA: clippy, rustfmt, cargo-audit, SonarQube
- SBOM: Syft, CycloneDX
- Docs: Markdown + Pandoc

## Local Development
- Rust nightly not required; prefer stable. Enable `tokio`, `tonic`, `rustls`, `tracing`.
- Windows MSVC toolchain; run as service in release builds; dev runs as console app.
- Python 3.11+ for orchestrator. Node (optional) for UI.

## Repository Structure (proposed)
- **hardware-server/** (Rust)
  - src/ drivers/ safety/ grpc/ audit/
- **orchestrator/** (Python) - [omniscan-orchestrator/](omniscan-orchestrator/)
  - Interactive CLI with FDA compliance features
  - Audit trails, RBAC, safety-first error handling
  - See [orchestrator README](omniscan-orchestrator/README.md)
- **certificate-center/** - [omniscan-certificate-center/](omniscan-certificate-center/)
  - mTLS certificate management and generation
- **hw-server/** - [omniscan-hw-server/](omniscan-hw-server/) (Python prototype)
  - FastAPI hardware server prototype
- **ui/** (web client) - React-based user interface (planned)
- **cloud/** (IaC, services stubs)
- **docs/** (DHF, requirements, risk, design)
- **tests/** (integration, simulation)

## Getting Started

### Hardware Server (Rust)
```bash
cd hardware-server && cargo run
```

### Orchestrator (Python)
```bash
cd omniscan-orchestrator
pip install -e .

# Interactive session with FDA compliance features
omni-orch interactive --cert <cert> --key <key> --ca-cert <ca>

# Generate engineer certificate
omni-orch cert generate --engineer-id ENG001 --device-uuid ABC123
```

See [Orchestrator README](omniscan-orchestrator/README.md) for complete documentation.

### Development Tools
- Lint: `cargo clippy -- -D warnings`
- Audit: `cargo audit`
- View requirements: Open `doorstop/html/index.html` in browser

## Licensing
TBD.
