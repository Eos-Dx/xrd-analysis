# OMNISCAN System Architecture - Technical Summary

**For External Specialists**

---

## System Overview

OMNISCAN is a medical X-ray diffraction device with three-tier architecture ensuring safety, auditability, and regulatory compliance (FDA 21 CFR Part 11).

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface (React)               │
│              Browser-based Control & Monitoring         │
└────────────────────┬────────────────────────────────────┘
                     │ WebSocket + REST (HTTPS)
                     │
┌────────────────────▼────────────────────────────────────┐
│              Orchestrator (Python/FastAPI)              │
│        Business Logic, Workflows, Authentication        │
└────────────────────┬────────────────────────────────────┘
                     │ gRPC (mTLS)
                     │
┌────────────────────▼────────────────────────────────────┐
│           Hardware Server (Rust/Tokio)                  │
│       Device Control, Safety, Audit Logging             │
└────────────────────┬────────────────────────────────────┘
                     │ Hardware Protocols
                     │
┌────────────────────▼────────────────────────────────────┐
│    Physical Devices (Detector, Motion, GPIO, PDU)       │
│           X-ray Detector, XY Stage, Interlocks          │
└─────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Hardware Server (Rust)
- **Language**: Rust 2021 Edition
- **Runtime**: Tokio async runtime
- **Communication**: gRPC (Tonic framework)
- **Database**: SQLite (audit logs, measurements)
- **Encryption**: AES-256-GCM (data at rest)
- **Certificates**: X.509 for mTLS
- **GUI**: egui (optional, for demo/debug)

### Orchestrator (Python)
- **Framework**: FastAPI
- **Communication**: 
  - gRPC client (to Hardware Server)
  - WebSocket (to UI)
  - REST API (to UI)
- **Database**: SQLite/PostgreSQL
- **Authentication**: JWT tokens

### User Interface (React)
- **Framework**: React.js
- **Communication**: 
  - REST API calls
  - WebSocket for real-time updates
- **Protocol**: HTTPS only

---

## Security Architecture

### Layer 1: Network Security

```
┌─────────────┐
│   Browser   │
│   (HTTPS)   │
└──────┬──────┘
       │ TLS 1.3
       │ Certificate Validation
       │
┌──────▼───────────┐
│  Orchestrator    │
│  (mTLS Client)   │
└──────┬───────────┘
       │ mTLS (Mutual TLS)
       │ Client Certificate Required
       │ Device UUID Binding
       │
┌──────▼───────────┐
│ Hardware Server  │
│ (mTLS Server)    │
└──────────────────┘
```

**Protocols**:
- **UI ↔ Orchestrator**: HTTPS (TLS 1.3)
- **Orchestrator ↔ Hardware**: gRPC with mTLS (framework implemented, production deployment required)
- **No plaintext communication** in production

---

### Layer 2: Authentication & Authorization

```
User → Login → Orchestrator → JWT Token → Session
                    ↓
              Role Assignment
                    ↓
         ┌──────────┼──────────┐
         │          │          │
    Operator    Engineer    Admin
         │          │          │
   Read Scans   Calibrate   Configure
   Run Scans    Maintain    User Mgmt
```

**Roles** (RBAC):
1. **Operator**: Run measurements, view results
2. **Engineer**: Calibration, maintenance, diagnostics
3. **Administrator**: User management, configuration, system settings

**Session Management**:
- JWT tokens with expiration
- Device lock (exclusive access)
- Session timeout (configurable)
- Audit trail for all actions

---

### Layer 3: Physical Safety System

```
┌──────────────────────────────────────────────┐
│            Safety State Machine              │
│  ┌────────┐   ┌──────────┐   ┌──────────┐    │
│  │ LOCKED │──→│   IDLE   │──→│ RUNNING  │    │
│  └────────┘   └──────────┘   └──────────┘    │
└───────────────────┬───────────────────────────┘
                   │ Interlocks Monitored
                   │
         ┌─────────┬─────────┐
         │         │         │
    ┌────┴────┐ ┌──┴───┐ ┌──┴────────────┐
    │E-Stop   │ │Door  │ │Beam-Stop   │
    │Not      │ │Closed│ │Position    │
    │Pressed  │ │      │ │Sensor      │
    └─────────┘ └──────┘ └─────────────┘
         │         │         │
    ┌────┴────┐ ┌──┴────┐ ┌──┴──────────┐
    │Key      │ │Cooling│ │Power OK   │
    │Switch   │ │OK     │ │           │
    │ON       │ │       │ │           │
    └─────────┘ └───────┘ └───────────┘
```

**Safety Requirements**:
- Key Switch: Physical activation
- Activation Button: 20-second timeout for non-safe operations
- Emergency Stop: Immediate halt (<100ms)
- Interlocks: Hardware + software monitoring
- Beam-Stop Position: Mechanical beam block position verification

---

### Layer 4: Data Integrity

```
┌─────────────────────────────────────────────────────┐
│                  Audit Trail                         │
│  • All commands logged (who, what, when, why)       │
│  • Measurement data with checksums                   │
│  • State changes recorded                            │
│  • Immutable SQLite database                         │
│  • AES-256-GCM encryption at rest                    │
└─────────────────────────────────────────────────────┘
         │
         │ Every 24 hours
         │
┌────────▼──────────────────────────────────────────┐
│           Calibration Requirement                  │
│  • 24-hour validity period                         │
│  • 5-stage QC cascade                              │
│  • System locks if expired                         │
│  • Automatic validation                            │
└────────────────────────────────────────────────────┘
```

**Data Protection**:
- Encryption: AES-256-GCM for sensitive data
- Checksums: Data integrity verification
- Audit logs: Complete command history
- Backups: Automated (future)
- TPM integration: Hardware key storage (planned)

---

## Communication Flow

### 1. Measurement Workflow

```
UI                Orchestrator           Hardware Server
│                       │                        │
│  Start Scan          	│                        │
├──────────────────────>│                        │
│                       │  Check Calibration     │
│                       │  Valid?                │
│                       ├───────────────────────>│
│                       │<───────────────────────┤
│                       │  Yes (valid < 24h)     │
│                       │                        │
│                       │  Initialize Detector   │
│                       ├───────────────────────>│
│                       │  (Requires Key+Button) │
│                       │<───────────────────────┤
│                       │  Ready                 │
│                       │                        │
│                       │  Home XY Stage         │
│                       ├───────────────────────>│
│                       │<───────────────────────┤
│                       │  Homed (2s)            │
│                       │                        │
│                       │  Move to Position      │
│                       ├───────────────────────>│
│                       │<───────────────────────┤
│                       │  In Position           │
│                       │                        │
│                       │  Start Exposure        │
│                       ├───────────────────────>│
│  Status Updates       │  (Safety checks)       │
│<──────────────────────┤<───────────────────────┤
│  "Exposing: 5s"       │  Exposure Running      │
│                       │                        │
│                       │<───────────────────────┤
│  Measurement Complete │  Data Ready            │
│<──────────────────────┤                        │
```

### 2. Safety Check Example

```
Command: Start Exposure
    │
    ├─> Check: Calibration valid? ────> NO ──> REJECT (LOCKED)
    │                                   YES  (Exception: calibration exposures bypass)
    │                                    │
    ├─> Check: Key switch ON? ───────> NO ──> REJECT
    │                                   YES
    │                                    │
    ├─> Check: Interlocks safe? ─────> NO ──> REJECT (SAFE)
    │                                   YES
    │                                    │
    ├─> Check: Beam-stop position? ───> BEAM_CLOSED ──> REJECT
    │   (for calibration)                │
    │                                   BEAM_OPEN
    │                                    │
    └─> Execute Command ─────────────┘

**Calibration Enforcement Authority**: Hardware Server is authoritative.
Orchestrator checks are for user feedback only.
```

---

## Key Components

### Hardware Server Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| gRPC Services | Tonic | Device control API 		|
| Safety State Machine | Rust | Enforce safety rules |
| Audit Logger | SQLite | FDA compliance |
| Detector Driver | Async Rust | X-ray image acquisition |
| Motion Controller | Async Rust | XY stage positioning |
| GPIO Interface | Async Rust | Safety interlocks |
| Calibration QC | Rust | 5-stage validation |

### Orchestrator Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| FastAPI Server | Python | REST API endpoints |
| gRPC Client | grpcio | Hardware communication |
| WebSocket Handler | FastAPI | Real-time UI updates |
| Workflow Engine | Python | Measurement sequences |
| Authentication | JWT | User session management |
| Database | SQLAlchemy | Persistent storage |

---

## Deployment Architecture

### Development/Demo Mode

```
┌─────────────────────────────────────────┐
│        Single Windows PC                │
│                                         │
│  ┌────────────┐  ┌──────────────────┐   │
│  │ Browser    │  │ Hardware Server  │   │
│  │ (UI)       │  │ (Rust)           │   │
│  └─────┬──────┘  │ • Demo Detector  │   │
│        │         │ • Demo Motion    │   │
│  ┌─────▼──────┐  │ • Demo GPIO      │   │
│  │Orchestrator│  │ • GUI enabled    │   │
│  │(Python)    │  └──────────────────┘   │
│  └────────────┘                         │
└─────────────────────────────────────────┘
```

### Production Mode

```
┌──────────────┐         ┌──────────────────┐
│   Client PC  │         │   Server PC      │
│   (Browser)  │◄───────►│  Orchestrator    │
└──────────────┘  HTTPS  └─────────┬────────┘
                                   │ mTLS
                         ┌─────────▼─────────┐
                         │  Hardware Server  │
                         │  (Rust)           │
                         └────────┬──────────┘
                                  │ Hardware
                         ┌────────▼──────────┐
                         │  Physical Devices │
                         │  • X-ray Detector │
                         │  • XY Stage       │
                         │  • Safety I/O     │
                         └───────────────────┘
```

---

## Regulatory Compliance

### FDA 21 CFR Part 11

**Requirements Met**:
- ✅ Audit trail (all user actions logged)
- ✅ Electronic signatures (user authentication)
- ✅ Record integrity (checksums, encryption)
- ✅ Access control (role-based permissions)
- ✅ System validation (automated QC checks)
- ✅ Secure storage (encrypted database)

**Audit Trail Contents**:
- User ID and authentication method
- Date and time stamp
- Command type and parameters
- Device state before/after
- Execution result (success/failure)
- Session context

---

## Performance Specifications

| Operation | Duration | Notes |
|-----------|----------|-------|
| System Startup | 10-30s | Includes device initialization |
| Calibration | 5-300s | Demo: 5s, Production: 5min |
| Homing | 2-20s | Demo: 2s, Production: varies |
| Single Exposure | 0.1-300s | Configurable exposure time |
| Data Transfer | <1s | Per measurement |
| State Updates | 100ms | Real-time WebSocket |

---

## Future Enhancements

### Security
- [ ] TPM-based key storage
- [ ] Hardware Security Module (HSM) integration
- [ ] Two-factor authentication (2FA)

### Features
- [ ] Real detector integration (Bruker, AdvaCam)
- [ ] Real motion controllers (Thorlabs, Newport)
- [ ] Cloud backup and disaster recovery
- [ ] Remote diagnostics (secure tunnel)
- [ ] Multi-device orchestration

---

## Quick Reference

### Ports
- UI: `https://localhost:3000`
- Orchestrator: `https://localhost:8000`
- Hardware Server: `grpc://localhost:50051`

### File Locations
- Audit logs: `data/audit.db`
- Configuration: `config/server.toml`
- Certificates: `certs/`
- Measurements: `data/measurements/`

### Key Commands
```bash
# Start hardware server
cargo run --release

# Start orchestrator
python -m omniscan_orchestrator

# Start UI
npm start

# Check logs
tail -f logs/server.log
```

---

## Support & Documentation

- **Architecture**: See `docs/ARCHITECTURE.md`
- **API Reference**: See `API_DOCUMENTATION.md`
- **Calibration**: See `CALIBRATION.md`
- **Detector**: See `DETECTOR.md`
- **Motion**: See `MOTION.md`
- **GPIO**: See `GPIO.md` (to be created)
- **Safety**: See `HARDWARE_COMPONENTS.md`

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-29  
**System Version**: 0.2.0
