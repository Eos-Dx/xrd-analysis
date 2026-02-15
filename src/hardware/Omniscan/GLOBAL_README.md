# 🩺 Omniscan Medical Diagnostic Platform

[![FDA Compliance](https://img.shields.io/badge/FDA-IEC%2062304%20Class%20B-blue.svg)](https://www.fda.gov/medical-devices/software-medical-device-samd)
[![Safety Critical](https://img.shields.io/badge/Safety-Critical-red.svg)](https://en.wikipedia.org/wiki/Safety-critical_system)
[![Medical Device](https://img.shields.io/badge/Medical-X--Ray%20Diffraction-purple.svg)]()

A **first-of-kind FDA-compliant medical diagnostic platform** utilizing X-ray Diffraction (XRD) technology for patient sample analysis. The system integrates safety-critical hardware control, secure maintenance operations, and comprehensive audit logging to meet IEC 62304 Class B and FDA De Novo requirements.

---

## 🏥 System Overview

Omniscan is a distributed medical device platform designed for clinical laboratory environments, combining real-time hardware control with secure cloud-based processing and reporting.

### Clinical Application
- **Primary Use**: X-ray diffraction analysis of patient samples
- **FDA Pathway**: De Novo (first-of-kind device)
- **Compliance Standards**: FDA/IEC 62304 Class B, ISO 14971, IEC 62366-1, ISO 13485, ISO 27001
- **Target Environment**: Clinical laboratories, medical diagnostic facilities
- **Safety Classification**: Class B safety-critical medical device software

### Key Features
- ✅ **Safety-Critical Control**: Hardware-enforced safety interlocks and watchdogs
- ✅ **mTLS Authentication**: Mutual TLS with device-scoped engineer certificates
- ✅ **Comprehensive Audit**: Encrypted, tamper-proof audit logging
- ✅ **Daily Calibration**: Mandatory 24-hour calibration enforcement
- ✅ **FDA Traceability**: Complete chain from requirements to deployment
- ✅ **Offline Operation**: Local buffering with delayed cloud synchronization

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OMNISCAN ECOSYSTEM                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐    REST/WS     ┌──────────────────────┐    │
│   │  Browser UI     │◄───────────────►│  Orchestrator (Py)   │    │
│   │  (Clinician)    │                 │  - Workflow logic    │    │
│   └─────────────────┘                 │  - Local buffering   │    │
│                                        │  - Cloud sync        │    │
│                                        └──────────┬───────────┘    │
│                                                   │                 │
│                                              gRPC/mTLS              │
│                                                   │                 │
│                                        ┌──────────▼───────────┐    │
│                                        │  HW Server (Rust)    │    │
│                                        │  - Safety authority  │    │
│                                        │  - Interlock control │    │
│   ┌────────────────┐    mTLS/HTTPS    │  - Audit logging     │    │
│   │  Maintenance   │◄───────────────► │  - Device drivers    │    │
│   │  Engineer CLI  │                  └──────────┬───────────┘    │
│   └────────────────┘                             │                 │
│                                                   │ TCP/DLL/GPIO    │
│                                        ┌──────────▼───────────┐    │
│                                        │  Medical Hardware    │    │
│                                        │  - X-ray detectors   │    │
│                                        │  - Motion devices    │    │
│                                        │  - PDU (power)       │    │
│                                        │  - GPIO interlocks   │    │
│                                        │  - Temp/vac (future) │    │
│                                        └──────────────────────┘    │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │                    Cloud Platform (AWS)                     │ │
│   │  - Ingress queues      - ML inference                       │ │
│   │  - Processing pipeline - Reporting services                 │ │
│   │  - Data storage        - Model governance                   │ │
│   └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Subprojects

### 1. 🦀 **omniscan-hw-server** (Rust)
**Safety-critical hardware control server**

**Purpose**: Exclusive authority over X-ray beam, motion systems, and safety interlocks. Implements comprehensive safety state machine and FDA-compliant audit logging.

**Key Features**:
- Safety state machine (IDLE → PENDING_ARMED → RUNNING → SAFE)
- Physical safety controls (E-stop, door interlocks, beam watchdog)
- gRPC services: Acquisition, Motion, Health, Safety, Beam Monitoring, Calibrant Quality
- Encrypted audit database with TPM-signed daily rolls
- Device abstraction layer (detectors, motion devices, PDU, GPIO)
- mTLS server authentication with certificate validation

**Technology Stack**:
- Rust 1.70+
- Tokio async runtime
- Tonic gRPC framework
- SQLite audit database
- AES-256-GCM encryption

**Quick Start**:
```bash
cd omniscan-hw-server
cargo build --release
cargo run -- --config config/server.toml --grpc-addr "[::1]:50051"
```

**Documentation**:
- [README.md](omniscan-hw-server/README.md) - Comprehensive hardware server documentation
- [MTLS_SETUP.md](omniscan-hw-server/MTLS_SETUP.md) - mTLS configuration guide

**Compliance**: 
- ✅ USR_OMNI-SERVER-001..003 (User requirements)
- ✅ SYS_OMNI-SERVER-001..010 (System requirements)
- ⚠️ RISK_OMNI-SERVER-001..005 (Risk controls)

---

### 2. 🐍 **omniscan-orchestrator** (Python)
**Maintenance operations CLI with certificate management**

**Purpose**: Command-line interface for maintenance engineers to perform device configuration, enter maintenance mode, and execute secure operations.

**Key Features**:
- Engineer certificate generation and management
- mTLS client authentication
- USB maintenance stub (development mode)
- Configuration management (get, set, patch, validate)
- Maintenance mode control (enter, renew, exit)
- Device status monitoring

**Technology Stack**:
- Python 3.11+
- Typer CLI framework
- Requests with mTLS support
- Cryptography library
- Rich terminal output

**Quick Start**:
```powershell
cd omniscan-orchestrator
pip install -e .

# Generate engineer certificate
omni-orch cert generate --engineer-id ENG001 --device-uuid ABC123

# Check server status
omni-orch status --cert <cert> --key <key> --ca-cert <ca> --base-url https://localhost:8443/api/v1
```

**Documentation**:
- [README.md](omniscan-orchestrator/README.md) - CLI command reference
- [QUICKSTART.md](omniscan-orchestrator/QUICKSTART.md) - 5-minute getting started guide
- [ENGINEER_CERTIFICATES.md](omniscan-orchestrator/ENGINEER_CERTIFICATES.md) - Complete certificate guide
- [IMPLEMENTATION_SUMMARY.md](omniscan-orchestrator/IMPLEMENTATION_SUMMARY.md) - Technical implementation details

---

### 3. 🔐 **omniscan-certificate-center** (Python)
**PKI certificate generation toolkit**

**Purpose**: Generate and manage PKI certificates following the OmniSoft Certificate Strategy for device servers and maintenance clients.

**Key Features**:
- Separate Root CAs for servers and clients
- Long-lived server certificates (1 year, TPM-backed)
- Short-lived client certificates (1 day default, device-scoped)
- ECDSA P-256 or RSA-3072 with SHA-256
- FDA/IEC compliance-aligned cryptography

**Technology Stack**:
- Python 3.11+
- Cryptography library (x509 certificates)
- OpenSSL-compatible PEM format

**Quick Start**:
```bash
cd omniscan-certificate-center
pip install -r requirements.txt
python certgen.py init

# Create Root CAs
python certgen.py create-root --type server
python certgen.py create-root --type client

# Create server certificate
python certgen.py create-server --device-uuid ABC123

# Create client certificate
python certgen.py create-client --engineer-id ENG001 --device-uuid ABC123 --validity-days 1
```

**Certificate Directory Structure**:
```
certs/
├── root/          # Root CA certificates (offline/secure storage)
├── server/        # Server certificates for OMNIScan devices
└── client/        # Client certificates for maintenance engineers
```

**Documentation**:
- [README.md](omniscan-certificate-center/README.md) - Certificate generation guide

**Security Strategy**:
- Follows [OmniSoft Certificate Strategy](info/OmniSoft_Certificate_Strategy.md)
- Mutual TLS (mTLS) with device UUID validation
- Short-lived client certs for time-limited access
- Tamper-evident audit logging

---

### 4. 🌐 **omniscan-ui** (Web Frontend)
**Browser-based clinician interface**

**Status**: 🚧 Placeholder (future development)

**Planned Features**:
- Sample registration and tracking
- Real-time measurement monitoring
- Calibration status display
- Historical data review
- Quality control dashboard

---

### 5. 📋 **info/** (Documentation)
**Requirements, specifications, and strategy documents**

**Contents**:
- [OMNIScan_Requirements_v2_Tagged.md](info/OMNIScan_Requirements_v2_Tagged.md) - Doorstop-compatible requirements with traceability tags
- [OmniSoft_Certificate_Strategy.md](info/OmniSoft_Certificate_Strategy.md) - PKI architecture and security design
- REQ/ - Additional requirements artifacts (TBD)

---

## 🔐 Security & PKI Architecture

### Certificate Hierarchy
```
Device Server Root CA ──┬──► Server Certificates (1 year)
                        │     └─► OMNIScan Hardware Servers
                        │
Maintenance Client Root CA ──► Client Certificates (1-7 days)
                                └─► Maintenance Engineers
```

### Authentication Flow
1. **Server Authentication**: Hardware server presents server certificate signed by Device Server Root CA
2. **Client Authentication**: Engineer/orchestrator presents client certificate signed by Maintenance Client Root CA
3. **Device Scope Validation**: Server verifies client certificate SAN contains `urn:omniscan:server:<UUID>`
4. **Access Control**: Server grants access only if certificate is valid, not expired, and device UUID matches
5. **Audit Logging**: All actions logged with certificate serial number and engineer ID

### Key Security Features
- **mTLS**: Mutual authentication for all communications
- **Device Scoping**: Engineer certificates restricted to specific devices
- **Short Validity**: Client certificates expire after 1 day (configurable)
- **Encrypted Audit**: AES-256-GCM encrypted logs with TPM-signed daily rolls
- **Physical Access**: Key switch and enable button requirements

---

## 🚀 Getting Started

### Prerequisites
- **Operating System**: Windows 10/11 (primary target), Linux (development)
- **Rust**: 1.70+ with MSVC toolchain (Windows)
- **Python**: 3.11+
- **Protocol Buffers**: protoc compiler for gRPC
- **SQLite**: Audit database (included in Rust/Python)

### Initial Setup

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd Omniscan
   ```

2. **Setup Certificate Infrastructure**
   ```bash
   cd omniscan-certificate-center
   pip install -r requirements.txt
   python certgen.py create-root --type server
   python certgen.py create-root --type client
   python certgen.py create-server --device-uuid ABC123
   cd ..
   ```

3. **Build Hardware Server**
   ```bash
   cd omniscan-hw-server
   cargo build --release
   cd ..
   ```

4. **Install Orchestrator CLI**
   ```bash
   cd omniscan-orchestrator
   pip install -e .
   cd ..
   ```

5. **Generate Engineer Certificate**
   ```bash
   omni-orch cert generate --engineer-id ENG001 --device-uuid ABC123 --validity-days 1
   ```

6. **Start Hardware Server**
   ```bash
   cd omniscan-hw-server
   cargo run --release -- --config configs/server_with_mtls.json
   ```

7. **Test Connection**
   ```bash
   omni-orch status \
     --cert ../omniscan-certificate-center/certs/client/client_ENG001_ABC123.crt \
     --key ../omniscan-certificate-center/certs/client/client_ENG001_ABC123.key \
     --ca-cert ../omniscan-certificate-center/certs/root/device_server_root_ca.crt \
     --base-url https://localhost:8443/api/v1
   ```

---

## 📋 Requirements Traceability

### User Requirements (USR_*)
| ID | Subsystem | Title | Status |
|----|-----------|-------|--------|
| USR_OMNI-SERVER-001 | Hardware Server | Reliable and safe operation | ✅ Complete |
| USR_OMNI-SERVER-002 | Hardware Server | Daily calibration capability | ✅ Complete |
| USR_OMNI-SERVER-003 | Hardware Server | Data integrity and traceability | ✅ Complete |
| USR_OMNI-ORCH-* | Orchestrator | (Reserved) | 🚧 Planned |
| USR_OMNI-UI-* | UI | (Reserved) | 🚧 Planned |

### System Requirements (SYS_*)
| ID | Subsystem | Title | Status |
|----|-----------|-------|--------|
| SYS_OMNI-SERVER-001 | Hardware Server | Safety authority | ✅ Complete |
| SYS_OMNI-SERVER-002 | Hardware Server | Interlocks and watchdogs | ✅ Complete |
| SYS_OMNI-SERVER-003 | Hardware Server | Operational states | ✅ Complete |
| SYS_OMNI-SERVER-004 | Hardware Server | Calibration enforcement | ✅ Complete |
| SYS_OMNI-SERVER-005 | Hardware Server | Beam fault handling | ⚠️ Framework |
| SYS_OMNI-SERVER-006 | Hardware Server | Data retention | ✅ Complete |
| SYS_OMNI-SERVER-007 | Hardware Server | Audit database | ✅ Complete |
| SYS_OMNI-SERVER-008 | Hardware Server | gRPC interface | ✅ Complete |
| SYS_OMNI-SERVER-009 | Hardware Server | Session security | ⚠️ Framework |
| SYS_OMNI-SERVER-010 | Hardware Server | Safety profile updates | ✅ Complete |

### Risk Controls (RISK_*)
| ID | Hazard | Mitigation Status |
|----|--------|-------------------|
| RISK_OMNI-SERVER-001 | Beam exposure when door open | ✅ Implemented |
| RISK_OMNI-SERVER-002 | Motion when E-stop pressed | ✅ Implemented |
| RISK_OMNI-SERVER-003 | Software watchdog failure | ⚠️ Framework |
| RISK_OMNI-SERVER-004 | Unauthorized configuration | ✅ Implemented |
| RISK_OMNI-SERVER-005 | Data loss during upload | ✅ Implemented |

**Legend**: ✅ Complete | ⚠️ Framework Ready | 🚧 Planned | ❌ Not Started

---

## 🧪 Testing & Validation

### Unit Testing
```bash
# Hardware server tests
cd omniscan-hw-server
cargo test

# Orchestrator tests (when implemented)
cd omniscan-orchestrator
pytest tests/
```

### Integration Testing
```bash
# Start server in test mode
cd omniscan-hw-server
cargo run -- --config config/server.toml --skip-interlocks

# Run gRPC tests
grpcurl -plaintext localhost:50051 hub.v1.Health/Liveness
grpcurl -plaintext localhost:50051 hub.v1.Acquisition/GetState
```

### Compliance Testing
- **IEC 62304**: Unit test coverage ≥80% for safety modules
- **ISO 14971**: Risk control verification tests
- **FDA Traceability**: Requirements ↔ code ↔ tests mapping via Doorstop

---

## 🛠️ Development Tools

### Code Quality
```bash
# Rust linting and formatting
cd omniscan-hw-server
cargo fmt
cargo clippy -- -D warnings
cargo audit

# Python linting (when configured)
cd omniscan-orchestrator
ruff check src/
```

### Documentation Generation
```bash
# Rust API docs
cd omniscan-hw-server
cargo doc --open

# Requirements traceability (when Doorstop configured)
doorstop publish all docs/doorstop_html
```

### Build & Release
```bash
# Hardware server release build
cd omniscan-hw-server
cargo build --release --target x86_64-pc-windows-msvc

# Python package build
cd omniscan-orchestrator
python -m build
```

---

## 📊 Project Status

### Implemented ✅
- ✅ Hardware server core architecture
- ✅ Safety state machine
- ✅ gRPC services (Acquisition, Motion, Health, Safety)
- ✅ Encrypted audit logging
- ✅ Device abstraction layer (DEMO implementations)
- ✅ mTLS server configuration
- ✅ PKI certificate infrastructure
- ✅ Engineer certificate management CLI
- ✅ Orchestrator maintenance operations
- ✅ Configuration management (encrypted)

### In Progress 🚧
- 🚧 Real hardware driver integration (Bruker BIS, Thorlabs, GPIO)
- 🚧 Hardware watchdog implementation
- 🚧 Beam intensity monitoring
- 🚧 Cloud platform integration
- 🚧 Web UI development

### Planned 📋
- 📋 Windows Service deployment
- 📋 Clinical validation testing
- 📋 EMR/LIMS integration
- 📋 Advanced calibration routines
- 📋 Predictive maintenance features
- 📋 FDA submission package completion

---

## 📚 Documentation Index

### Core Documentation
- **This File**: Global project overview and quick start
- [omniscan-hw-server/README.md](omniscan-hw-server/README.md) - Hardware server comprehensive guide
- [omniscan-orchestrator/README.md](omniscan-orchestrator/README.md) - CLI command reference
- [omniscan-certificate-center/README.md](omniscan-certificate-center/README.md) - Certificate generation

### Quick Start Guides
- [omniscan-orchestrator/QUICKSTART.md](omniscan-orchestrator/QUICKSTART.md) - 5-minute setup for engineers
- [omniscan-hw-server/MTLS_SETUP.md](omniscan-hw-server/MTLS_SETUP.md) - mTLS configuration step-by-step

### Specifications & Strategy
- [info/OMNIScan_Requirements_v2_Tagged.md](info/OMNIScan_Requirements_v2_Tagged.md) - Tagged requirements for traceability
- [info/OmniSoft_Certificate_Strategy.md](info/OmniSoft_Certificate_Strategy.md) - PKI security architecture
- [omniscan-orchestrator/ENGINEER_CERTIFICATES.md](omniscan-orchestrator/ENGINEER_CERTIFICATES.md) - Certificate authentication guide

### Implementation Details
- [omniscan-orchestrator/IMPLEMENTATION_SUMMARY.md](omniscan-orchestrator/IMPLEMENTATION_SUMMARY.md) - Engineer auth implementation

---

## 🔮 Roadmap

### Phase 1: Hardware Integration (Weeks 1-4)
- Replace DEMO devices with real hardware drivers
- Integrate Bruker BIS detector (TCP protocol)
- Add Thorlabs motion controller (C DLL FFI)
- Implement PCIe GPIO for safety interlocks

### Phase 2: Security Hardening (Weeks 3-6)
- Complete mTLS certificate validation
- Integrate Windows Event Log
- Add OCSP/CRL revocation checking
- Implement hardware key storage (TPM)

### Phase 3: Medical Device Validation (Weeks 5-8)
- Beam intensity monitoring with thresholds
- Exposure timer watchdogs
- Calibration validation procedures
- Safety interlock verification testing

### Phase 4: Production Deployment (Weeks 7-10)
- Windows Service deployment
- Performance optimization
- Clinical environment integration
- FDA documentation completion

---

## 🤝 Contributing

### Development Standards
- **Rust**: Follow official Rust style guide, use `clippy` and `rustfmt`
- **Python**: PEP 8 compliance, type hints required
- **Safety**: All unsafe code confined to hardware FFI boundaries
- **Testing**: Minimum 80% coverage for safety-critical modules
- **Documentation**: Comprehensive inline and external docs

### Git Workflow
1. Create feature branch from `main`
2. Implement feature with tests
3. Run linters and tests locally
4. Submit pull request with requirements traceability tags
5. Code review with safety/compliance focus
6. Merge after approval and CI/CD pass

### Compliance Requirements
- All changes must link to requirements (USR_*, SYS_*, RISK_*)
- Safety-related changes require dual review
- Audit log format changes require compliance team approval
- Certificate changes must align with OmniSoft Certificate Strategy

---

## 📞 Support & Contact

### Project Team
- **Hardware Server**: Rust safety-critical team
- **Orchestrator**: Python operations team
- **Certificates**: PKI/security team
- **Compliance**: FDA regulatory team

### Issue Reporting
- **Safety Issues**: Report immediately to safety team
- **Security Issues**: Follow responsible disclosure process
- **Feature Requests**: Submit via issue tracker
- **Bug Reports**: Include logs, config, and reproduction steps

---

## 📄 License & Legal

**Copyright © 2024 OMNIScan Medical Systems**

This software is proprietary medical device software intended for FDA-regulated medical diagnostic equipment.

- **Compliance**: FDA/IEC 62304 Class B Medical Device Software
- **Safety Critical**: Safety-related medical device software
- **Regulatory Status**: Under FDA review (De Novo pathway)
- **Distribution**: Authorized personnel only

### ⚠️ IMPORTANT MEDICAL DEVICE NOTICE

This software controls X-ray emitting medical equipment. Improper use may result in radiation exposure. Only qualified personnel should operate this system. All safety procedures must be followed.

---

## 🏆 Acknowledgments

Built with safety, compliance, and patient care as the highest priorities. This project represents the cutting edge of medical device software development, combining Rust's memory safety with comprehensive FDA compliance frameworks.

**Technologies**:
- 🦀 Rust - Memory-safe systems programming
- 🐍 Python - Maintenance operations and tooling
- 🌐 gRPC/Tonic - High-performance RPC
- 🔒 Cryptography - Medical-grade security
- 🛡️ SQLite - Reliable audit logging
- ⚡ Tokio - Async concurrent operations

**Standards Compliance**:
- FDA Cybersecurity Guidance (2023)
- IEC 62304 Class B Medical Device Software
- ISO 14971 Risk Management
- IEC 62366-1 Human Factors
- ISO 13485 Quality Management System
- ISO 27001 Information Security
- HIPAA/GDPR Data Protection

---

*🩺 Building the future of medical diagnostics with safety, security, and compliance.*
