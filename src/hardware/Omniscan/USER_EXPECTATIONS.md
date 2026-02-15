# Omniscan Software - User Expectations

**Document Version:** 1.0  
**Date:** October 24, 2025  
**Classification:** Design History File - User Expectations  
**Purpose:** Define what users expect from Omniscan software functionality

---

## Overview for Non-Technical Readers

The Omniscan software is the central control system for a medical X-ray diffraction diagnostic device used in clinical laboratories. Think of it as the "brain" that makes sure the medical device operates safely, accurately, and reliably when analyzing patient samples.

### What the Software Does

The software has three primary responsibilities:

**1. Safety Guardian**  
The software acts as a constant safety monitor, ensuring that X-ray equipment never operates under unsafe conditions. It continuously checks safety doors, emergency buttons, and other protective systems. If anything is wrong—like a door being open—the software immediately prevents the X-ray from turning on. This protects laboratory staff from radiation exposure.

**2. Precision Controller**  
The software precisely controls how long and at what intensity the X-ray beam operates during each patient sample test. It monitors the beam in real-time and automatically stops if anything goes outside safe parameters. This ensures every diagnostic measurement is accurate and consistent.

**3. Data Protector**  
The software saves every measurement immediately and permanently, linking it to the patient sample, the operator who ran the test, and the calibration used. All data is encrypted and cannot be lost—even if the computer crashes or the internet connection fails. This creates a complete, traceable record required for medical diagnostics and regulatory compliance.

### Why This Software Matters

In a clinical laboratory, staff run dozens of patient sample tests every day. The software must:
- **Be trustworthy**: Lab technicians need confidence that the device will never operate unsafely
- **Be reliable**: Tests cannot be delayed because software crashed or data was lost
- **Be simple**: Busy lab staff don't have time for complicated interfaces or confusing error messages
- **Be compliant**: Hospitals and regulatory agencies need complete records of every test performed

The software runs continuously in the background, managing safety systems, controlling hardware, storing data, and helping users troubleshoot problems—all while maintaining the detailed records required for FDA-regulated medical devices.

---

## Introduction

This document outlines the key expectations that users have for the Omniscan medical diagnostic software. These expectations reflect the needs of laboratory technicians, maintenance engineers, and IT administrators who will interact with the software daily.

**Target Users:**
- **Clinical Operators** - Laboratory technicians performing diagnostic measurements
- **Maintenance Engineers** - Service personnel performing calibration and maintenance
- **IT Administrators** - Personnel managing system configuration and data security

---

## 1. Safety and Radiation Protection

### Absolute Protection from X-ray Exposure

The software must guarantee that operators are never exposed to X-ray radiation under unsafe conditions. This includes:

- Preventing X-ray beam activation when safety doors are not fully closed
- Immediately terminating all operations upon emergency stop button activation
- Continuously monitoring safety interlocks and displaying real-time status
- Automatically transitioning to safe state upon any fault detection
- Providing clear visual and audible warnings when safety systems are compromised

### Controlled Maintenance Access

The software must provide maintenance engineers with diagnostic capabilities while maintaining safety accountability:

- Maintenance mode requiring multi-factor authentication (password and physical key switch)
- Explicit logging of all safety system bypasses during maintenance
- Automatic restoration of full safety mode when maintenance session ends
- Complete audit trail of all maintenance activities

---

## 2. Precise X-ray Exposure Control

### Accurate Dose Management

The software must control X-ray exposure with precision to protect patient samples and ensure diagnostic quality:

- Execute X-ray exposure for the exact programmed duration
- Monitor beam intensity in real-time and terminate if values deviate from acceptable ranges
- Prevent multiple concurrent exposures through software controls
- Display countdown timers and exposure status in real-time
- Record all exposure parameters with precise timestamps

### Quality Assurance

The software must provide indicators enabling assessment of measurement validity:

- Flag measurements where exposure parameters deviated from programmed values
- Report beam intensity variations with statistical data
- Identify any data collected during safety warnings or faults
- Clearly distinguish validated measurements from questionable data

---

## 3. Data Integrity and Security

### Zero Data Loss

The software must ensure that no measurement data is lost under any circumstances:

- Commit data to persistent storage immediately after acquisition
- Implement fault-tolerant storage surviving system crashes and power failures
- Automatically associate measurements with patient sample identifiers
- Prevent accidental deletion or overwriting of historical data
- Maintain full operation during network outages with local data buffering

### HIPAA-Compliant Security

The software must protect patient data according to regulatory requirements:

- Encrypt all data at rest using AES-256-GCM or equivalent
- Encrypt data in transit using TLS 1.2 minimum
- Log all data access events with user identification
- Store credentials and encryption keys in hardware-protected storage (TPM)
- Provide automated backup with verification mechanisms

### Complete Traceability

The software must automatically link every measurement to:

- Device identification (serial number, hardware configuration)
- Operator identification (username, authentication timestamp)
- Temporal information (timestamp with timezone)
- Calibration reference (calibration ID and validity status)
- Measurement parameters (exposure settings, beam intensity)
- Quality metrics (SNR, beam stability, distance verification)

---

## 4. Calibration Management

### Daily Calibration Enforcement

The software must enforce daily calibration to maintain diagnostic accuracy:

- Block diagnostic measurements if calibration has not been performed within 24 hours
- Display prominent calibration status on all interfaces
- Provide guided workflow with clear step-by-step instructions
- Validate calibration results automatically against acceptance criteria
- Display clear pass/fail indication with corrective action guidance
- Transition system to LOCKED state when calibration expires

### Diagnostic Support for Engineers

The software must assist maintenance engineers in troubleshooting calibration issues:

- Display detailed calibration parameters and measured values
- Provide graphical trending of calibration results over time
- Indicate which specific parameters failed acceptance criteria
- Maintain historical data for failure pattern analysis
- Support multiple calibration attempts for repair verification

---

## 5. Usability and Workflow Efficiency

### Intuitive Operator Interface

The software must support efficient clinical workflow:

- Measurement initiation achievable in maximum 4 clicks
- Large, clearly visible buttons and status indicators
- Status display in plain language: "Ready", "Running", "Calibration Required"
- Real-time status updates with minimal lag
- Error messages providing actionable troubleshooting steps
- Minimal menu navigation for common operations

### Clear Error Handling

The software must help operators resolve problems effectively:

- Plain language error descriptions for operators
- Numbered checklists of troubleshooting steps
- Clear indication of when to contact support
- Technical details logged separately for maintenance personnel
- Unique error reference numbers for support coordination

---

## 6. System Reliability and Availability

### High Availability

The software must be operational when users need it:

- Automatic startup as system service on computer boot
- Automatic hardware reconnection after power cycling
- Target uptime of 99% during operational hours
- Graceful handling of Windows updates without data loss
- Maximum restart time of 3 minutes to operational state

### Supportability

The software must enable IT administrators to maintain the system:

- Structured log files with configurable detail levels
- Operation under standard user accounts (no administrator rights required)
- Specific error messages including component identification
- Service restart capability without full system reboot
- Compatibility with standard Windows updates and antivirus software

---

## 7. Multi-User Environment and Access Control

### Session Management

The software must clearly manage multiple users:

- Display current operator name and session status
- Show estimated time remaining for active measurements
- Prevent session interference through exclusive access control
- Automatic session termination on logout or disconnect
- Concurrent read-only monitoring for supervisors

### Role-Based Access Control

The software must enforce appropriate permissions:

- Clinical operators: Measurement execution and calibration
- Maintenance engineers: Full diagnostic access and maintenance mode
- Administrators: User management and system configuration
- Permission enforcement with cryptographic authentication
- Failed access attempt logging

---

## 8. Regulatory Compliance and Audit Support

### Complete Audit Trail

The software must maintain comprehensive logs supporting FDA and hospital audits:

**Events to be logged:**
- All measurement operations with complete parameters
- User authentication and authorization events
- Configuration changes with before/after values
- Safety system status changes and interlock events
- Calibration results and validation outcomes
- Software updates and version changes

**Audit trail requirements:**
- Append-only storage preventing retroactive modification
- Cryptographic integrity protection
- Exportable in standard formats (CSV, PDF, JSON)
- Searchable and filterable by time, user, and event type
- Retention for minimum 7 years

### Quality Analytics

The software must support continuous improvement:

- Measurement success/failure rate reporting
- Calibration trend analysis
- Sample re-run analysis with root cause tracking
- Quality control charts for key parameters
- Equipment utilization metrics

---

## 9. Network Integration and Offline Operation

### Offline Capability

The software must operate independently of network availability:

- Complete measurements without cloud connectivity
- Local data storage with automatic synchronization when network restores
- Clear visual indication of connectivity status
- No performance degradation during network outages
- Local storage monitoring with capacity alerts

### Network Management

The software must provide IT administrators with oversight:

- Real-time cloud connectivity status
- Upload queue monitoring
- Failed upload notifications with manual retry
- Bandwidth throttling controls
- Cloud synchronization pause/resume capability

---

## 10. Software Updates and Maintenance

### Controlled Updates

The software must minimize disruption from updates:

- Comprehensive release notes explaining changes
- Staged deployment capability (test before production)
- Rollback capability if issues arise
- Scheduled installation during non-operational hours
- Advance notification (minimum 48 hours)
- Preservation of calibration data during updates
- User confirmation required before installation

---

## 11. Performance Expectations

The software must meet performance benchmarks supporting clinical workflow:

| Operation | Expected Response Time |
|-----------|----------------------|
| Measurement initiation | < 2 seconds |
| Emergency stop response | < 100 milliseconds |
| State transition display | < 1 second |
| Data commit to storage | < 5 seconds |
| Calibration routine | < 10 minutes |
| Historical data search | < 5 seconds |

**System capacity:**
- Support 100 measurements per day per device
- Maintain 10 years of historical data locally
- Continuous 12-hour daily operation without degradation
- 30-day uptime without restart requirement

---

## 12. Training and Documentation

### Efficient Training

The software must support rapid user proficiency:

- Clinical operators reach basic proficiency in 4 hours
- In-application help system with context-sensitive guidance
- Visual workflow guides for common operations
- Quick reference cards for operators

### Comprehensive Documentation

The software must be supported by clear documentation:

- User manual with step-by-step procedures
- Administrator guide for configuration and maintenance
- Troubleshooting guide with flowcharts
- Emergency procedure reference materials

---

## Summary

The Omniscan software must prioritize three fundamental expectations:

1. **Safety First**: Absolute protection from X-ray exposure through software-enforced interlocks and automatic fault detection.

2. **Data Integrity**: Zero tolerance for data loss, with immediate persistence, encryption, and complete traceability for every measurement.

3. **Clinical Reliability**: High availability, offline operation capability, and intuitive interfaces enabling efficient clinical workflow.

These expectations form the foundation for a medical diagnostic system that users can trust for safe, accurate, and reliable patient sample analysis.

---

## Document Control

**Created:** October 24, 2025  
**Author:** Omniscan Development Team  
**Review Cycle:** Annually or when major software changes are planned  
**Classification:** FDA Design History File (DHF) - User Expectations  

**Next Review:** Prior to FDA submission

---

**End of Document**
