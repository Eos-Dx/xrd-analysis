# Omniscan Complete User Requirements Specification

**Document Version:** 2.0  
**Date:** November 17, 2025  
**Classification:** FDA Design History File (DHF) - User Requirements  
**Compliance:** IEC 62304 Class B, ISO 14971, IEC 62366-1, HIPAA, ISO 13485, ISO 27001  
**Status:** Consolidated from all subsystem requirements

---

## Executive Summary

This document consolidates all user requirements for the Omniscan medical X-ray diffraction (XRD) diagnostic platform. The system comprises three major subsystems:

1. **Hardware Server (Rust)** - Safety-critical hardware control system
2. **Orchestrator (Python)** - Clinical workflow coordination and data management
3. **User Interface (Web/React)** - Clinical operator and administrator interfaces

**Target User Groups:**
- **Clinical Operators** - Laboratory technicians performing diagnostic measurements
- **Clinicians** - Medical professionals reviewing diagnostic results and patient data
- **Maintenance Engineers** - Service personnel performing calibration, diagnostics, and maintenance
- **IT Administrators** - Personnel managing system configuration, security, and data management
- **Quality Managers** - Personnel overseeing compliance, audits, and quality assurance

---

## Table of Contents

1. [Safety and Radiation Protection](#1-safety-and-radiation-protection)
2. [X-ray Exposure Control](#2-x-ray-exposure-control)
3. [Data Integrity and Security](#3-data-integrity-and-security)
4. [Calibration Management](#4-calibration-management)
5. [User Interface and Usability](#5-user-interface-and-usability)
6. [System Reliability](#6-system-reliability)
7. [Multi-User and Access Control](#7-multi-user-and-access-control)
8. [Error Handling and Recovery](#8-error-handling-and-recovery)
9. [Software Update Management](#9-software-update-management)
10. [Regulatory Compliance and Audit Support](#10-regulatory-compliance-and-audit-support)
11. [Network and Cloud Integration](#11-network-and-cloud-integration)
12. [Performance Requirements](#12-performance-requirements)
13. [Training and Documentation](#13-training-and-documentation)
14. [Hardware Initialization and Power Management](#14-hardware-initialization-and-power-management)
15. [Motion Control](#15-motion-control)
16. [Complete Requirements Traceability](#16-complete-requirements-traceability)

---

## 1. Safety and Radiation Protection

### 1.1 Operator Safety Protection

**Requirement ID:** USR_OMNI-SERVER_001  
**Priority:** CRITICAL  
**Subsystem:** Hardware Server

**User Need:**
The hardware server must ensure safe beam and motion control under all conditions; interlocks must disable beam on any fault.

**What Users Expect:**

Clinical operators need absolute confidence that:
- The system will **never expose operators to X-ray radiation** under unsafe conditions
- **Physical safety interlocks** prevent operation when doors are open or emergency stop is pressed
- The device will **automatically halt** all operations if any safety condition is violated
- **Clear visual and audible indicators** show when the system is safe to approach
- Operations can be **immediately stopped** at any time using the emergency stop button

**Specific Requirements:**

The software shall provide absolute protection from X-ray exposure under all unsafe conditions:

- Prevent X-ray beam activation when safety doors are not fully closed under any circumstances
- Immediately terminate all hazardous operations upon emergency stop button activation
- Continuously monitor all safety interlocks and sensors with real-time status verification
- Automatically transition to SAFE state upon any safety fault detection
- Display clear visual and audible warnings when safety systems are not operational

**User Scenario:**
> *"As a laboratory technician, I need to know the X-ray machine will never turn on when the door is open, so I can safely load samples without risk of radiation exposure."*

**Safety Features Users Rely On:**
- Key switch must be in "operate" position
- Physical enable button confirmation before each measurement
- Door interlock sensors
- Emergency stop button (accessible at all times)
- Beam watchdog monitoring X-ray intensity
- Automatic transition to SAFE state on any fault

**Operational Scenario:** 
When a clinical operator loads a patient sample without fully closing the safety door, the software shall refuse to initiate X-ray exposure and display a clear message: *"Safety door not fully closed - X-ray initiation prohibited."*

**Verification:** 
- Safety interlock testing
- Fault injection testing
- Emergency stop response time measurement (< 100 milliseconds)

**Risk Controls:** 
- RISK_OMNI-SERVER_001: Door sensor interlock disables beam output
- RISK_OMNI-SERVER_002: Hardware E-stop line breaks power to actuators
- RISK_OMNI-SERVER_003: Independent hardware watchdog resets beam enable line

**Maps To:**
- SYS_OMNI-SERVER_001: Safety authority
- SYS_OMNI-SERVER_002: Interlocks and watchdogs
- SYS_OMNI-SERVER_003: Operational states

---

### 1.2 Maintenance Mode Safety

**Requirement ID:** USR_OMNI-SERVER_004  
**Priority:** CRITICAL  
**Subsystem:** Hardware Server

**User Need:**
Maintenance engineers require secure access mode for configuration updates and hardware maintenance while maintaining comprehensive safety controls and audit trails.

**What Maintenance Engineers Expect:**

- **Secure access mode** for configuration updates and hardware maintenance
- Ability to **bypass interlocks temporarily** for testing (with proper authorization)
- **Complete diagnostic information** for troubleshooting
- **Configuration management** with cryptographic signing to prevent unauthorized changes
- **Maintenance logs** automatically recorded

**Specific Requirements:**

The software shall provide controlled maintenance access while maintaining safety audit trails:

- Maintenance mode accessible only through multi-factor authentication (password + physical key switch)
- Explicit interlock bypass capability with comprehensive warning displays
- Real-time logging of all bypassed safety systems and maintenance actions
- Automatic safety mode restoration upon maintenance session termination
- Complete audit trail of maintenance activities with timestamps and operator identification

**Maintenance Workflow:**
1. Turn key switch to "maintenance" position
2. Authenticate with maintenance password and certificate
3. System enters **MAINTENANCE state**
4. Perform hardware service, configuration updates, or diagnostics
5. Run verification tests
6. Exit maintenance mode
7. System requires fresh calibration before clinical use

**Rationale:** 
Maintenance personnel require diagnostic capabilities while maintaining safety accountability.

**Verification:** 
- Maintenance mode access testing
- Audit log validation
- Automatic safety restoration testing

**Risk Controls:**
- RISK_OMNI-SERVER_004: Require signed update package and key switch activation
- RISK_OMNI-SERVER_006: Maintenance session logging

**Maps To:**
- SYS_OMNI-SERVER_011: Maintenance mode controls

---

## 2. X-ray Exposure Control

### 2.1 Precise Exposure Management

**Requirement ID:** USR_OMNI-SERVER_005  
**Priority:** CRITICAL  
**Subsystem:** Hardware Server

**User Need:**
The operator must be able to perform accurate diagnostic measurements with precise X-ray exposure control.

**What Clinical Operators Expect:**

The software shall provide precise control over X-ray exposure duration and intensity to ensure patient safety and data quality:

- Activate X-ray beam for the exact programmed exposure duration only
- Immediately terminate X-ray exposure if beam intensity exceeds or falls below defined thresholds
- Prevent concurrent exposure operations through mutex locking mechanisms
- Display real-time exposure status with clear visual indicators
- Maintain permanent records of all exposures with precise timestamps and exposure parameters

**Specific Requirements:**

- Execute X-ray exposure for the exact programmed duration with ±50ms tolerance
- Implement real-time beam intensity monitoring with defined threshold limits (±10% of target)
- Immediately terminate exposure if beam intensity deviates beyond thresholds
- Prevent concurrent exposure operations through software mutex controls
- Provide real-time visual feedback of exposure status and countdown timer
- Record all exposure parameters with millisecond-precision timestamps

**User Scenario:**
> *"When I start a 60-second measurement, the system should display a countdown timer. If something goes wrong with the X-ray source, the system should stop immediately and tell me what to do."*

**Operational Scenario:** 
During a 60-second diagnostic measurement, if the X-ray source experiences intensity fluctuation exceeding thresholds, the software shall immediately abort the exposure and alert the operator: *"X-ray beam intensity out of range. Measurement aborted. Contact maintenance services."*

**Rationale:** 
Precise exposure control ensures diagnostic quality and prevents over-exposure of samples.

**Verification:** 
- Exposure timing accuracy testing (±50ms)
- Beam intensity monitoring validation
- Concurrent operation prevention testing

**Risk Controls:**
- RISK_OMNI-SERVER_007: Beam intensity watchdog with automatic shutdown

**Maps To:**
- SYS_OMNI-SERVER_005: Beam fault handling
- SYS_OMNI-SERVER_012: Exposure control and timing

---

### 2.2 Measurement Quality Validation

**Requirement ID:** USR_OMNI-SERVER_006  
**Priority:** HIGH  
**Subsystem:** Hardware Server

**User Need:**
Clinicians must be able to assess measurement validity and reliability for diagnostic confidence.

**What Clinicians Expect:**

The software shall provide comprehensive quality indicators enabling clinicians to assess measurement validity:

- Flag measurements with non-compliant X-ray exposure parameters
- Report beam intensity variations throughout the measurement duration
- Identify data collected during active safety warnings or fault conditions
- Clearly distinguish between validated and questionable measurement data through visual indicators
- Provide quality metrics including SNR, beam stability, and calibration validity

**Specific Requirements:**

- Flag measurements where exposure parameters deviated from programmed values
- Report beam intensity variations with statistical analysis (mean, standard deviation)
- Identify data collected during active safety warnings or interlock violations
- Visually differentiate validated measurements from questionable data
- Provide quality metrics: SNR, beam stability, distance verification results, calibration validity

**User Scenario:**
> *"As a clinician reviewing results, I need clear indicators if a measurement was taken with unstable beam intensity or other quality issues, so I know whether to trust the diagnostic data."*

**Rationale:** 
Clinicians must assess measurement reliability for diagnostic confidence.

**Verification:** 
- Quality metric calculation validation
- Visual indicator testing
- Flagging accuracy assessment

**Risk Controls:**
- RISK_OMNI-SERVER_008: Data quality monitoring and flagging

**Maps To:**
- SYS_OMNI-SERVER_013: Quality validation and reporting

---

## 3. Data Integrity and Security

### 3.1 Data Persistence and Fault Tolerance

**Requirement ID:** USR_OMNI-SERVER_007  
**Priority:** CRITICAL  
**Subsystem:** Hardware Server

**User Need:**
All measurements and logs must remain traceable to device, operator, and calibration data. No data loss under any circumstances.

**What Users Expect:**

The software shall ensure zero measurement data loss under all operational conditions, including system failures and network interruptions:

- Commit measurement data to persistent storage immediately upon acquisition completion
- Implement fault-tolerant data storage with transaction logging to preserve data integrity during system crashes
- Automatically associate each measurement with the corresponding patient sample identifier
- Prevent accidental data overwrite or deletion through write-once mechanisms and access controls
- Maintain full operational capability during network connectivity loss with local data buffering

**Specific Requirements:**

- Commit measurement data to persistent storage immediately upon acquisition completion
- Implement transaction-based data storage with automatic rollback on incomplete operations
- Automatically associate measurements with patient sample identifiers
- Implement write-once data storage with access control preventing accidental deletion
- Maintain full operational capability during network outages with local buffering
- Implement graceful shutdown with data preservation on unexpected system termination

**User Scenario:**
> *"As a laboratory operator, if the computer crashes during a measurement, I expect the system to either save all the data collected or clearly tell me nothing was saved—never lose data silently."*

**Operational Scenario:** 
When a measurement is in progress and the operating system initiates an update-triggered reboot, the software shall either block the reboot until measurement completion or implement graceful shutdown with data preservation, providing explicit notification of measurement status.

**Rationale:** 
Patient diagnostic data is irreplaceable and must survive all failure scenarios.

**Verification:** 
- Power failure testing
- Crash recovery testing
- Network disconnection testing
- Data integrity validation

**Risk Controls:**
- RISK_OMNI-SERVER_005: Local database keeps all raw data until verified upload confirmation

**Maps To:**
- SYS_OMNI-SERVER_006: Data retention
- SYS_OMNI-SERVER_014: Fault-tolerant storage

---

### 3.2 Data Security and HIPAA Compliance

**Requirement ID:** USR_OMNI-SERVER_008  
**Priority:** CRITICAL  
**Subsystem:** All Subsystems

**User Need:**
Patient data must be protected according to HIPAA regulations and industry security standards.

**What IT Administrators Expect:**

The software shall implement comprehensive data protection controls compliant with HIPAA regulations:

- Encrypt all patient data stored locally using AES-256-GCM or equivalent encryption standards
- Encrypt data in transit to cloud services using TLS 1.3 or TLS 1.2 minimum
- Maintain comprehensive access logs recording all patient data access events with user identification
- Store credentials and encryption keys using secure key management systems (hardware TPM or equivalent)
- Provide automated backup capabilities with verification mechanisms to ensure backup integrity

**Specific Requirements:**

- Encrypt all patient data at rest using AES-256-GCM encryption
- Encrypt data in transit using TLS 1.3 or TLS 1.2 minimum
- Maintain access logs for all patient data operations with user identification
- Store credentials and encryption keys in hardware-protected storage (TPM)
- Implement automated backup with cryptographic verification
- Provide secure data export with maintained encryption
- Support data retention policies with automated archival

**Rationale:** 
Patient data protection is legally mandated under HIPAA regulations.

**Verification:** 
- Encryption validation
- Access logging verification
- Backup/restore testing
- Penetration testing

**Risk Controls:**
- RISK_OMNI-SERVER_004: Unauthorized configuration protection
- RISK_OMNI-SERVER_009: Data encryption and access control

**Maps To:**
- SYS_OMNI-SERVER_007: Audit database
- SYS_OMNI-SERVER_009: Session security
- SYS_OMNI-SERVER_015: Encryption and security

---

### 3.3 Complete Traceability

**Requirement ID:** USR_OMNI-SERVER_009 (was USR_OMNI-SERVER_003)  
**Priority:** HIGH  
**Subsystem:** All Subsystems

**User Need:**
Every measurement must be permanently linked to patient sample ID, operator, device, calibration data, and exact timestamp for regulatory compliance and quality assurance.

**What Users Expect:**

The software shall provide complete traceability from measurement results to all contributing factors:

- Every measurement is **permanently linked** to:
  - Patient sample ID
  - Operator name/ID
  - Device serial number
  - Calibration data used
  - Exact timestamp
- **No data can be lost or altered** after acquisition
- **Complete audit trail** for regulatory compliance and quality assurance
- Data remains **traceable for years** for follow-up analysis or audits

**Automatically Captured Metadata:**

- **Device identification:** Serial number, hardware configuration, firmware versions
- **Operator identification:** Username, credential level, authentication timestamp
- **Temporal data:** Measurement timestamp with timezone, duration, completion status
- **Calibration linkage:** Calibration reference ID, calibration timestamp, validity status
- **Measurement parameters:** Exposure settings, beam intensity, detector configuration
- **Quality metrics:** SNR, beam stability, distance verification results
- **Environmental data:** System state, active warnings, interlock status

**User Scenario:**
> *"As a quality manager, I need to trace any diagnostic result back to who performed it, when it was done, which device was used, and what calibration was active—even years later for FDA audits."*

**Traceability Information Captured:**
- **Measurement metadata:** run_id (UUID), sample_id, operator, timestamp
- **Device information:** serial number, hardware configuration, software version
- **Calibration linkage:** calibrant used, calibration timestamp, QC parameters
- **Quality metrics:** SNR, beam intensity, distance check results
- **Audit records:** All commands, state transitions, safety events

**Rationale:** 
Complete traceability enables audit compliance and quality management.

**Verification:** 
- Metadata completeness testing
- Traceability link validation
- Audit report generation testing

**Maps To:**
- SYS_OMNI-SERVER_016: Traceability metadata capture

---

## 4. Calibration Management

### 4.1 Daily Calibration Enforcement

**Requirement ID:** USR_OMNI-SERVER_002  
**Priority:** CRITICAL  
**Subsystem:** Hardware Server

**User Need:**
The operator must be able to verify device health via a daily calibration routine. The system must enforce calibration to ensure measurement accuracy.

**What Lab Technicians Need:**

The software must make daily calibration simple and enforce it:

- Remind me if I haven't calibrated today (or block me from running patient tests)
- Walk me through the calibration steps clearly
- Tell me pass or fail - not confusing numbers I have to interpret
- If calibration fails, tell me what to do: "Call service" or "Try again"
- Remember when the last good calibration was done

**Specific Requirements:**

The software shall enforce daily calibration requirements to maintain diagnostic accuracy:

- Block diagnostic measurements if calibration has not been performed within 24 hours
- Display prominent calibration status indicators on all operator interfaces
- Provide guided calibration workflow with step-by-step instructions
- Automatically validate calibration results against defined acceptance criteria
- Display clear pass/fail indication with corrective action guidance
- Maintain calibration history with timestamps and results
- Transition system to LOCKED state when calibration expires

**User Scenario:**
> *"I come in Monday morning after a weekend. The software shows a big orange banner: 'Calibration expired - last calibrated Friday at 8:15 AM. Please run daily calibration before patient measurements.'"*

**Calibration Workflow Users Follow:**
1. Load calibration standard into device
2. Initiate calibration via user interface
3. Wait for automated measurement and analysis
4. Receive pass/fail status
5. System unlocks for diagnostic use (if passed)

**What Happens Without Calibration:**
- System remains in **LOCKED state**
- Diagnostic measurements are **prohibited**
- Clear message: *"Daily calibration required. Last calibration: 26 hours ago."*

**Operational Scenario:** 
On Monday morning following a weekend, the system shall display: *"Calibration expired - Last performed: Friday 8:15 AM. Daily calibration required before diagnostic measurements."*

**Rationale:** 
Daily calibration ensures measurement accuracy and device reliability.

**Verification:** 
- Calibration enforcement testing
- 24-hour timeout validation
- Workflow usability testing
- VER_OMNI-SERVER_002: Integration test
- VER_OMNI-SERVER_004: Calibration procedure test

**Maps To:**
- SYS_OMNI-SERVER_004: Calibration enforcement

---

### 4.2 Calibration Diagnostics

**Requirement ID:** USR_OMNI-SERVER_010  
**Priority:** MEDIUM  
**Subsystem:** Hardware Server

**User Need:**
When calibration fails, maintenance engineers need detailed diagnostic information to troubleshoot the problem.

**What Maintenance Engineers Need:**

- Detailed calibration results (not just pass/fail)
- Graphs showing calibration trends over time
- Clear indication of which parameter failed
- Historical data so I can see if this is a new problem or getting worse
- The ability to run calibration multiple times to verify repairs

**Specific Requirements:**

The software shall provide maintenance engineers with comprehensive calibration diagnostic capabilities:

- Display detailed calibration parameters and measured values
- Provide graphical trending of calibration results over time
- Indicate specific parameters failing acceptance criteria
- Maintain historical calibration data for failure analysis
- Support multiple calibration attempts for verification
- Export calibration data for external analysis

**Rationale:** 
Detailed diagnostics enable efficient troubleshooting of calibration failures.

**Verification:** 
- Diagnostic data accuracy testing
- Trending graph validation
- Export functionality testing

**Maps To:**
- SYS_OMNI-SERVER_017: Calibration diagnostics and trending

---

## 5. User Interface and Usability

### 5.1 Operator Interface Requirements

**Requirement ID:** USR_OMNI-SERVER_011  
**Priority:** HIGH  
**Subsystem:** User Interface

**User Need:**
The software must be simple enough for busy laboratory workday with minimal training required.

**What Lab Technicians Need:**

I handle dozens of samples per day and don't have time for complicated software:

- Starting a measurement takes 3-4 clicks maximum
- Big buttons I can see from across the room
- Status shown in plain English: "Ready", "Running", "Calibration needed"
- Error messages that tell me what to DO, not just what went wrong
- Common tasks shouldn't require scrolling through menus

**What I Don't Want:**
- Technical jargon in error messages
- Needing to remember command sequences
- The software crashing and losing my place
- Pop-ups that interrupt measurements

**Specific Requirements:**

The software shall provide an intuitive interface optimized for clinical workflow efficiency:

- Measurement initiation achievable in maximum 4 user interactions
- Large touch-friendly buttons visible from 2 meters distance
- Status display using plain language: "Ready", "Running", "Calibration Required", "Fault"
- Real-time status updates with no perceptible lag (<200ms)
- Error messages providing actionable corrective steps
- Minimal menu navigation for common operations
- Responsive interface with no freezing during measurements

**Key UI Elements:**
- **Status panel:** Current system state, interlock status, calibration status
- **Sample entry:** Quick barcode scan or manual ID entry
- **Start/Stop controls:** Large, clearly labeled buttons
- **Safety indicators:** Visual confirmation all interlocks satisfied
- **History log:** Recent measurements and system events

**Rationale:** 
Efficient interface design reduces operator training time and operational errors.

**Verification:** 
- Usability testing with representative users
- Response time measurement (<200ms)
- Workflow efficiency analysis

**Maps To:**
- SYS_OMNI-SERVER_008: gRPC interface
- SYS_OMNI-SERVER_022: User-facing error messages

---

### 5.2 Clinician Interface Requirements

**Requirement ID:** USR_OMNI-SERVER_012  
**Priority:** MEDIUM  
**Subsystem:** User Interface

**User Need:**
Clinicians need clear presentation of results for diagnostic interpretation.

**What Clinicians Need:**

When reviewing diagnostic results, the software should:

- Show the XRD pattern as a clear, zoomable graph
- Highlight any quality issues visually
- Present measurement parameters in a table I can quickly scan
- Let me export data for my own analysis tools
- Make it obvious which results are from the same patient

**Specific Requirements:**

The software shall provide clinicians with comprehensive data visualization and analysis capabilities:

- Display XRD patterns as high-resolution zoomable graphs
- Visual highlighting of quality issues and data anomalies
- Tabular presentation of measurement parameters for rapid review
- Data export in standard formats (CSV, JSON, DICOM)
- Clear visual grouping of measurements by patient sample
- Comparison tools for multiple measurements

**Rationale:** 
Clear data presentation enables accurate diagnostic interpretation.

**Verification:** 
- Data visualization accuracy testing
- Export format validation
- Usability testing with clinicians

**Maps To:**
- SYS_OMNI-SERVER_025: Analytics and reporting

---

## 6. System Reliability

### 6.1 Availability and Recovery

**Requirement ID:** USR_OMNI-SERVER_013  
**Priority:** HIGH  
**Subsystem:** All Subsystems

**User Need:**
The software must be running when I come in each morning. I can't afford to lose time troubleshooting computers.

**What Lab Technicians Need:**

- Start automatically when the computer boots
- Reconnect to hardware devices if they were power-cycled
- Show me clearly if it's ready or still starting up
- Not crash multiple times per week
- Recover gracefully if Windows updates overnight

**Specific Requirements:**

The software shall provide high availability and automatic recovery capabilities:

- Automatic startup as system service on computer boot
- Automatic hardware device reconnection after power cycling
- Clear system status indication during startup sequence
- Target uptime of 99% during operational hours
- Graceful handling of Windows updates without data loss
- Maximum restart time of 3 minutes to operational state
- Automatic recovery from transient hardware failures

**Operational Scenario:** 
At 7:00 AM shift start, the software shall be fully operational with status indicating "Ready" or "Calibration Required" as appropriate.

**Rationale:** 
Clinical operations require reliable system availability.

**Verification:** 
- Availability monitoring
- Restart time measurement
- Failure recovery testing

**Maps To:**
- SYS_OMNI-SERVER_018: Availability and auto-recovery

---

### 6.2 Diagnostic and Troubleshooting Support

**Requirement ID:** USR_OMNI-SERVER_014  
**Priority:** MEDIUM  
**Subsystem:** Hardware Server

**User Need:**
IT administrators need to keep the system running without calling vendor support every time.

**What IT Administrators Need:**

- Clear log files I can read to diagnose problems
- Not require administrator rights for normal operation (lab techs run as standard users)
- Tell me specifically what's wrong: "Cannot connect to detector at 192.168.1.100"
- Have a documented way to restart services without rebooting
- Not conflict with our standard Windows updates and antivirus

**Specific Requirements:**

The software shall provide IT administrators with comprehensive diagnostic capabilities:

- Structured log files with configurable verbosity levels
- Standard user account operation (no administrator rights required for normal use)
- Specific error messages including component identification and network addresses
- Service restart capability without full system reboot
- Compatibility with standard Windows updates and antivirus software
- Remote monitoring capability for system health
- Automated log rotation with configurable retention

**Rationale:** 
Supportability reduces downtime and vendor dependency.

**Verification:** 
- Log analysis testing
- Restart procedure validation
- Update compatibility testing

**Maps To:**
- SYS_OMNI-SERVER_019: Diagnostic logging and troubleshooting

---

## 7. Multi-User and Access Control

### 7.1 Session Management

**Requirement ID:** USR_OMNI-SERVER_015  
**Priority:** HIGH  
**Subsystem:** Orchestrator

**User Need:**
Lab technicians need to know who's using the machine and when it will be available.

**What Lab Technicians Need:**

If another technician is running a test, I need to know:

- Who is currently logged in and using the machine
- How long until their measurement is done
- Whether I can queue up my sample or need to wait
- My session doesn't interfere with theirs

**Specific Requirements:**

The software shall provide clear session management for multi-user environments:

- Display current operator name and session status prominently
- Show estimated time remaining for active measurements
- Prevent session interference through exclusive device access control
- Automatic session termination on operator logout or disconnect
- Session activity logging for audit trails
- Concurrent read-only monitoring for supervisory personnel

**Rationale:** 
Clear session management prevents operational conflicts in shared environments.

**Verification:** 
- Multi-user testing
- Session timeout validation
- Concurrent access testing

**Maps To:**
- SYS_OMNI-SERVER_020: Multi-user session management

---

### 7.2 Role-Based Access Control

**Requirement ID:** USR_OMNI-SERVER_016  
**Priority:** HIGH  
**Subsystem:** All Subsystems

**User Need:**
IT administrators must control who can do what in the system.

**What IT Administrators Need:**

Different users need different permissions:

- Lab technicians can run measurements and calibrations
- Senior staff can review audit logs
- Only maintenance engineers can enter maintenance mode
- I can add/remove users without editing config files
- User permissions are enforced by the software, not honor system

**Role Definitions:**

- **Clinical Operator:** Measurement execution, calibration, basic troubleshooting
- **Senior Staff:** Audit log access, historical data review, quality analysis
- **Maintenance Engineer:** Full diagnostic access, maintenance mode, configuration updates
- **Administrator:** User management, system configuration, security controls

**Specific Requirements:**

The software shall implement role-based access control (RBAC):

- Graphical user management interface (no configuration file editing)
- Permission enforcement at software level with cryptographic authentication
- Failed access attempt logging
- Password complexity requirements and expiration policies
- Multi-factor authentication for privileged roles

**Rationale:** 
Access control ensures security and regulatory compliance.

**Verification:** 
- Access control testing
- Privilege escalation testing
- Authentication validation

**Risk Controls:**
- RISK_OMNI-SERVER_009: Role-based access enforcement

**Maps To:**
- SYS_OMNI-SERVER_009: Session security
- SYS_OMNI-SERVER_021: Role-based access control

---

## 8. Error Handling and Recovery

### 8.1 Operator-Facing Error Messages

**Requirement ID:** USR_OMNI-SERVER_017  
**Priority:** HIGH  
**Subsystem:** All Subsystems

**User Need:**
When an error happens, the software must help me fix problems, not just report them.

**What Lab Technicians Need:**

When an error happens, I don't want cryptic messages. I expect:

**Good Error Message:**
```
X-ray Detector Not Responding (Error: DET-001)

Please check:
1. Detector power cable is firmly connected
2. Network cable is connected to detector
3. Detector power indicator shows green light

If all checks pass and problem persists:
Contact Service: 1-800-555-OMNI
Reference: DET-001, Session: abc-123-xyz
```

**Bad Error Message:**
```
Error code 0x8007274C - BIS socket timeout exception
```

**Message Components Required:**

The software should:
- Explain what happened in plain language
- Tell me what I can try to fix it
- Tell me when to call for help vs. when I can fix it myself
- Not make me remember error codes

**Specific Requirements:**

- Plain language description of the problem
- Numbered checklist of user-actionable troubleshooting steps
- Clear indication of when to escalate to maintenance/IT support
- Contact information for technical support
- Unique error reference number for support coordination

**Rationale:** 
Clear error messages reduce operator stress and improve problem resolution time.

**Verification:** 
- Error message comprehension testing
- Troubleshooting success rate measurement

**Maps To:**
- SYS_OMNI-SERVER_022: User-friendly error messaging

---

### 8.2 Technical Diagnostic Information

**Requirement ID:** USR_OMNI-SERVER_018  
**Priority:** MEDIUM  
**Subsystem:** All Subsystems

**User Need:**
Maintenance engineers need technical details when troubleshooting.

**What Maintenance Engineers Need:**

While lab techs need simple messages, I need the technical details:

- Error codes and stack traces in a separate log file
- Exact network addresses and port numbers
- Timing information (how long did it wait before timeout?)
- What the software tried to do before failing
- A way to export logs to send to vendor support

**Specific Requirements:**

The software shall provide maintenance engineers and IT staff with comprehensive technical diagnostics:

- Detailed error codes and stack traces in structured log files
- Network connectivity details (IP addresses, ports, protocols)
- Timing information (timeouts, retry attempts, durations)
- Hardware communication logs with protocol-level details
- System state snapshots at time of error
- Log export capability for vendor support escalation

**Rationale:** 
Technical diagnostics enable efficient problem resolution by qualified personnel.

**Verification:** 
- Log completeness testing
- Diagnostic information accuracy validation

**Maps To:**
- SYS_OMNI-SERVER_019: Diagnostic logging
- SYS_OMNI-SERVER_022: Error handling framework

---

## 9. Software Update Management

### 9.1 Update Deployment Control

**Requirement ID:** USR_OMNI-SERVER_019  
**Priority:** HIGH  
**Subsystem:** All Subsystems

**User Need:**
IT administrators need software updates that don't break everything.

**What IT Administrators Need:**

When a software update is released:

- Clear release notes explaining what changed
- A way to test the update before deploying to production
- The ability to roll back if something goes wrong
- Updates that don't require recalibration unless necessary
- Notification of updates well in advance, not surprise automatic updates

**Specific Requirements:**

The software shall provide controlled update deployment minimizing operational disruption:

- Comprehensive release notes detailing changes, fixes, and impacts
- Staged deployment capability (test → production)
- Rollback capability to previous version if issues arise
- Update installation scheduled during non-operational hours
- Advance notification of available updates (minimum 48 hours)
- Preservation of calibration data and configuration during updates
- User confirmation required before update installation
- Automatic backup before update application

**Rationale:** 
Controlled updates prevent operational disruptions and data loss.

**Verification:** 
- Update/rollback testing
- Data preservation validation
- Notification timing verification

**Risk Controls:**
- RISK_OMNI-SERVER_011: Update validation and rollback capability

**Maps To:**
- SYS_OMNI-SERVER_010: Safety profile updates
- SYS_OMNI-SERVER_023: Update management

---

### 9.2 Update Impact Transparency

**Requirement ID:** USR_OMNI-SERVER_020  
**Priority:** MEDIUM  
**Subsystem:** All Subsystems

**User Need:**
Lab technicians need to know how software updates will affect their work.

**What Lab Technicians Need:**

Software updates must not interrupt my work:

- Updates should install after-hours or when I choose
- If an update requires downtime, tell me how long
- After update, my workflow should be the same (don't surprise me with UI changes)
- Don't lose my calibration when you update

**Specific Requirements:**

The software shall provide clear communication of update impacts:

- Estimated downtime duration
- Indication of workflow changes or UI modifications
- Calibration requirements post-update
- New features or capabilities introduced
- Deprecated features or compatibility changes
- Security patches and vulnerability remediation details

**Rationale:** 
Transparency enables users to plan for update impacts.

**Verification:** 
- Release note completeness review
- Impact communication testing

**Maps To:**
- SYS_OMNI-SERVER_023: Update management and communication

---

## 10. Regulatory Compliance and Audit Support

### 10.1 Comprehensive Audit Trail

**Requirement ID:** USR_OMNI-SERVER_021  
**Priority:** CRITICAL  
**Subsystem:** All Subsystems

**User Need:**
Administrators need the software to help pass audits, not make them harder.

**What Administrators Need:**

When FDA or hospital auditors come, the software must:

- Provide a complete audit trail of all operations
- Export logs in a readable format (CSV, PDF)
- Show me who did what, when, and why
- Prove that safety interlocks are always enforced
- Demonstrate data cannot be altered after collection

**Questions an Auditor Might Ask (Software Must Answer):**

- "Show me all measurements performed by technician Sarah in March 2025"
- "Prove that calibration was valid for this patient measurement"
- "What happened on June 15th when there was a safety alarm?"
- "Has anyone disabled safety interlocks in the past year?"
- "Show me all failed login attempts"

**Specific Requirements:**

The software shall maintain a complete, tamper-evident audit trail supporting FDA and hospital audits:

**Logged Events:**
- All measurement operations with complete parameters
- User authentication and authorization events
- Configuration changes with before/after values
- Safety system status changes and interlock events
- Data access operations with user identification
- Calibration results and validation outcomes
- Software updates and version changes
- Maintenance mode sessions and activities

**Audit Trail Requirements:**
- Append-only storage preventing retroactive modification
- Cryptographic integrity protection (digital signatures)
- Exportable in standard formats (CSV, PDF, JSON)
- Searchable and filterable by time, user, event type
- Retention for minimum 7 years (configurable)
- Daily automated integrity verification

**Rationale:** 
Comprehensive audit trails are required for FDA compliance and quality management.

**Verification:** 
- Audit trail completeness testing
- Tamper detection testing
- Export validation
- Regulatory compliance review

**Risk Controls:**
- RISK_OMNI-SERVER_010: Audit trail integrity and tamper protection

**Maps To:**
- SYS_OMNI-SERVER_007: Audit database
- SYS_OMNI-SERVER_024: Comprehensive audit logging

---

### 10.2 Quality Improvement Analytics

**Requirement ID:** USR_OMNI-SERVER_022  
**Priority:** MEDIUM  
**Subsystem:** Orchestrator, User Interface

**User Need:**
Clinicians and quality managers need to review system performance over time for quality improvement.

**What Clinicians Need:**

- Success/failure rates for measurements
- Trends in calibration results (is the machine drifting?)
- Which samples had to be re-run and why
- Statistical quality control charts
- This information helps me know if we need service or training

**Specific Requirements:**

The software shall provide clinicians and quality managers with analytical capabilities:

- Measurement success/failure rate reporting
- Calibration trend analysis with statistical process control
- Sample re-run analysis with root cause categorization
- Quality control charts (X-bar, R charts) for key parameters
- Equipment utilization and performance metrics
- Operator performance comparison (anonymous)
- Predictive maintenance indicators

**Rationale:** 
Analytics support continuous quality improvement and proactive maintenance.

**Verification:** 
- Analytics accuracy testing
- Report generation validation
- Statistical calculation verification

**Maps To:**
- SYS_OMNI-SERVER_025: Analytics and quality reporting

---

## 11. Network and Cloud Integration

### 11.1 Offline Operation Capability

**Requirement ID:** USR_OMNI-SERVER_023  
**Priority:** CRITICAL  
**Subsystem:** Orchestrator

**User Need:**
The software must keep working if the internet goes down. Patient care can't wait.

**What Lab Technicians Need:**

Our internet isn't perfect:

- Let me run measurements even if cloud is unreachable
- Store all data locally first, upload to cloud when available
- Show me clearly if cloud connection is down
- Not slow down because it's trying to reach the cloud
- Warn me if local storage is getting full

**Specific Requirements:**

The software shall maintain full diagnostic capability during network outages:

- Complete measurement operations without cloud connectivity
- Local data storage with automatic synchronization upon reconnection
- Clear visual indication of network connectivity status
- No performance degradation due to network unavailability
- Local storage capacity monitoring with alerts at 80% utilization
- Data queue management with prioritization
- Automatic retry with exponential backoff for failed uploads

**Operational Scenario:** 
At 2:00 PM, hospital network experiences outage. Clinical operators continue scheduled measurements with local storage. At 4:00 PM, network restores and all data automatically uploads to cloud with verification.

**Rationale:** 
Clinical operations cannot be dependent on network availability.

**Verification:** 
- Offline operation testing
- Data synchronization validation
- Performance testing during network loss

**Maps To:**
- SYS_OMNI-SERVER_026: Offline capability and synchronization

---

### 11.2 Network Management

**Requirement ID:** USR_OMNI-SERVER_024  
**Priority:** MEDIUM  
**Subsystem:** Orchestrator

**User Need:**
IT administrators need visibility into cloud connectivity and upload status.

**What IT Administrators Need:**

- Current connection status to cloud services
- How much data is queued for upload
- Whether any uploads failed and need retry
- Network bandwidth usage (not saturating our connection)
- The ability to pause cloud sync during backups

**Specific Requirements:**

The software shall provide IT administrators with network management capabilities:

- Real-time cloud connectivity status dashboard
- Upload queue monitoring with data volume indicators
- Failed upload notification and manual retry capability
- Bandwidth throttling controls to prevent network saturation
- Cloud synchronization pause/resume controls
- Network connectivity diagnostics and logging
- Secure proxy configuration support

**Rationale:** 
Network management capabilities enable IT oversight and troubleshooting.

**Verification:** 
- Network monitoring accuracy testing
- Bandwidth control validation
- Proxy configuration testing

**Maps To:**
- SYS_OMNI-SERVER_026: Network management and monitoring

---

## 12. Performance Requirements

### 12.1 Responsiveness

**Requirement ID:** USR_OMNI-SERVER_025  
**Priority:** HIGH  
**Subsystem:** All Subsystems

**User Need:**
The software must keep up with my pace when processing multiple samples.

**What Lab Technicians Need:**

When I'm processing multiple samples, timing matters:

- Clicking "Start Measurement" should respond in under 2 seconds
- Status updates should be real-time, not delayed
- The interface shouldn't freeze during measurements
- Searching for old results should be fast (under 5 seconds)
- The software shouldn't slow down after running for hours

**Specific Requirements:**

The software shall meet defined performance benchmarks:

| Operation | Maximum Response Time |
|-----------|----------------------|
| Measurement initiation | 2 seconds |
| Emergency stop response | 100 milliseconds |
| State transition display | 1 second |
| Data commit to storage | 5 seconds post-measurement |
| Calibration routine completion | 10 minutes |
| Historical data search | 5 seconds |
| UI interaction response | 200 milliseconds |

**Rationale:** 
Responsive performance supports efficient clinical workflow.

**Verification:** 
- Performance benchmarking under normal and stress conditions
- VER_OMNI-SERVER_014: Performance testing

**Maps To:**
- SYS_OMNI-SERVER_027: Performance specifications

---

### 12.2 Capacity and Scalability

**Requirement ID:** USR_OMNI-SERVER_026  
**Priority:** MEDIUM  
**Subsystem:** All Subsystems

**User Need:**
The system should handle typical clinical workload without degradation.

**Expected Workload:**

- 50-100 measurements per day per machine
- 5-10 years of historical data stored locally
- Multiple users accessing the system throughout the day
- Continuous operation during business hours (8-12 hours/day)
- Software should run for weeks without restart

**Specific Requirements:**

The software shall support the following operational capacity:

- 100 measurements per day per device
- 10 years of historical data storage locally
- 10 concurrent users (with single active operator)
- Continuous 12-hour daily operation
- 30-day uptime without restart requirement
- 1TB local data storage minimum
- Sub-second performance degradation over time

**Rationale:** 
Capacity planning ensures long-term system viability.

**Verification:** 
- Load testing
- Long-duration operation testing
- Storage capacity testing

---

## 13. Training and Documentation

### 13.1 User Training Requirements

**Requirement ID:** USR_OMNI-SERVER_027  
**Priority:** MEDIUM  
**Subsystem:** All Subsystems

**User Need:**
The software must be learnable without weeks of training.

**What Lab Technicians Need:**

A new technician should be able to:

- Learn basic operations in 2-4 hours
- Access help within the software (not a separate manual)
- Get visual guides for common tasks
- Practice on test samples before doing patient work
- Have a quick-reference card for rare operations

**Training Time Targets:**

- Clinical operators: 4 hours to basic proficiency
- Administrators: 8 hours including configuration
- Maintenance engineers: 16 hours including diagnostics

**Specific Requirements:**

The software shall support efficient user training:

- In-application help system with context-sensitive content
- Visual workflow guides for common operations
- Interactive training mode with simulated samples
- Quick reference cards exportable as PDF
- Video tutorials for complex procedures

**Rationale:** 
Efficient training reduces operational errors and deployment time.

**Verification:** 
- Training time measurement with representative users
- Help system usability testing

---

### 13.2 Documentation Requirements

**Requirement ID:** USR_OMNI-SERVER_028  
**Priority:** MEDIUM  
**Subsystem:** All Subsystems

**User Need:**
All users need documentation that actually helps.

**What All Users Need:**

- Step-by-step guides with screenshots
- Troubleshooting flowcharts
- Video tutorials for complex procedures
- Searchable help system
- Emergency procedures clearly posted

**Specific Requirements:**

The software shall be supported by comprehensive documentation:

- User manual with step-by-step procedures and screenshots
- Administrator guide covering installation, configuration, and maintenance
- Troubleshooting guide with flowcharts and decision trees
- API documentation for integration developers
- Release notes for all versions
- Searchable online help system
- Emergency procedure posters (printable)

**Rationale:** 
Complete documentation supports independent problem resolution.

**Verification:** 
- Documentation completeness review
- Accuracy verification
- Usability testing

---

## 14. Hardware Initialization and Power Management

### 14.1 Device Initialization Workflow

**Requirement ID:** USR_OMNI-HW_001  
**Priority:** CRITICAL  
**Subsystem:** Hardware Server

**User Need:**
The system must safely power on and initialize hardware devices with proper safety checks.

**What Users Expect:**

**System Startup Sequence:**

1. **Computer Power-On**
   - GPIO Hardware Auto-Starts
   - Server Application Starts
   - All interlocks immediately readable

2. **Initial State**
   - Key Switch: OFF
   - GPIO: Powered and reading interlocks
   - Detector: Not initialized (no power)
   - Motion: Not initialized (no power)
   - Safety State: LOCKED

3. **Key Switch ON → Initialization Allowed**
   - User turns physical key switch
   - GPIO detects key switch ON
   - Safety State: LOCKED → IDLE
   - Orchestrator can now request device initialization

**Initialization Prerequisites:**
- ✅ Key Switch: ON
- ✅ All Interlocks: SAFE
- ✅ Activation Button: ACTIVE (20-second window)

**Specific Requirements:**

The software shall implement controlled device initialization:

- Require key switch ON before any initialization
- Require activation button press within 20 seconds before initialization
- Verify all safety interlocks before powering devices
- Power detector and motion systems separately
- Provide clear status indication during initialization
- Log all initialization attempts and outcomes

**User Scenario:**
> *"When I arrive in the morning, I turn the key switch to ON, press the activation button, and click 'Initialize Detector' in the software. The system powers up the detector and shows me when it's ready."*

**Verification:**
- Initialization sequence testing
- Safety prerequisite validation
- Activation button timeout testing

**Maps To:**
- SYS_OMNI-SERVER_011: Device initialization control

---

### 14.2 Activation Button Logic

**Requirement ID:** USR_OMNI-HW_002  
**Priority:** CRITICAL  
**Subsystem:** Hardware Server

**User Need:**
Physical confirmation should be required for potentially harmful operations.

**Purpose:**
Physical confirmation required for **potentially harmful** operations that arrive from the orchestrator.

**Harmful Operations (Require Enable Button):**
- **Initialize Detector:** Powers ON X-ray detector
- **Initialize Motion:** Activates motion system (can cause physical movement)
- **Start Exposure:** Activates X-rays (radiation hazard)
- **Move Motion:** Causes physical movement (collision hazard)

**Safe Operations (No Enable Button Required):**
- **Read States:** Query GPIO/Detector/Motion state (read-only)
- **Stop Operations:** Stop exposure, stop motion (safety operation)
- **Power Off:** Controlled shutdown (safety operation)
- **Get Interlocks:** Read safety status (read-only)

**Specific Requirements:**

The software shall implement activation button control:

- **Click:** Activates for 20 seconds
- **Countdown:** Live timer displayed in GUI
- **Auto-Expire:** Automatically deactivates after 20 seconds
- **Scope:** Each harmful operation must occur within the 20-second window
- Clear visual indication of activation status and remaining time
- Rejection of harmful operations when button not active

**User Scenario:**
> *"Before starting a measurement, I press the activation button. The screen shows '18 seconds remaining'. I have time to click 'Start Measurement' before the timer expires."*

**Verification:**
- Activation button timeout testing
- Operation rejection when inactive
- Visual countdown accuracy

**Maps To:**
- SYS_OMNI-SERVER_002: Interlocks and activation button
- SYS_OMNI-SERVER_011: Initialization prerequisites

---

## 15. Motion Control

### 15.1 Motion Operations

**Requirement ID:** USR_OMNI-MOTION_001  
**Priority:** HIGH  
**Subsystem:** Hardware Server

**User Need:**
Operators and maintenance engineers need safe, precise control of sample positioning.

**What Users Expect:**

The software shall provide safe and precise motion control:

- **MoveTo:** Move to absolute position with feedback
- **MoveRelative:** Move by specified distance
- **Home:** Find home position with multi-phase homing
- **Stop:** Immediately halt motion
- **SetVelocity:** Adjust motion speed
- **GetPosition:** Query current position and homed status

**Specific Requirements:**

**Motion Operations:**
- Non-blocking execution with 10 Hz position updates
- Real-time position feedback during movement
- Limit detection and protection
- Homing with multiple phases (init, search, backoff, latch, done)
- Emergency stop capability
- Position accuracy verification

**User Scenario:**
> *"I need to move the sample to 50mm position for calibration. I click 'Move To 50mm' and watch the position update every 0.1 seconds until it reaches the target."*

**Verification:**
- Motion accuracy testing
- Position feedback validation
- Emergency stop response time
- Homing sequence verification

**Maps To:**
- Motion service commands in COMMANDS_SPEC.md
- SYS_OMNI-SERVER_020: Motion control system

---

## 16. Complete Requirements Traceability

### 16.1 Requirements Coverage Summary

**User Requirements (USR_OMNI-SERVER_xxx):**
- Total: 28 user requirements
- Critical Priority: 10 requirements
- High Priority: 10 requirements
- Medium Priority: 8 requirements

**System Requirements Coverage:**
- Hardware Server: 27 system requirements
- Orchestrator: To be detailed in separate document
- User Interface: To be detailed in separate document

**Verification Requirements:**
- 14 verification requirements defined
- Coverage across all critical user requirements

**Risk Control Requirements:**
- 11 risk control requirements
- Mapped to safety-critical user requirements

---

### 16.2 Traceability Matrix

| User Requirement | System Requirements | Risk Controls | Verification |
|-----------------|---------------------|---------------|--------------|
| USR_OMNI-SERVER_001 (Safety) | SYS_001, 002, 003 | RISK_001, 002, 003 | VER_001, 003 |
| USR_OMNI-SERVER_002 (Calibration) | SYS_004 | - | VER_002, 004 |
| USR_OMNI-SERVER_003 (Traceability) | SYS_006, 016 | - | VER_004, 011 |
| USR_OMNI-SERVER_004 (Maintenance) | SYS_011 | RISK_006 | VER_006 |
| USR_OMNI-SERVER_005 (Exposure) | SYS_005, 012 | RISK_007 | VER_005, 007 |
| USR_OMNI-SERVER_006 (Quality) | SYS_013 | RISK_008 | VER_008 |
| USR_OMNI-SERVER_007 (Data Integrity) | SYS_006, 014 | RISK_005 | VER_004, 009 |
| USR_OMNI-SERVER_008 (Security) | SYS_007, 009, 015 | RISK_004, 009 | VER_005, 010 |
| USR_OMNI-SERVER_009 (Traceability) | SYS_016 | - | VER_011 |
| USR_OMNI-SERVER_010 (Cal Diag) | SYS_017 | - | - |
| USR_OMNI-SERVER_011 (UI Operator) | SYS_008, 022 | - | VER_002 |
| USR_OMNI-SERVER_012 (UI Clinician) | SYS_008, 025 | - | - |
| USR_OMNI-SERVER_013 (Reliability) | SYS_018 | - | - |
| USR_OMNI-SERVER_014 (Diagnostics) | SYS_019 | - | - |
| USR_OMNI-SERVER_015 (Session Mgmt) | SYS_020 | - | - |
| USR_OMNI-SERVER_016 (RBAC) | SYS_009, 021 | RISK_009 | VER_005 |
| USR_OMNI-SERVER_017 (Error UI) | SYS_022 | - | - |
| USR_OMNI-SERVER_018 (Error Tech) | SYS_019, 022 | - | - |
| USR_OMNI-SERVER_019 (Updates) | SYS_010, 023 | RISK_011 | - |
| USR_OMNI-SERVER_020 (Update Trans) | SYS_023 | - | - |
| USR_OMNI-SERVER_021 (Audit) | SYS_007, 024 | RISK_010 | VER_004, 005, 012 |
| USR_OMNI-SERVER_022 (Analytics) | SYS_025 | - | - |
| USR_OMNI-SERVER_023 (Offline) | SYS_026 | - | VER_013 |
| USR_OMNI-SERVER_024 (Network) | SYS_026 | - | - |
| USR_OMNI-SERVER_025 (Performance) | SYS_027 | - | VER_014 |
| USR_OMNI-SERVER_026 (Capacity) | - | - | - |
| USR_OMNI-SERVER_027 (Training) | - | - | - |
| USR_OMNI-SERVER_028 (Documentation) | - | - | - |

---

## 17. Contraindications - What Software Must NOT Do

### 17.1 Forbidden Behaviors

The software shall NOT exhibit the following behaviors:

**Safety Violations:**
- ❌ Never allow X-ray activation with safety interlocks unsatisfied
- ❌ Never bypass safety systems without explicit maintenance mode authorization
- ❌ Never fail silently on safety system faults

**Data Integrity Violations:**
- ❌ Never allow data modification post-commit
- ❌ Never store patient data unencrypted
- ❌ Never lose data without explicit notification

**Usability Violations:**
- ❌ Never display technical error codes to clinical operators without plain language explanation
- ❌ Never require manual data saving for measurement results
- ❌ Never implement ambiguous UI elements that can be misinterpreted

**System Behavior Violations:**
- ❌ Never perform automatic updates during operational hours
- ❌ Never require system restart to recover from common errors
- ❌ Never hard-code configuration parameters that vary by installation

**Rationale:** 
Explicit prohibition of dangerous or frustrating behaviors improves safety and usability.

---

## 18. Summary of Core User Promises

**If we had to summarize what users expect in one paragraph:**

The Omniscan software must guarantee safety above all else—never allowing X-ray exposure when unsafe, and controlling dose precisely. It must reliably save every measurement with complete traceability, even if computers crash or networks fail. Daily calibration must be enforced but simple. The interface should be clear enough that a technician can run tests confidently after a few hours of training. When errors happen, the software should help fix them, not just report them. Updates must not disrupt patient care. And through it all, the software must provide a complete audit trail proving that safety, accuracy, and data integrity were maintained at every step.

**In other words:** Be safe, be reliable, be clear, and help us do our jobs well.

---

## Appendix A: User Personas

### Persona 1: Laboratory Technician (Primary User)
- **Name:** Sarah, Clinical Lab Technician
- **Experience:** 5 years in medical laboratory
- **Goals:** Process patient samples accurately and efficiently, maintain compliance
- **Concerns:** Patient safety, data accuracy, meeting turnaround times
- **Technical skill:** Moderate; comfortable with medical lab equipment

### Persona 2: Laboratory Manager (Secondary User)
- **Name:** Dr. James, Laboratory Director
- **Experience:** 15 years, responsible for quality and compliance
- **Goals:** Ensure regulatory compliance, maintain quality standards, manage staff
- **Concerns:** FDA audits, device validation, data integrity
- **Technical skill:** High; understands medical device regulations

### Persona 3: Service Engineer (Maintenance User)
- **Name:** Mike, Biomedical Equipment Technician
- **Experience:** 10 years servicing medical devices
- **Goals:** Maintain device uptime, troubleshoot issues efficiently
- **Concerns:** Device reliability, clear diagnostic information, parts availability
- **Technical skill:** Very high; hardware and software troubleshooting

### Persona 4: IT Administrator
- **Name:** Lisa, Hospital IT Specialist
- **Experience:** 8 years managing clinical systems
- **Goals:** Keep systems secure, reliable, and compliant
- **Concerns:** Network security, data backups, update management
- **Technical skill:** Very high; Windows administration and networking

### Persona 5: Quality Manager
- **Name:** Dr. Chen, Quality Assurance Director
- **Experience:** 12 years in medical device quality
- **Goals:** Ensure FDA compliance, prepare for audits
- **Concerns:** Audit trails, traceability, quality metrics
- **Technical skill:** High; regulatory and quality systems expertise

---

## Appendix B: Regulatory References

- **FDA 21 CFR Part 11** - Electronic Records and Signatures
- **IEC 62304** - Medical Device Software Lifecycle Processes
- **ISO 14971** - Application of Risk Management to Medical Devices
- **IEC 62366-1** - Application of Usability Engineering to Medical Devices
- **HIPAA Security Rule** - 45 CFR Part 164
- **FDA Cybersecurity Guidance (2023)**
- **ISO 13485** - Quality Management Systems for Medical Devices
- **ISO 27001** - Information Security Management

---

## Document Control

**Revision History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-24 | Omniscan Development Team | Initial release |
| 2.0 | 2025-11-17 | Omniscan Development Team | Consolidated all subsystem requirements |

**Review and Approval:**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Clinical Representative | | | |
| Quality Assurance | | | |
| Regulatory Affairs | | | |
| Software Development Lead | | | |

**Next Review:** Prior to FDA Design History File submission

**Document Classification:** FDA DHF - Requirements Phase

---

**End of Document**
