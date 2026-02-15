# Omniscan User Requirements Document

**Document Version:** 1.0  
**Date:** 2025-10-24  
**Project:** Omniscan Medical X-ray Diffraction Diagnostic Platform  
**Compliance:** FDA/IEC 62304 Class B, ISO 14971, IEC 62366-1

---

## 1. Executive Summary

Omniscan is a first-of-kind medical diagnostic device using X-ray Diffraction (XRD) technology for patient sample analysis in clinical laboratories. This document describes what users—clinicians, laboratory operators, maintenance engineers, and administrators—expect from the Omniscan software system.

**Target Users:**
- **Clinical Operators**: Laboratory technicians performing diagnostic measurements
- **Clinicians**: Medical professionals reviewing diagnostic results
- **Maintenance Engineers**: Service personnel performing calibration and maintenance
- **Administrators**: IT staff managing system configuration and data

---

## 2. Clinical User Expectations

### 2.1 Safe and Reliable Operation (USR_OMNI-SERVER_001)

**What Users Expect:**
- The system will **never expose operators to X-ray radiation** under unsafe conditions
- **Physical safety interlocks** prevent operation when doors are open or emergency stop is pressed
- The device will **automatically halt** all operations if any safety condition is violated
- **Clear visual and audible indicators** show when the system is safe to approach
- Operations can be **immediately stopped** at any time using the emergency stop button

**User Scenario:**
> *"As a laboratory technician, I need to know the X-ray machine will never turn on when the door is open, so I can safely load samples without risk of radiation exposure."*

**Safety Features Users Rely On:**
- Key switch must be in "operate" position
- Physical enable button confirmation before each measurement
- Door interlock sensors
- Emergency stop button (accessible at all times)
- Beam watchdog monitoring X-ray intensity
- Automatic transition to SAFE state on any fault

---

### 2.2 Daily Calibration and Quality Assurance (USR_OMNI-SERVER_002)

**What Users Expect:**
- Ability to perform **daily calibration routine** to verify device accuracy
- System **blocks diagnostic measurements** if daily calibration has not been performed (24-hour rule)
- **Clear status indicators** showing when calibration is required
- **Calibration results** are automatically recorded and traceable
- **Simple workflow** that can be completed in under 10 minutes

**User Scenario:**
> *"As a laboratory operator, I want to run a quick calibration check each morning before processing patient samples, so I know the results will be accurate and the device is functioning properly."*

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

---

### 2.3 Data Integrity and Traceability (USR_OMNI-SERVER_003)

**What Users Expect:**
- Every measurement is **permanently linked** to:
  - Patient sample ID
  - Operator name/ID
  - Device serial number
  - Calibration data used
  - Exact timestamp
- **No data can be lost or altered** after acquisition
- **Complete audit trail** for regulatory compliance and quality assurance
- Data remains **traceable for years** for follow-up analysis or audits

**User Scenario:**
> *"As a quality manager, I need to trace any diagnostic result back to who performed it, when it was done, which device was used, and what calibration was active—even years later for FDA audits."*

**Traceability Information Captured:**
- **Measurement metadata**: run_id (UUID), sample_id, operator, timestamp
- **Device information**: serial number, hardware configuration, software version
- **Calibration linkage**: calibrant used, calibration timestamp, QC parameters
- **Quality metrics**: SNR, beam intensity, distance check results
- **Audit records**: All commands, state transitions, safety events

---

## 3. Operational User Workflows

### 3.1 Standard Diagnostic Measurement Workflow

**User Steps:**
1. **Morning startup**: Turn key switch to "operate", verify daily calibration is current
2. **Sample preparation**: Load patient sample into sample holder
3. **Start measurement**: 
   - Enter sample ID and operator information in UI
   - Initiate measurement command
4. **Physical confirmation**: Press physical enable button when prompted
5. **Wait for completion**: System performs automated measurement (typical: 30-120 seconds)
6. **Collect results**: Raw XRD data and metadata are automatically saved
7. **Remove sample**: System returns to IDLE state, safe to open door

**What Users See During Measurement:**
- State: **PENDING_ARMED** → *"Press enable button to start"*
- State: **RUNNING** → *"Measurement in progress: 45s elapsed"*
- State: **STOPPING** → *"Completing measurement..."*
- State: **IDLE** → *"Measurement complete. Safe to open door."*

---

### 3.2 Emergency Situations

**User Expects:**
- **Immediate response** to emergency stop button (beam off within milliseconds)
- **Clear indication** that system is in safe state
- **Guidance on recovery** steps to return to normal operation
- **No data loss** if emergency stop occurs during measurement

**Emergency Scenarios:**

| Situation | System Response | User Action Required |
|-----------|----------------|---------------------|
| Emergency stop pressed | Beam OFF, motion HALTED, enter SAFE state | Call supervisor, reset interlocks after investigation |
| Door opened during measurement | Beam OFF immediately, abort measurement | Close door, review safety procedures, restart |
| Key switch turned off | All operations disabled | Turn key back to "operate" if authorized |
| Beam intensity drops | Abort measurement, enter SAFE state | Contact service engineer for beam source inspection |
| System software crash | Watchdog disables beam, operations halted | Restart system, perform calibration before resuming |

---

### 3.3 Maintenance and Engineering Access

**Maintenance Engineer Expectations:**
- **Secure access mode** for configuration updates and hardware maintenance
- Ability to **bypass interlocks temporarily** for testing (with proper authorization)
- **Complete diagnostic information** for troubleshooting
- **Configuration management** with cryptographic signing to prevent unauthorized changes
- **Maintenance logs** automatically recorded

**Maintenance Workflow:**
1. Turn key switch to "maintenance" position
2. Authenticate with maintenance password and certificate
3. System enters **MAINTENANCE state**
4. Perform hardware service, configuration updates, or diagnostics
5. Run verification tests
6. Exit maintenance mode
7. System requires fresh calibration before clinical use

---

## 4. User Interface Expectations

### 4.1 Clinical Operator Interface (Web Browser)

**What Users Expect:**
- **Simple, intuitive workflow** requiring minimal training
- **Large, clear status indicators** visible from across the room
- **Minimal clicks** to perform common operations
- **Error messages in plain language** with corrective actions
- **Real-time feedback** during measurements

**Key UI Elements:**
- **Status panel**: Current system state, interlock status, calibration status
- **Sample entry**: Quick barcode scan or manual ID entry
- **Start/Stop controls**: Large, clearly labeled buttons
- **Safety indicators**: Visual confirmation all interlocks satisfied
- **History log**: Recent measurements and system events

### 4.2 Administrator Interface

**What Administrators Expect:**
- **User management**: Add/remove operators, assign permissions
- **Audit log access**: Review all system activities for compliance
- **System health monitoring**: Device status, error rates, calibration history
- **Data export**: CSV/JSON export for external analysis
- **Configuration review**: View (but not modify) device settings

---

## 5. System Integration Expectations

### 5.1 Data Flow User Expectations

**Users Expect:**
- **Local data storage** with immediate availability (no waiting for cloud)
- **Automatic cloud upload** for backup and advanced analysis
- **Offline operation capability** if network is down (with warnings)
- **No data loss** regardless of network status
- **Protected health information (PHI)** handling per HIPAA

**Data Journey Users Understand:**
1. **Acquisition**: Raw XRD data captured by hardware server
2. **Local storage**: Immediately saved to local encrypted database
3. **Processing**: Cloud processes data and runs ML models
4. **Results**: Diagnostic report returned to clinician
5. **Archive**: Long-term secure storage with traceability

### 5.2 Multi-User Environment

**What Multiple Users Expect:**
- **Clear session ownership**: System shows who is currently in control
- **Session security**: Only authorized operator can control device during their session
- **Automatic logout**: Session ends when operator disconnects
- **Concurrent monitoring**: Administrators can view status without interrupting operations

---

## 6. Performance and Reliability Expectations

### 6.1 Response Time

**User Expectations:**
| Action | Expected Response Time |
|--------|----------------------|
| Start measurement command | < 2 seconds to reach PENDING_ARMED |
| Emergency stop | < 100 milliseconds for beam disable |
| State transition | < 1 second with UI update |
| Data save | < 5 seconds after measurement completion |
| Calibration routine | < 10 minutes total |

### 6.2 Reliability

**What Users Expect:**
- **Uptime**: 99%+ availability during clinical hours
- **No data loss**: Zero tolerance for lost measurements or audit records
- **Graceful degradation**: System fails to safe state, never unsafe state
- **Predictable behavior**: Same inputs produce same results
- **Error recovery**: Clear guidance when things go wrong

---

## 7. Compliance and Quality Expectations

### 7.1 Regulatory Compliance (From User Perspective)

**Users Trust That:**
- System meets **FDA requirements** for medical device software
- **IEC 62304 Class B** software lifecycle processes followed
- **ISO 14971 risk management** controls are implemented
- **HIPAA compliance** for patient data protection
- **Regular software updates** with proper validation

### 7.2 Quality Assurance

**Users Expect:**
- **Validated software**: Thorough testing before deployment
- **Documented procedures**: Clear SOPs for all operations
- **Version control**: Software version displayed and logged
- **Change control**: Controlled updates with change documentation
- **Audit readiness**: System logs support FDA/ISO audits

---

## 8. Training and Support Expectations

### 8.1 User Training Requirements

**What Users Need to Learn:**
- **Basic operation**: Sample loading, starting measurements, reading status
- **Daily calibration**: When and how to perform
- **Safety procedures**: Emergency stop, interlock understanding
- **Error handling**: Common error messages and corrective actions
- **Documentation**: How to log issues and report to service

**Expected Training Time:**
- Clinical operators: 4 hours initial training + supervised practice
- Administrators: 8 hours including system configuration
- Maintenance engineers: 16 hours including hardware service

### 8.2 Support and Documentation

**Users Expect:**
- **Quick reference guides** posted near device
- **On-screen help** for common procedures
- **Error code lookup** with troubleshooting steps
- **Technical support contact** clearly displayed
- **Regular software updates** with release notes

---

## 9. User Acceptance Criteria

### 9.1 Clinical Operator Acceptance

**System is acceptable if:**
- ✅ Can complete standard measurement in < 5 minutes start-to-finish
- ✅ Safety interlocks provide confidence in protection
- ✅ Calibration workflow is straightforward and reliable
- ✅ Error messages are understandable and actionable
- ✅ No measurement data has been lost in testing
- ✅ UI is responsive and intuitive

### 9.2 Administrator Acceptance

**System is acceptable if:**
- ✅ Complete audit trail available for all operations
- ✅ User management is secure and straightforward
- ✅ System health can be monitored remotely
- ✅ Data export works reliably for compliance reporting
- ✅ Backup and recovery procedures are documented

### 9.3 Maintenance Engineer Acceptance

**System is acceptable if:**
- ✅ Diagnostic information is comprehensive for troubleshooting
- ✅ Configuration updates follow secure procedures
- ✅ Hardware replacement is straightforward with clear procedures
- ✅ Calibration verification can be performed after service
- ✅ System logs provide detailed fault analysis

---

## 10. Summary of Core User Requirements

| Requirement ID | User Expectation | Priority | Status |
|----------------|-----------------|----------|--------|
| **USR_OMNI-SERVER_001** | **Reliable and safe operation**: Interlocks prevent unsafe radiation exposure | 🔴 CRITICAL | ✅ Implemented |
| **USR_OMNI-SERVER_002** | **Daily calibration capability**: Simple routine verifies device health | 🔴 CRITICAL | ✅ Implemented |
| **USR_OMNI-SERVER_003** | **Data integrity and traceability**: All measurements permanently linked to metadata | 🔴 CRITICAL | ✅ Implemented |

---

## 11. Future User-Requested Enhancements

**Based on User Feedback:**
- **Batch processing**: Load multiple samples and run unattended
- **Mobile notifications**: Alert when measurement complete or calibration needed
- **Advanced reporting**: Statistical analysis of measurement quality trends
- **Remote monitoring**: Authorized personnel can view status remotely
- **Predictive maintenance**: Alert before calibration drift or hardware issues

---

## Appendix A: User Personas

### Persona 1: Laboratory Technician (Primary User)
- **Name**: Sarah, Clinical Lab Technician
- **Experience**: 5 years in medical laboratory
- **Goals**: Process patient samples accurately and efficiently, maintain compliance
- **Concerns**: Patient safety, data accuracy, meeting turnaround times
- **Technical skill**: Moderate; comfortable with medical lab equipment

### Persona 2: Laboratory Manager (Secondary User)
- **Name**: Dr. James, Laboratory Director
- **Experience**: 15 years, responsible for quality and compliance
- **Goals**: Ensure regulatory compliance, maintain quality standards, manage staff
- **Concerns**: FDA audits, device validation, data integrity
- **Technical skill**: High; understands medical device regulations

### Persona 3: Service Engineer (Maintenance User)
- **Name**: Mike, Biomedical Equipment Technician
- **Experience**: 10 years servicing medical devices
- **Goals**: Maintain device uptime, troubleshoot issues efficiently
- **Concerns**: Device reliability, clear diagnostic information, parts availability
- **Technical skill**: Very high; hardware and software troubleshooting

---

## Appendix B: Requirement Traceability

| User Requirement | System Requirements | Risk Controls | Verification |
|-----------------|-------------------|---------------|--------------|
| USR_OMNI-SERVER_001 (Safe Operation) | SYS_OMNI-SERVER_001, 002, 003, 005, 009 | RISK_OMNI-SERVER_001, 002, 003 | VER_OMNI-SERVER_001, 003 |
| USR_OMNI-SERVER_002 (Calibration) | SYS_OMNI-SERVER_004, 008 | — | VER_OMNI-SERVER_002 |
| USR_OMNI-SERVER_003 (Traceability) | SYS_OMNI-SERVER_006, 007 | RISK_OMNI-SERVER_005 | VER_OMNI-SERVER_004, 005 |

---

**Document Control:**
- **Created**: 2025-10-24
- **Author**: Omniscan Development Team
- **Review Required**: Clinical staff, Quality assurance, Regulatory affairs
- **Next Review**: Prior to FDA submission
- **Classification**: FDA Design History File (DHF) - Requirements Phase
