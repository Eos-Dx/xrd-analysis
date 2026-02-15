# Omniscan Software User Requirements Specification

**Document Version:** 1.0  
**Date:** October 24, 2025  
**Classification:** FDA Design History File (DHF) - User Requirements  
**Compliance:** IEC 62304 Class B, ISO 14971, IEC 62366-1, HIPAA  
**Status:** Draft for Review

---

## Executive Summary

This document specifies the functional and operational requirements for the Omniscan medical diagnostic software from the user perspective. The requirements address safety, data integrity, usability, and regulatory compliance for a first-of-kind X-ray Diffraction (XRD) diagnostic system.

**Target User Groups:**
- **Clinical Operators** - Laboratory technicians performing diagnostic measurements
- **Clinicians** - Medical professionals reviewing diagnostic results
- **Maintenance Engineers** - Service personnel performing calibration and maintenance
- **IT Administrators** - Personnel managing system configuration and data security

---

## 1. Safety and Radiation Protection Requirements

### 1.1 Operator Safety Protection

**Requirement ID:** UR-SAFE-001  
**Priority:** CRITICAL

The software shall provide absolute protection from X-ray exposure under all unsafe conditions. The system must:

- Prevent X-ray beam activation when safety doors are not fully closed
- Immediately terminate all hazardous operations upon emergency stop activation
- Continuously monitor all safety interlocks and sensors with real-time status verification
- Automatically transition to SAFE state upon any safety fault detection
- Display clear visual and audible warnings when safety systems are not operational

**Rationale:** Operator safety is paramount in X-ray equipment operation. The software must enforce hardware interlocks to prevent radiation exposure.

**Verification:** Safety interlock testing, fault injection testing, emergency stop response time measurement

**Operational Scenario:** When a clinical operator attempts to load a sample with the safety door ajar, the software shall block X-ray initiation and display: "Safety door not fully closed - X-ray initiation prohibited."

### 1.2 Maintenance Mode Safety

**Requirement ID:** UR-SAFE-002  
**Priority:** CRITICAL

The software shall provide controlled maintenance access while maintaining safety audit trails. Requirements include:

- Maintenance mode accessible only through multi-factor authentication (password + physical key switch)
- Explicit interlock bypass capability with comprehensive warning displays
- Real-time logging of all bypassed safety systems and maintenance actions
- Automatic safety mode restoration upon maintenance session termination
- Complete audit trail of maintenance activities with timestamps and operator identification

**Rationale:** Maintenance personnel require diagnostic capabilities while maintaining safety accountability.

**Verification:** Maintenance mode access testing, audit log validation, automatic safety restoration testing

---

## 2. X-ray Exposure Control

### 2.1 Precise Exposure Management

**Requirement ID:** UR-EXPO-001  
**Priority:** CRITICAL

The software shall control X-ray exposure with precision to ensure patient sample integrity and operator safety. Requirements:

- Execute X-ray exposure for the exact programmed duration with ±50ms tolerance
- Implement real-time beam intensity monitoring with defined threshold limits
- Immediately terminate exposure if beam intensity deviates beyond ±10% of target
- Prevent concurrent exposure operations through software mutex controls
- Provide real-time visual feedback of exposure status and countdown timer
- Record all exposure parameters with millisecond-precision timestamps

**Rationale:** Precise exposure control ensures diagnostic quality and prevents over-exposure of samples.

**Verification:** Exposure timing accuracy testing, beam intensity monitoring validation, concurrent operation prevention testing

**Operational Scenario:** During a 60-second diagnostic measurement, if the X-ray source experiences intensity fluctuation exceeding thresholds, the software shall immediately abort the exposure and alert the operator to contact maintenance services.

### 2.2 Measurement Quality Validation

**Requirement ID:** UR-EXPO-002  
**Priority:** HIGH

The software shall provide clinicians with comprehensive quality indicators for measurement validation:

- Flag measurements where exposure parameters deviated from programmed values
- Report beam intensity variations with statistical analysis (mean, standard deviation)
- Identify data collected during active safety warnings or interlock violations
- Visually differentiate validated measurements from questionable data
- Provide quality metrics including SNR, beam stability, and calibration validity

**Rationale:** Clinicians must assess measurement reliability for diagnostic confidence.

**Verification:** Quality metric calculation validation, visual indicator testing, flagging accuracy assessment

---

## 3. Data Integrity and Security

### 3.1 Data Persistence and Fault Tolerance

**Requirement ID:** UR-DATA-001  
**Priority:** CRITICAL

The software shall ensure zero measurement data loss under all conditions, including system failures:

- Commit measurement data to persistent storage immediately upon acquisition completion
- Implement transaction-based data storage with automatic rollback on incomplete operations
- Automatically associate measurements with patient sample identifiers
- Implement write-once data storage with access control preventing accidental deletion
- Maintain full operational capability during network outages with local buffering
- Implement graceful shutdown with data preservation on unexpected system termination

**Rationale:** Patient diagnostic data is irreplaceable and must survive all failure scenarios.

**Verification:** Power failure testing, crash recovery testing, network disconnection testing, data integrity validation

**Operational Scenario:** If the operating system initiates an update reboot during measurement, the software shall either block the reboot until completion or preserve all acquired data with explicit notification of measurement status.

### 3.2 Data Security and HIPAA Compliance

**Requirement ID:** UR-DATA-002  
**Priority:** CRITICAL

The software shall implement comprehensive data protection compliant with HIPAA and cybersecurity standards:

- Encrypt all patient data at rest using AES-256-GCM encryption
- Encrypt data in transit using TLS 1.3 or TLS 1.2 minimum
- Maintain access logs for all patient data operations with user identification
- Store credentials and encryption keys in hardware-protected storage (TPM)
- Implement automated backup with cryptographic verification
- Provide secure data export with maintained encryption
- Support data retention policies with automated archival

**Rationale:** Patient data protection is legally mandated under HIPAA regulations.

**Verification:** Encryption validation, access logging verification, backup/restore testing, penetration testing

### 3.3 Complete Traceability

**Requirement ID:** UR-DATA-003  
**Priority:** HIGH

The software shall provide comprehensive traceability linking measurements to all contributing factors:

**Automatically captured metadata:**
- Device identification: Serial number, hardware configuration, firmware versions
- Operator identification: Username, credential level, authentication timestamp
- Temporal data: Measurement timestamp with timezone, duration, completion status
- Calibration linkage: Calibration reference ID, calibration timestamp, validity status
- Measurement parameters: Exposure settings, beam intensity, detector configuration
- Quality metrics: SNR, beam stability, distance verification results
- Environmental data: System state, active warnings, interlock status

**Rationale:** Complete traceability enables audit compliance and quality management.

**Verification:** Metadata completeness testing, traceability link validation, audit report generation testing

---

## 4. Calibration Management

### 4.1 Daily Calibration Enforcement

**Requirement ID:** UR-CAL-001  
**Priority:** CRITICAL

The software shall enforce daily calibration requirements to maintain diagnostic accuracy:

- Block diagnostic measurements if calibration has not been performed within 24 hours
- Display prominent calibration status indicators on all operator interfaces
- Provide guided calibration workflow with step-by-step instructions
- Automatically validate calibration results against defined acceptance criteria
- Display clear pass/fail indication with corrective action guidance
- Maintain calibration history with timestamps and results
- Transition system to LOCKED state when calibration expires

**Rationale:** Daily calibration ensures measurement accuracy and device reliability.

**Verification:** Calibration enforcement testing, 24-hour timeout validation, workflow usability testing

**Operational Scenario:** On Monday morning following a weekend, the system shall display: "Calibration expired - Last performed: Friday 8:15 AM. Daily calibration required before diagnostic measurements."

### 4.2 Calibration Diagnostics

**Requirement ID:** UR-CAL-002  
**Priority:** MEDIUM

The software shall provide maintenance engineers with comprehensive calibration diagnostic capabilities:

- Display detailed calibration parameters and measured values
- Provide graphical trending of calibration results over time
- Indicate specific parameters failing acceptance criteria
- Maintain historical calibration data for failure analysis
- Support multiple calibration attempts for verification
- Export calibration data for external analysis

**Rationale:** Detailed diagnostics enable efficient troubleshooting of calibration failures.

**Verification:** Diagnostic data accuracy testing, trending graph validation, export functionality testing

---

## 5. User Interface and Usability

### 5.1 Operator Interface Requirements

**Requirement ID:** UR-UI-001  
**Priority:** HIGH

The software shall provide an intuitive interface optimized for clinical workflow efficiency:

- Measurement initiation achievable in maximum 4 user interactions
- Large touch-friendly buttons visible from 2 meters distance
- Status display using plain language: "Ready", "Running", "Calibration Required", "Fault"
- Real-time status updates with no perceptible lag (<200ms)
- Error messages providing actionable corrective steps
- Minimal menu navigation for common operations
- Responsive interface with no freezing during measurements

**Rationale:** Efficient interface design reduces operator training time and operational errors.

**Verification:** Usability testing with representative users, response time measurement, workflow efficiency analysis

### 5.2 Clinician Interface Requirements

**Requirement ID:** UR-UI-002  
**Priority:** MEDIUM

The software shall provide clinicians with comprehensive data visualization and analysis capabilities:

- Display XRD patterns as high-resolution zoomable graphs
- Visual highlighting of quality issues and data anomalies
- Tabular presentation of measurement parameters for rapid review
- Data export in standard formats (CSV, JSON, DICOM)
- Clear visual grouping of measurements by patient sample
- Comparison tools for multiple measurements

**Rationale:** Clear data presentation enables accurate diagnostic interpretation.

**Verification:** Data visualization accuracy testing, export format validation, usability testing with clinicians

---

## 6. System Reliability

### 6.1 Availability and Recovery

**Requirement ID:** UR-REL-001  
**Priority:** HIGH

The software shall provide high availability and automatic recovery capabilities:

- Automatic startup as system service on computer boot
- Automatic hardware device reconnection after power cycling
- Clear system status indication during startup sequence
- Target uptime of 99% during operational hours
- Graceful handling of Windows updates without data loss
- Maximum restart time of 3 minutes to operational state
- Automatic recovery from transient hardware failures

**Rationale:** Clinical operations require reliable system availability.

**Verification:** Availability monitoring, restart time measurement, failure recovery testing

**Operational Scenario:** At 7:00 AM shift start, the software shall be fully operational with status indicating "Ready" or "Calibration Required" as appropriate.

### 6.2 Diagnostic and Troubleshooting Support

**Requirement ID:** UR-REL-002  
**Priority:** MEDIUM

The software shall provide IT administrators with comprehensive diagnostic capabilities:

- Structured log files with configurable verbosity levels
- Standard user account operation (no administrator rights required for normal use)
- Specific error messages including component identification and network addresses
- Service restart capability without full system reboot
- Compatibility with standard Windows updates and antivirus software
- Remote monitoring capability for system health
- Automated log rotation with configurable retention

**Rationale:** Supportability reduces downtime and vendor dependency.

**Verification:** Log analysis testing, restart procedure validation, update compatibility testing

---

## 7. Multi-User and Access Control

### 7.1 Session Management

**Requirement ID:** UR-USER-001  
**Priority:** HIGH

The software shall provide clear session management for multi-user environments:

- Display current operator name and session status prominently
- Show estimated time remaining for active measurements
- Prevent session interference through exclusive device access control
- Automatic session termination on operator logout or disconnect
- Session activity logging for audit trails
- Concurrent read-only monitoring for supervisory personnel

**Rationale:** Clear session management prevents operational conflicts in shared environments.

**Verification:** Multi-user testing, session timeout validation, concurrent access testing

### 7.2 Role-Based Access Control

**Requirement ID:** UR-USER-002  
**Priority:** HIGH

The software shall implement role-based access control (RBAC):

**Role Definitions:**
- **Clinical Operator**: Measurement execution, calibration, basic troubleshooting
- **Senior Staff**: Audit log access, historical data review, quality analysis
- **Maintenance Engineer**: Full diagnostic access, maintenance mode, configuration updates
- **Administrator**: User management, system configuration, security controls

**Requirements:**
- Graphical user management interface (no configuration file editing)
- Permission enforcement at software level with cryptographic authentication
- Failed access attempt logging
- Password complexity requirements and expiration policies
- Multi-factor authentication for privileged roles

**Rationale:** Access control ensures security and regulatory compliance.

**Verification:** Access control testing, privilege escalation testing, authentication validation

---

## 8. Error Handling and Recovery

### 8.1 Operator-Facing Error Messages

**Requirement ID:** UR-ERR-001  
**Priority:** HIGH

The software shall provide clear, actionable error messages for clinical operators:

**Message Components:**
- Plain language description of the problem
- Numbered checklist of user-actionable troubleshooting steps
- Clear indication of when to escalate to maintenance/IT support
- Contact information for technical support
- Unique error reference number for support coordination

**Example Acceptable Message:**
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

**Rationale:** Clear error messages reduce operator stress and improve problem resolution time.

**Verification:** Error message comprehension testing, troubleshooting success rate measurement

### 8.2 Technical Diagnostic Information

**Requirement ID:** UR-ERR-002  
**Priority:** MEDIUM

The software shall provide maintenance engineers and IT staff with comprehensive technical diagnostics:

- Detailed error codes and stack traces in structured log files
- Network connectivity details (IP addresses, ports, protocols)
- Timing information (timeouts, retry attempts, durations)
- Hardware communication logs with protocol-level details
- System state snapshots at time of error
- Log export capability for vendor support escalation

**Rationale:** Technical diagnostics enable efficient problem resolution by qualified personnel.

**Verification:** Log completeness testing, diagnostic information accuracy validation

---

## 9. Software Update Management

### 9.1 Update Deployment Control

**Requirement ID:** UR-UPD-001  
**Priority:** HIGH

The software shall provide controlled update deployment minimizing operational disruption:

- Comprehensive release notes detailing changes, fixes, and impacts
- Staged deployment capability (test → production)
- Rollback capability to previous version if issues arise
- Update installation scheduled during non-operational hours
- Advance notification of available updates (minimum 48 hours)
- Preservation of calibration data and configuration during updates
- User confirmation required before update installation
- Automatic backup before update application

**Rationale:** Controlled updates prevent operational disruptions and data loss.

**Verification:** Update/rollback testing, data preservation validation, notification timing verification

### 9.2 Update Impact Transparency

**Requirement ID:** UR-UPD-002  
**Priority:** MEDIUM

The software shall provide clear communication of update impacts:

- Estimated downtime duration
- Indication of workflow changes or UI modifications
- Calibration requirements post-update
- New features or capabilities introduced
- Deprecated features or compatibility changes
- Security patches and vulnerability remediation details

**Rationale:** Transparency enables users to plan for update impacts.

**Verification:** Release note completeness review, impact communication testing

---

## 10. Regulatory Compliance and Audit Support

### 10.1 Comprehensive Audit Trail

**Requirement ID:** UR-AUDIT-001  
**Priority:** CRITICAL

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

**Rationale:** Comprehensive audit trails are required for FDA compliance and quality management.

**Verification:** Audit trail completeness testing, tamper detection testing, export validation, regulatory compliance review

**Example Audit Queries:**
- "List all measurements performed by operator 'Sarah Johnson' in March 2025"
- "Prove calibration validity for measurement ID xyz-789"
- "Show all events on June 15, 2025 related to safety alarm incident"
- "Identify all instances of safety interlock bypass in past 12 months"
- "Display failed authentication attempts for user account 'admin'"

### 10.2 Quality Improvement Analytics

**Requirement ID:** UR-AUDIT-002  
**Priority:** MEDIUM

The software shall provide clinicians and quality managers with analytical capabilities:

- Measurement success/failure rate reporting
- Calibration trend analysis with statistical process control
- Sample re-run analysis with root cause categorization
- Quality control charts (X-bar, R charts) for key parameters
- Equipment utilization and performance metrics
- Operator performance comparison (anonymous)
- Predictive maintenance indicators

**Rationale:** Analytics support continuous quality improvement and proactive maintenance.

**Verification:** Analytics accuracy testing, report generation validation, statistical calculation verification

---

## 11. Network and Cloud Integration

### 11.1 Offline Operation Capability

**Requirement ID:** UR-NET-001  
**Priority:** CRITICAL

The software shall maintain full diagnostic capability during network outages:

- Complete measurement operations without cloud connectivity
- Local data storage with automatic synchronization upon reconnection
- Clear visual indication of network connectivity status
- No performance degradation due to network unavailability
- Local storage capacity monitoring with alerts at 80% utilization
- Data queue management with prioritization
- Automatic retry with exponential backoff for failed uploads

**Rationale:** Clinical operations cannot be dependent on network availability.

**Verification:** Offline operation testing, data synchronization validation, performance testing during network loss

**Operational Scenario:** At 2:00 PM, hospital network experiences outage. Clinical operators continue scheduled measurements with local storage. At 4:00 PM, network restores and all data automatically uploads to cloud with verification.

### 11.2 Network Management

**Requirement ID:** UR-NET-002  
**Priority:** MEDIUM

The software shall provide IT administrators with network management capabilities:

- Real-time cloud connectivity status dashboard
- Upload queue monitoring with data volume indicators
- Failed upload notification and manual retry capability
- Bandwidth throttling controls to prevent network saturation
- Cloud synchronization pause/resume controls
- Network connectivity diagnostics and logging
- Secure proxy configuration support

**Rationale:** Network management capabilities enable IT oversight and troubleshooting.

**Verification:** Network monitoring accuracy testing, bandwidth control validation, proxy configuration testing

---

## 12. Performance Requirements

### 12.1 Responsiveness

**Requirement ID:** UR-PERF-001  
**Priority:** HIGH

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

**Rationale:** Responsive performance supports efficient clinical workflow.

**Verification:** Performance benchmarking under normal and stress conditions

### 12.2 Capacity and Scalability

**Requirement ID:** UR-PERF-002  
**Priority:** MEDIUM

The software shall support the following operational capacity:

- 100 measurements per day per device
- 10 years of historical data storage locally
- 10 concurrent users (with single active operator)
- Continuous 12-hour daily operation
- 30-day uptime without restart requirement
- 1TB local data storage minimum
- Sub-second performance degradation over time

**Rationale:** Capacity planning ensures long-term system viability.

**Verification:** Load testing, long-duration operation testing, storage capacity testing

---

## 13. Training and Documentation

### 13.1 User Training Requirements

**Requirement ID:** UR-TRAIN-001  
**Priority:** MEDIUM

The software shall support efficient user training:

**Training Time Targets:**
- Clinical operators: 4 hours to basic proficiency
- Administrators: 8 hours including configuration
- Maintenance engineers: 16 hours including diagnostics

**Training Support:**
- In-application help system with context-sensitive content
- Visual workflow guides for common operations
- Interactive training mode with simulated samples
- Quick reference cards exportable as PDF
- Video tutorials for complex procedures

**Rationale:** Efficient training reduces operational errors and deployment time.

**Verification:** Training time measurement with representative users, help system usability testing

### 13.2 Documentation Requirements

**Requirement ID:** UR-TRAIN-002  
**Priority:** MEDIUM

The software shall be supported by comprehensive documentation:

- User manual with step-by-step procedures and screenshots
- Administrator guide covering installation, configuration, and maintenance
- Troubleshooting guide with flowcharts and decision trees
- API documentation for integration developers
- Release notes for all versions
- Searchable online help system
- Emergency procedure posters (printable)

**Rationale:** Complete documentation supports independent problem resolution.

**Verification:** Documentation completeness review, accuracy verification, usability testing

---

## 14. Contraindications - Requirements for What Software Must NOT Do

### 14.1 Forbidden Behaviors

The software shall NOT exhibit the following behaviors:

**Safety Violations:**
- Never allow X-ray activation with safety interlocks unsatisfied
- Never bypass safety systems without explicit maintenance mode authorization
- Never fail silently on safety system faults

**Data Integrity Violations:**
- Never allow data modification post-commit
- Never store patient data unencrypted
- Never lose data without explicit notification

**Usability Violations:**
- Never display technical error codes to clinical operators without plain language explanation
- Never require manual data saving for measurement results
- Never implement ambiguous UI elements that can be misinterpreted

**System Behavior Violations:**
- Never perform automatic updates during operational hours
- Never require system restart to recover from common errors
- Never hard-code configuration parameters that vary by installation

**Rationale:** Explicit prohibition of dangerous or frustrating behaviors improves safety and usability.

---

## 15. Summary of Critical Requirements

| Requirement ID | Category | Description | Priority |
|---------------|----------|-------------|----------|
| UR-SAFE-001 | Safety | Absolute protection from unsafe X-ray exposure | CRITICAL |
| UR-EXPO-001 | Exposure Control | Precise X-ray exposure timing and intensity control | CRITICAL |
| UR-DATA-001 | Data Integrity | Zero data loss under all failure conditions | CRITICAL |
| UR-DATA-002 | Security | HIPAA-compliant encryption and access control | CRITICAL |
| UR-CAL-001 | Calibration | Daily calibration enforcement with 24-hour timeout | CRITICAL |
| UR-AUDIT-001 | Compliance | Complete tamper-evident audit trail | CRITICAL |
| UR-NET-001 | Reliability | Full offline operation capability | CRITICAL |

---

## 16. Requirements Traceability

| User Requirement | Doorstop System Requirement | Risk Control | Verification Method |
|-----------------|----------------------------|--------------|---------------------|
| UR-SAFE-001 | SYS_OMNI-SERVER_001, 002, 003 | RISK_OMNI-SERVER_001, 002, 003 | VER_OMNI-SERVER_001, 003 |
| UR-EXPO-001 | SYS_OMNI-SERVER_005 | RISK_OMNI-SERVER_001 | VER_OMNI-SERVER_001 |
| UR-DATA-001 | SYS_OMNI-SERVER_006 | RISK_OMNI-SERVER_005 | VER_OMNI-SERVER_004 |
| UR-DATA-002 | SYS_OMNI-SERVER_007, 009 | RISK_OMNI-SERVER_004 | VER_OMNI-SERVER_005 |
| UR-CAL-001 | SYS_OMNI-SERVER_004 | - | VER_OMNI-SERVER_002 |
| UR-AUDIT-001 | SYS_OMNI-SERVER_007 | RISK_OMNI-SERVER_004 | VER_OMNI-SERVER_004, 005 |

---

## Document Control

**Revision History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-24 | Omniscan Development Team | Initial release |

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

## Appendix A: Glossary

- **Calibration**: Verification procedure ensuring measurement accuracy using reference standards
- **Exposure**: Period during which X-ray beam is active for sample measurement
- **Interlock**: Hardware safety mechanism preventing unsafe operations
- **Session**: Period of authenticated user access to the system
- **Traceability**: Ability to track measurement data to contributing factors and personnel

## Appendix B: Regulatory References

- FDA 21 CFR Part 11 - Electronic Records and Signatures
- IEC 62304 - Medical Device Software Lifecycle Processes
- ISO 14971 - Application of Risk Management to Medical Devices
- IEC 62366-1 - Application of Usability Engineering to Medical Devices
- HIPAA Security Rule - 45 CFR Part 164
- FDA Cybersecurity Guidance (2023)

---

**End of Document**
