# Omniscan Software User Requirements

**Document Version:** 1.0  
**Date:** October 24, 2025  
**Classification:** FDA Design History File (DHF) - User Requirements Specification  
**Purpose:** Comprehensive description of user expectations for Omniscan software functionality

---

## Document Scope

This document defines the functional and operational requirements for the Omniscan software from the user perspective. It specifies what clinical operators, clinicians, maintenance engineers, and administrators require from the software to perform their roles safely and effectively.

**Target User Groups:**
- **Clinical Operators** - Laboratory technicians performing diagnostic measurements
- **Clinicians** - Medical professionals reviewing diagnostic results and patient data
- **Maintenance Engineers** - Service personnel performing calibration, diagnostics, and maintenance
- **IT Administrators** - Personnel managing system configuration, security, and data management

---

## 1. Safety and Radiation Protection

### 1.1 Clinical Operator Safety Requirements

**Primary Requirement:** The software shall provide absolute protection from X-ray exposure under all unsafe conditions.

The software must enforce the following safety controls for clinical operators:

- Prevent X-ray beam activation when the safety door is open under any circumstances
- Respond immediately to emergency stop button activation by cutting power to all hazardous systems
- Continuously monitor all safety switches and interlock sensors
- Automatically shut down all operations upon detection of any safety fault condition
- Display clear visual warnings when any safety system is not in operational state

**Operational Scenario:** When a clinical operator loads a patient sample without fully closing the safety door, the software shall refuse to initiate X-ray exposure and display a clear message: "Safety door not closed - cannot start measurement."

### 1.2 Maintenance Engineer Safety Requirements

**Primary Requirement:** The software shall provide secure maintenance access while maintaining comprehensive safety controls and audit trails.

The software must support maintenance operations through:

- A dedicated maintenance mode requiring multi-factor authentication (password and physical key switch)
- Controlled diagnostic capabilities with safety interlock bypass authority (with explicit warnings)
- Comprehensive logging of active and bypassed safety systems during maintenance sessions
- Automatic restoration of full safety mode upon maintenance session termination
- Complete audit trail of all maintenance activities and configuration changes

---

## 2. X-ray Exposure Control and Patient Protection

### 2.1 Exposure Control Requirements for Clinical Operators

**Primary Requirement:** The software shall provide precise control over X-ray exposure duration and intensity to ensure patient safety and data quality.

The software must implement the following exposure control mechanisms:

- Activate X-ray beam for the exact programmed exposure duration only
- Immediately terminate X-ray exposure if beam intensity exceeds or falls below defined thresholds
- Prevent concurrent exposure operations through mutex locking mechanisms
- Display real-time exposure status with clear visual indicators
- Maintain permanent records of all exposures with precise timestamps and exposure parameters

**Operational Scenario:** When a clinical operator initiates a 60-second measurement, the software shall display a real-time countdown timer. If the X-ray source experiences a malfunction causing beam intensity deviation, the software shall automatically terminate the exposure and display an alert instructing the operator to contact service personnel.

### 2.2 Measurement Validation Requirements for Clinicians

**Primary Requirement:** The software shall provide comprehensive quality indicators enabling clinicians to assess measurement validity and reliability.

The software must provide the following validation capabilities:

- Flag measurements with non-compliant X-ray exposure parameters
- Report beam intensity variations throughout the measurement duration
- Identify data collected during active safety warnings or fault conditions
- Clearly distinguish between validated and questionable measurement data through visual indicators

---

## 3. Data Integrity and Security

### 3.1 Data Persistence Requirements for Clinical Operators

**Primary Requirement:** The software shall ensure zero data loss under all operational conditions, including system failures and network interruptions.

The software must implement the following data protection mechanisms:

- Commit measurement data to persistent storage immediately upon acquisition completion
- Implement fault-tolerant data storage with transaction logging to preserve data integrity during system crashes
- Automatically associate each measurement with the corresponding patient sample identifier
- Prevent accidental data overwrite or deletion through write-once mechanisms and access controls
- Maintain full operational capability during network connectivity loss with local data buffering

**Operational Scenario:** When a measurement is in progress and the operating system initiates an update-triggered reboot, the software shall either block the reboot until measurement completion or implement graceful shutdown with data preservation, providing clear notification of any data loss to the operator.

### 3.2 Data Security Requirements for IT Administrators

**Primary Requirement:** The software shall implement comprehensive data protection controls compliant with HIPAA regulations and industry security standards.

The software must provide the following security capabilities:

- Encrypt all patient data stored locally using AES-256-GCM or equivalent encryption standards
- Encrypt data in transit to cloud services using TLS 1.2 or higher
- Maintain comprehensive access logs recording all patient data access events with user identification
- Store credentials and encryption keys using secure key management systems (hardware TPM or equivalent)
- Provide automated backup capabilities with verification mechanisms to ensure backup integrity

### 3.3 Traceability Requirements for Clinicians

**Primary Requirement:** The software shall provide complete traceability from measurement results to all contributing factors, enabling comprehensive audit trails and quality assurance.

The software must automatically capture and link the following metadata to each measurement:

- Device identification (Omniscan unit serial number and hardware configuration)
- Operator identification (laboratory technician name/ID and credentials)
- Temporal information (precise timestamp with timezone)
- Calibration linkage (calibration reference used and calibration validity timestamp)
- Measurement parameters (complete exposure settings and acquisition parameters)

The software shall implement traceability as an automated system function, not dependent on manual operator data entry beyond required identifiers.

---

## 4. Daily Calibration - Keeping the Machine Accurate

### What Lab Technicians Need

**The software must make daily calibration simple and force me to do it.**

Every morning before I run patient tests, I need to calibrate the machine. The software should:

- Remind me if I haven't calibrated today (or block me from running patient tests)
- Walk me through the calibration steps clearly
- Tell me pass or fail - not confusing numbers I have to interpret
- If calibration fails, tell me what to do: "Call service" or "Try again"
- Remember when the last good calibration was done

**Real-life scenario:** I come in Monday morning after a weekend. The software shows a big orange banner: "Calibration expired - last calibrated Friday at 8:15 AM. Please run daily calibration before patient measurements."

### What Maintenance Engineers Need

**The software must help me diagnose calibration problems.**

When a lab calls me because calibration failed, I need:

- Detailed calibration results (not just pass/fail)
- Graphs showing calibration trends over time
- Clear indication of which parameter failed
- Historical data so I can see if this is a new problem or getting worse
- The ability to run calibration multiple times to verify repairs

---

## 5. Easy to Use - Not Complicated

### What Lab Technicians Need

**The software must be simple enough for my busy workday.**

I handle dozens of samples per day and don't have time for complicated software. I expect:

- Starting a measurement takes 3-4 clicks maximum
- Big buttons I can see from across the room
- Status shown in plain English: "Ready", "Running", "Calibration needed"
- Error messages that tell me what to DO, not just what went wrong
- Common tasks shouldn't require scrolling through menus

**What I don't want:**
- Technical jargon in error messages
- Needing to remember command sequences
- The software crashing and losing my place
- Pop-ups that interrupt measurements

### What Clinicians Need

**The software must present results clearly.**

When I'm reviewing diagnostic results, the software should:

- Show the XRD pattern as a clear, zoomable graph
- Highlight any quality issues visually
- Present measurement parameters in a table I can quickly scan
- Let me export data for my own analysis tools
- Make it obvious which results are from the same patient

---

## 6. Reliability - It Must Work When I Need It

### What Lab Technicians Need

**The software must be running when I come in each morning.**

I can't afford to lose time troubleshooting computers. The software needs to:

- Start automatically when the computer boots
- Reconnect to hardware devices if they were power-cycled
- Show me clearly if it's ready or still starting up
- Not crash multiple times per week
- Recover gracefully if Windows updates overnight

**Real-life scenario:** I arrive at 7 AM to start the day's testing. The software should already be running, showing "Ready" status, with a note if calibration is needed.

### What IT Administrators Need

**The software must be supportable by me, not just by the vendor.**

I need to keep this system running without calling vendor support every time. The software should:

- Have clear log files I can read to diagnose problems
- Not require administrator rights for normal operation (lab techs run as standard users)
- Tell me specifically what's wrong: "Cannot connect to detector at 192.168.1.100"
- Have a documented way to restart services without rebooting
- Not conflict with our standard Windows updates and antivirus

---

## 7. Multi-User Environment

### What Lab Technicians Need

**The software must show me who's using the machine.**

If another technician is running a test, I need to know:

- Who is currently logged in and using the machine
- How long until their measurement is done
- Whether I can queue up my sample or need to wait
- My session doesn't interfere with theirs

### What IT Administrators Need

**The software must control who can do what.**

Different users need different permissions:

- Lab technicians can run measurements and calibrations
- Senior staff can review audit logs
- Only maintenance engineers can enter maintenance mode
- I can add/remove users without editing config files
- User permissions are enforced by the software, not honor system

---

## 8. When Things Go Wrong - Error Handling

### What Lab Technicians Need

**The software must help me fix problems, not just report them.**

When an error happens, I don't want cryptic messages. I expect:

**Good error message:**
> "X-ray detector not responding. Please check:
> 1. Detector power cable is connected
> 2. Network cable is plugged in
> 3. Detector power light is green
> If problem continues, call service: 1-800-555-1234"

**Bad error message:**
> "Error code 0x8007274C - BIS socket timeout exception"

The software should:
- Explain what happened in plain language
- Tell me what I can try to fix it
- Tell me when to call for help vs. when I can fix it myself
- Not make me remember error codes

### What Maintenance Engineers Need

**The software must give me technical details when I ask.**

While lab techs need simple messages, I need the technical details:

- Error codes and stack traces in a separate log file
- Exact network addresses and port numbers
- Timing information (how long did it wait before timeout?)
- What the software tried to do before failing
- A way to export logs to send to vendor support

---

## 9. Software Updates and Maintenance

### What IT Administrators Need

**The software must update without breaking everything.**

When a software update is released, I need:

- Clear release notes explaining what changed
- A way to test the update before deploying to production
- The ability to roll back if something goes wrong
- Updates that don't require recalibration unless necessary
- Notification of updates well in advance, not surprise automatic updates

### What Lab Technicians Need

**Software updates must not interrupt my work.**

I run patient tests all day, so:

- Updates should install after-hours or when I choose
- If an update requires downtime, tell me how long
- After update, my workflow should be the same (don't surprise me with UI changes)
- Don't lose my calibration when you update

---

## 10. Compliance and Audit Support

### What Administrators Need

**The software must help me pass audits, not make them harder.**

When FDA or hospital auditors come, the software must:

- Provide a complete audit trail of all operations
- Export logs in a readable format (CSV, PDF)
- Show me who did what, when, and why
- Prove that safety interlocks are always enforced
- Demonstrate data cannot be altered after collection

**Questions an auditor might ask (software must answer these):**
- "Show me all measurements performed by technician Sarah in March 2025"
- "Prove that calibration was valid for this patient measurement"
- "What happened on June 15th when there was a safety alarm?"
- "Has anyone disabled safety interlocks in the past year?"
- "Show me all failed login attempts"

### What Clinicians Need

**The software must support quality improvement.**

I need to review system performance over time:

- Success/failure rates for measurements
- Trends in calibration results (is the machine drifting?)
- Which samples had to be re-run and why
- Statistical quality control charts
- This information helps me know if we need service or training

---

## 11. Cloud Integration and Offline Operation

### What Lab Technicians Need

**The software must keep working if the internet goes down.**

Our internet isn't perfect, and patient care can't wait. The software should:

- Let me run measurements even if cloud is unreachable
- Store all data locally first, upload to cloud when available
- Show me clearly if cloud connection is down
- Not slow down because it's trying to reach the cloud
- Warn me if local storage is getting full

**Real-life scenario:** The hospital network goes down at 2 PM. I can still run my scheduled patient tests, the software stores everything locally, and uploads automatically when network comes back at 4 PM.

### What IT Administrators Need

**The software must handle network problems gracefully.**

I need to know:

- Current connection status to cloud services
- How much data is queued for upload
- Whether any uploads failed and need retry
- Network bandwidth usage (not saturating our connection)
- The ability to pause cloud sync during backups

---

## 12. Performance Expectations

### What Lab Technicians Need

**The software must keep up with my pace.**

When I'm processing multiple samples, timing matters:

- Clicking "Start Measurement" should respond in under 2 seconds
- Status updates should be real-time, not delayed
- The interface shouldn't freeze during measurements
- Searching for old results should be fast (under 5 seconds)
- The software shouldn't slow down after running for hours

### What System Should Handle

**Expected workload:**
- 50-100 measurements per day per machine
- 5-10 years of historical data stored locally
- Multiple users accessing the system throughout the day
- Continuous operation during business hours (8-12 hours/day)
- Software should run for weeks without restart

---

## 13. Training and Documentation

### What Lab Technicians Need

**The software must be learnable without weeks of training.**

A new technician should be able to:

- Learn basic operations in 2-4 hours
- Access help within the software (not a separate manual)
- Get visual guides for common tasks
- Practice on test samples before doing patient work
- Have a quick-reference card for rare operations

### What All Users Need

**Documentation that actually helps:**
- Step-by-step guides with screenshots
- Troubleshooting flowcharts
- Video tutorials for complex procedures
- Searchable help system
- Emergency procedures clearly posted

---

## 14. What Users Don't Want

Sometimes it's easier to describe what we DON'T want:

### Lab Technicians Don't Want:
- ❌ Confusing error messages with no solution
- ❌ Having to remember to manually save data
- ❌ The software guessing what I meant (be explicit)
- ❌ Features that seemed like a good idea but complicate daily work
- ❌ Pop-ups that require clicking "OK" when I'm across the room

### Clinicians Don't Want:
- ❌ Data formatted in ways I can't analyze
- ❌ Results that can't be exported to my own tools
- ❌ Uncertainty about whether data is valid
- ❌ Having to call IT to access historical results

### Maintenance Engineers Don't Want:
- ❌ "Black box" software I can't diagnose
- ❌ Logs that tell me nothing useful
- ❌ Features locked behind vendor service calls
- ❌ Undocumented configuration files

### IT Administrators Don't Want:
- ❌ Software that requires specific old Windows versions
- ❌ Hard-coded IP addresses I can't change
- ❌ Dependence on outdated libraries with security vulnerabilities
- ❌ "Call vendor support" for every minor issue

---

## 15. Summary - The Core Promise

**If we had to summarize what users expect in one paragraph:**

The Omniscan software must guarantee safety above all else - never allowing X-ray exposure when unsafe, and controlling dose precisely. It must reliably save every measurement with complete traceability, even if computers crash or networks fail. Daily calibration must be enforced but simple. The interface should be clear enough that a technician can run tests confidently after a few hours of training. When errors happen, the software should help fix them, not just report them. Updates must not disrupt patient care. And through it all, the software must provide a complete audit trail proving that safety, accuracy, and data integrity were maintained at every step.

**In other words:** Be safe, be reliable, be clear, and help us do our jobs well.

---

## Document Information

**Created:** October 2025  
**Contributors:** Lab staff, clinical reviewers, IT team, maintenance engineers  
**Review Cycle:** Every 6 months or when major software changes are planned  
**Purpose:** To keep developers focused on real user needs, not just technical requirements  

**This is a living document.** As we learn from using the system, we'll update what we expect the software to do.
