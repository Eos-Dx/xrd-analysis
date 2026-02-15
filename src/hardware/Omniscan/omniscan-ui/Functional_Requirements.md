# Omniscan UI Functional Requirements

## Overview
This document defines the functional requirements for the Omniscan Medical XRD Diagnostic System UI, aligned with IEC 62304 Class B medical device standards.

---

## 1. User Roles & Access Control

### 1.1 Role Definitions

**Operator**
- Run measurements
- View measurement results
- Perform daily calibration
- View system status
- Access: Dashboard, Measurement, Calibration (limited)

**Engineer**
- All operator permissions
- Hardware diagnostics
- System configuration (read-only)
- Advanced troubleshooting
- Access: All operator pages + Maintenance

**Administrator**
- All engineer permissions
- User management
- System configuration (read/write)
- Audit log access
- Access: All pages including Admin panel

### 1.2 Authentication Requirements

**REQ-AUTH-001:** User must authenticate with user ID and password  
**REQ-AUTH-002:** Session must expire after 30 minutes of inactivity  
**REQ-AUTH-003:** Failed login attempts must be logged  
**REQ-AUTH-004:** User must be able to log out manually  
**REQ-AUTH-005:** Session ID must be used for all API requests  

---

## 2. Safety Requirements

### 2.1 Interlock Monitoring

**REQ-SAFE-001:** Safety interlock status must be displayed at all times  
**REQ-SAFE-002:** All interlock fields must be required (not optional)  
**REQ-SAFE-003:** Interlock changes must trigger immediate UI updates via WebSocket  
**REQ-SAFE-004:** Critical interlock failures must display prominent alerts  
**REQ-SAFE-005:** System must prevent measurements when interlocks are violated  

**Required Interlock Fields:**
- overall_safe
- key_switch
- enable_button
- door_closed
- emergency_stop
- radiation_safe
- cooling_ok
- power_ok

### 2.2 Emergency Controls

**REQ-SAFE-010:** Emergency abort button must be accessible from all pages  
**REQ-SAFE-011:** Abort action must not require confirmation dialog  
**REQ-SAFE-012:** Abort confirmation must be displayed after action  
**REQ-SAFE-013:** System state must update immediately after abort  

### 2.3 Visual Safety Indicators

**REQ-SAFE-020:** System state must use color coding:
- Green: IDLE, safe to operate
- Yellow: PENDING_ARMED, STOPPING (transitional states)
- Red: SAFE (fault condition)
- Blue: RUNNING (active operation)

**REQ-SAFE-021:** Status indicators must be large and clearly visible  
**REQ-SAFE-022:** Critical status changes must show notifications  

---

## 3. Measurement Workflow

### 3.1 Pre-Measurement Checks

**REQ-MEAS-001:** System must verify calibration is valid before measurement  
**REQ-MEAS-002:** System must verify all interlocks are satisfied  
**REQ-MEAS-003:** System must verify all devices are initialized  
**REQ-MEAS-004:** Start button must be disabled if any check fails  
**REQ-MEAS-005:** UI must display reason for disabled start button  

### 3.2 Measurement Parameters

**REQ-MEAS-010:** User must enter Sample ID (required field)  
**REQ-MEAS-011:** User must set exposure duration in milliseconds  
**REQ-MEAS-012:** System must validate exposure duration range (min/max)  
**REQ-MEAS-013:** Optional: Link measurement to patient MRN  
**REQ-MEAS-014:** Optional: Add measurement notes  

### 3.3 Measurement Execution

**REQ-MEAS-020:** Measurement must start immediately after user confirms  
**REQ-MEAS-021:** UI must show real-time progress updates  
**REQ-MEAS-022:** User must be able to stop measurement at any time  
**REQ-MEAS-023:** User must be able to abort measurement (emergency)  
**REQ-MEAS-024:** System must display elapsed time during measurement  
**REQ-MEAS-025:** System must display estimated time remaining  

### 3.4 Measurement Completion

**REQ-MEAS-030:** UI must display success notification on completion  
**REQ-MEAS-031:** UI must display error notification on failure  
**REQ-MEAS-032:** Measurement must be saved to history immediately  
**REQ-MEAS-033:** User must be able to view results immediately  
**REQ-MEAS-034:** Measurement ID must be displayed for traceability  

### 3.5 Measurement History

**REQ-MEAS-040:** UI must display table of recent measurements  
**REQ-MEAS-041:** Table must show: timestamp, sample ID, operator, status  
**REQ-MEAS-042:** Table must be sortable by column  
**REQ-MEAS-043:** User must be able to filter measurements  
**REQ-MEAS-044:** User must be able to export measurement data  

---

## 4. Calibration Workflow

### 4.1 Calibration Status Display

**REQ-CAL-001:** Current calibration status must be visible in header  
**REQ-CAL-002:** Status must show: Valid, Expired, or Never Calibrated  
**REQ-CAL-003:** Expiration time must be displayed  
**REQ-CAL-004:** Visual indicator must use color coding (green/red/yellow)  
**REQ-CAL-005:** Calibration must be valid for 24 hours  

### 4.2 Starting Calibration

**REQ-CAL-010:** User must navigate to Calibration page  
**REQ-CAL-011:** UI must verify hardware readiness before allowing start  
**REQ-CAL-012:** UI must display readiness checklist:
- PDU initialized
- GPIO initialized
- Detector initialized
- Motion initialized

**REQ-CAL-013:** Start button must be disabled if hardware not ready  
**REQ-CAL-014:** Calibration must be asynchronous (non-blocking)  
**REQ-CAL-015:** API must return immediately (< 100ms)  

### 4.3 Calibration Progress

**REQ-CAL-020:** UI must show progress indicator during calibration  
**REQ-CAL-021:** Progress indicator must include:
- Spinner animation
- "Calibration in Progress" message
- Calibration ID
- Estimated time (optional)

**REQ-CAL-022:** User must be able to navigate away during calibration  
**REQ-CAL-023:** Progress indicator must persist across page navigation  
**REQ-CAL-024:** System must prevent starting multiple calibrations  

### 4.4 Calibration Completion

**REQ-CAL-030:** System must broadcast completion via WebSocket  
**REQ-CAL-031:** UI must display notification on completion  
**REQ-CAL-032:** Notification must indicate success or failure  
**REQ-CAL-033:** QC report must auto-display on calibration page  
**REQ-CAL-034:** All connected clients must receive notification  

### 4.5 QC Report Display

**REQ-CAL-040:** QC report must display:
- Overall PASS/FAIL status
- Calibration ID
- Timestamp
- Calibrant material (e.g., LaB₆)

**REQ-CAL-041:** QC report must show four quality checks:
- Total Intensity (measured vs threshold)
- Goodness of Fit (measured vs threshold)
- Signal-to-Noise Ratio (measured vs threshold)
- Ring Quality (measured vs threshold)

**REQ-CAL-042:** Each QC check must display:
- Check name
- PASS/FAIL badge
- Measured value (3 decimal places)
- Threshold value (3 decimal places)
- Details/description

**REQ-CAL-043:** QC report must show PONI calibration results:
- Sample distance (mm)
- Wavelength (Å)
- Beam center X (pixels)
- Beam center Y (pixels)
- SUCCESS/FAILED indicator

**REQ-CAL-044:** QC report must show formatted text report  
**REQ-CAL-045:** User must be able to download report as `.txt` file  
**REQ-CAL-046:** Filename must be `calibration-{calibration_id}.txt`  

### 4.6 Accepting Calibration

**REQ-CAL-050:** "Accept Calibration" button must appear only when overall_pass is true  
**REQ-CAL-051:** Failed calibrations must not be acceptable  
**REQ-CAL-052:** User must be able to close report without accepting (review only)  
**REQ-CAL-053:** Accepting calibration must show success notification  

### 4.7 Viewing Previous Calibrations

**REQ-CAL-060:** UI must fetch latest calibration on page load  
**REQ-CAL-061:** "View Last Calibration Report" button must appear if calibration exists  
**REQ-CAL-062:** Clicking button must display full QC report  
**REQ-CAL-063:** System must handle 404 gracefully (no previous calibration)  
**REQ-CAL-064:** Previous reports must be read-only (no accept button)  

### 4.8 Timeout Protection

**REQ-CAL-070:** Calibration must have 13-second timeout  
**REQ-CAL-071:** Timeout must prevent hanging if hardware fails  
**REQ-CAL-072:** Timeout must broadcast error notification  
**REQ-CAL-073:** User must receive timeout error message  
**REQ-CAL-074:** System must allow new calibration after timeout  

---

## 5. Patient Management

### 5.1 Patient Registration

**REQ-PAT-001:** User must be able to register new patients  
**REQ-PAT-002:** Required fields: First name, Last name, MRN, Date of birth  
**REQ-PAT-003:** Optional fields: Gender, additional notes  
**REQ-PAT-004:** MRN must be unique  
**REQ-PAT-005:** Date of birth must be validated  

### 5.2 Patient Search

**REQ-PAT-010:** User must be able to search by MRN  
**REQ-PAT-011:** Search must return exact matches  
**REQ-PAT-012:** System must handle "not found" gracefully  
**REQ-PAT-013:** Search results must display patient summary  

### 5.3 Patient Selection

**REQ-PAT-020:** User must be able to link patient to measurement  
**REQ-PAT-021:** Patient selection must be optional  
**REQ-PAT-022:** Selected patient must be displayed during measurement  
**REQ-PAT-023:** Patient data must be included in measurement record  

---

## 6. System Status Monitoring

### 6.1 Dashboard Display

**REQ-STAT-001:** Dashboard must show system state (IDLE, RUNNING, etc.)  
**REQ-STAT-002:** Dashboard must show device status for:
- PDU (powered, status)
- GPIO (powered, status)
- Detector (powered, status, temperature)
- Motion (powered, status, position)

**REQ-STAT-003:** Dashboard must show safety interlock status  
**REQ-STAT-004:** Dashboard must show calibration status  
**REQ-STAT-005:** Dashboard must show active measurement (if any)  

### 6.2 Real-Time Updates

**REQ-STAT-010:** System must use WebSocket for real-time updates  
**REQ-STAT-011:** Updates must occur within 1 second of change  
**REQ-STAT-012:** UI must show WebSocket connection status  
**REQ-STAT-013:** UI must auto-reconnect on disconnect  
**REQ-STAT-014:** UI must display connection lost warning  

### 6.3 Device Status

**REQ-STAT-020:** Device status must use standard values:
- Detector: OFF, IDLE, EXPOSING, READING, ERROR
- Motion: OFF, IDLE, MOVING, HOMING, ERROR, LIMIT_HIT
- GPIO: OFF, IDLE, ERROR

**REQ-STAT-021:** Device status must use color coding  
**REQ-STAT-022:** Initialized status must be displayed separately  
**REQ-STAT-023:** Error states must show error details  

---

## 7. Hardware Control

### 7.1 Device Initialization

**REQ-HW-001:** Engineer/Admin must be able to initialize devices  
**REQ-HW-002:** Initialization must be per-device (PDU, GPIO, Detector, Motion)  
**REQ-HW-003:** UI must show progress during initialization  
**REQ-HW-004:** UI must show success/failure notification  
**REQ-HW-005:** Initialization errors must display details  

### 7.2 Device Shutdown

**REQ-HW-010:** Engineer/Admin must be able to stop devices  
**REQ-HW-011:** Stop action must require confirmation  
**REQ-HW-012:** UI must show progress during stop  
**REQ-HW-013:** UI must show success/failure notification  

### 7.3 Diagnostics

**REQ-HW-020:** Engineer must be able to view detailed diagnostics  
**REQ-HW-021:** Diagnostics must show:
- Device health metrics
- Uptime
- Temperature (detector)
- Voltage (detector)
- Position (motion)
- Total operations count

**REQ-HW-022:** Diagnostics must update in real-time  
**REQ-HW-023:** GPIO diagnostics must show:
- Key switch status
- Activation button status
- Activation timer (if active)
- LED states

---

## 8. Data Integrity & Traceability

### 8.1 Audit Trail

**REQ-DATA-001:** All user actions must be logged  
**REQ-DATA-002:** Audit log must include:
- User ID
- Action type
- Timestamp
- Affected entities
- Action result (success/failure)

**REQ-DATA-003:** Audit logs must be immutable  
**REQ-DATA-004:** Audit logs must be accessible to administrators  

### 8.2 Measurement Traceability

**REQ-DATA-010:** Each measurement must have unique ID  
**REQ-DATA-011:** Measurement must link to:
- Operator user ID
- Calibration ID
- Patient ID (if applicable)
- Sample ID
- System state at time of measurement

**REQ-DATA-012:** Measurement records must be immutable  
**REQ-DATA-013:** Measurement metadata must include:
- Start timestamp
- End timestamp
- Exposure duration
- Device states

### 8.3 Calibration Traceability

**REQ-DATA-020:** Each calibration must have unique ID  
**REQ-DATA-021:** Calibration must link to operator user ID  
**REQ-DATA-022:** Calibration QC report must be stored permanently  
**REQ-DATA-023:** Calibration must include timestamp  
**REQ-DATA-024:** Measurements must reference calibration used  

---

## 9. Error Handling

### 9.1 API Errors

**REQ-ERR-001:** Network errors must display user-friendly messages  
**REQ-ERR-002:** API errors must show error details  
**REQ-ERR-003:** Critical errors must prevent unsafe operations  
**REQ-ERR-004:** Transient errors must support retry  
**REQ-ERR-005:** Error messages must not expose sensitive information  

### 9.2 WebSocket Errors

**REQ-ERR-010:** Connection loss must show warning notification  
**REQ-ERR-011:** System must attempt automatic reconnection  
**REQ-ERR-012:** Failed reconnection must show manual reconnect option  
**REQ-ERR-013:** Connection status must be visible in UI  

### 9.3 Validation Errors

**REQ-ERR-020:** Form validation must occur on submit  
**REQ-ERR-021:** Validation errors must display next to fields  
**REQ-ERR-022:** User must not be able to submit invalid forms  
**REQ-ERR-023:** Required fields must be clearly marked  

---

## 10. User Experience

### 10.1 Performance

**REQ-UX-001:** Page load time must be < 3 seconds  
**REQ-UX-002:** API responses must be < 1 second (excluding hardware operations)  
**REQ-UX-003:** UI must show loading indicators for operations > 500ms  
**REQ-UX-004:** Large tables must support pagination  
**REQ-UX-005:** Heavy operations must not block UI  

### 10.2 Notifications

**REQ-UX-010:** Notifications must appear in top-right corner  
**REQ-UX-011:** Notification types: Success (green), Warning (yellow), Error (red), Info (blue)  
**REQ-UX-012:** Notifications must auto-dismiss after 5 seconds (except errors)  
**REQ-UX-013:** User must be able to dismiss notifications manually  
**REQ-UX-014:** Multiple notifications must stack vertically  

### 10.3 Navigation

**REQ-UX-020:** Navigation menu must be visible on all pages  
**REQ-UX-021:** Current page must be highlighted in navigation  
**REQ-UX-022:** User info must be visible in header  
**REQ-UX-023:** Logout button must be accessible from header  
**REQ-UX-024:** System status must be visible in header  

### 10.4 Responsive Design

**REQ-UX-030:** UI must be usable on desktop (primary)  
**REQ-UX-031:** UI must be readable on tablets  
**REQ-UX-032:** Critical functions must work on mobile (view-only)  
**REQ-UX-033:** Minimum screen width: 1280px (desktop), 768px (tablet)  

---

## 11. Compliance & Security

### 11.1 HIPAA Compliance

**REQ-COMP-001:** Patient data must not be logged in plain text  
**REQ-COMP-002:** Patient data must be transmitted securely (TLS in production)  
**REQ-COMP-003:** Access to patient data must be audited  
**REQ-COMP-004:** Patient data must be encrypted at rest (backend responsibility)  

### 11.2 IEC 62304 Compliance

**REQ-COMP-010:** All software changes must be traceable  
**REQ-COMP-011:** Risk analysis must be performed for new features  
**REQ-COMP-012:** Verification and validation must be documented  
**REQ-COMP-013:** Change control must be enforced  

### 11.3 FDA Cybersecurity

**REQ-COMP-020:** Session IDs must be cryptographically secure  
**REQ-COMP-021:** Passwords must not be stored in UI code  
**REQ-COMP-022:** API keys must not be exposed in client code  
**REQ-COMP-023:** XSS protection must be enabled  
**REQ-COMP-024:** CSRF protection must be implemented  

---

## 12. Workflow Summary

### Simple 4-Click Measurement Workflow

**REQ-WORK-001:** Measurement must be achievable in 4 clicks:
1. Click "Measurement" in navigation
2. Enter sample ID
3. Set exposure duration
4. Click "Start Measurement"

**REQ-WORK-002:** Pre-checks must be automatic (no manual verification)  
**REQ-WORK-003:** Workflow must have minimal distractions  
**REQ-WORK-004:** Status messages must use plain language  

### Clinical Reliability

**REQ-WORK-010:** UI must enforce daily calibration  
**REQ-WORK-011:** Expired calibration must block measurements  
**REQ-WORK-012:** Safety violations must block measurements  
**REQ-WORK-013:** All critical actions must have audit trail  

---

## 13. Acceptance Criteria

### Definition of Done

**REQ-ACC-001:** Feature must compile without errors  
**REQ-ACC-002:** Feature must pass TypeScript strict mode  
**REQ-ACC-003:** Feature must be tested manually  
**REQ-ACC-004:** Feature must have error handling  
**REQ-ACC-005:** Feature must be documented  
**REQ-ACC-006:** Feature must follow existing patterns  
**REQ-ACC-007:** Feature must not break existing functionality  

### Quality Standards

**REQ-ACC-010:** Code must follow project conventions  
**REQ-ACC-011:** UI must be accessible (keyboard navigation)  
**REQ-ACC-012:** UI must have proper loading states  
**REQ-ACC-013:** UI must have proper error states  
**REQ-ACC-014:** UI must have proper empty states  

---

**Last Updated:** 2025-11-04  
**Compliance:** IEC 62304 Class B, HIPAA, FDA Cybersecurity Guidelines
