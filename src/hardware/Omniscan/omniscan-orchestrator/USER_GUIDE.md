# Omniscan Orchestrator - User Guide

## Overview

This document provides user-facing documentation, quickstart guides, and operational procedures for the Omniscan Orchestrator.

---

## Quick Start

### For Clinical Operators

#### Daily Workflow

```bash
# 1. Login to system via UI
#    - Enter credentials
#    - System validates access

# 2. Perform morning calibration
#    - Click "Start Calibration" button
#    - Click ENABLE button (physical hardware)
#    - Wait for calibration to complete (~5-10 seconds)
#    - Verify QC checks passed

# 3. Search for patient
#    - Enter medical record number (MRN)
#    - Select patient from results

# 4. Start measurement
#    - Select sample ID (e.g., "left_breast_1")
#    - Set exposure time (default: 5000ms)
#    - Click "Start Measurement"
#    - Click ENABLE button (20-second window)
#    - Wait for measurement to complete

# 5. Review results
#    - Check QC status (pass/fail)
#    - Review diffraction data
#    - Add clinical notes
#    - Save results

# 6. End of day
#    - System automatically backs up database
#    - Logout from UI
```

### For Maintenance Engineers

#### Quick Connection

```bash
# 1. Generate certificate (once per day)
omni-orch cert generate \
  --engineer-id ENG001 \
  --device-uuid ABC123 \
  --validity-days 1

# 2. Start interactive session
omni-orch interactive \
  --cert C:\...\client_ENG001_ABC123.crt \
  --key C:\...\client_ENG001_ABC123.key \
  --ca-cert C:\...\device_server_root_ca.crt

# 3. Inside session
omniscan> status              # Check system state
omniscan> enter-maintenance 900  # Enter maintenance (15 min)
omniscan> get-config          # View configuration
omniscan> exit-maintenance    # Exit when done
omniscan> quit
```

#### Helper Script (Recommended)

Save as `engineer-connect.ps1`:

```powershell
# Configuration
$ENGINEER_ID = "ENG001"
$DEVICE_UUID = "ABC123"
$CERT_DIR = "C:\dev\Omniscan\omniscan-certificate-center\certs"

# Certificate paths
$cert = "$CERT_DIR\client\client_${ENGINEER_ID}_${DEVICE_UUID}.crt"
$key = "$CERT_DIR\client\client_${ENGINEER_ID}_${DEVICE_UUID}.key"
$ca = "$CERT_DIR\root\device_server_root_ca.crt"

# Check certificate age
if (Test-Path $cert) {
    $age = (Get-Date) - (Get-Item $cert).LastWriteTime
    if ($age.TotalHours -gt 23) {
        Write-Host "⚠️  Certificate is $($age.TotalHours.ToString('0.0')) hours old"
        Write-Host "Generate new certificate? (Y/N)"
        $response = Read-Host
        if ($response -eq "Y") {
            omni-orch cert generate --engineer-id $ENGINEER_ID --device-uuid $DEVICE_UUID
        }
    }
} else {
    Write-Host "📝 Generating certificate..."
    omni-orch cert generate --engineer-id $ENGINEER_ID --device-uuid $DEVICE_UUID
}

# Execute command
omni-orch $args --cert $cert --key $key --ca-cert $ca --base-url https://localhost:8443/api/v1
```

Usage:
```bash
# Run any command through helper
.\engineer-connect.ps1 status
.\engineer-connect.ps1 interactive
.\engineer-connect.ps1 enter-maintenance --ttl 900
```

---

## Enable Button Workflow Guide

### What is the Enable Button?

The Enable Button is a **physical safety confirmation** required before potentially harmful operations (X-ray activation, motion system initialization).

### When is it Required?

#### Require Enable Button ✅:
- Initialize detector (powers on X-ray source)
- Initialize motion (activates motion system)
- Start exposure (activates X-rays)
- Motion commands (physical movement)

#### Do NOT Require ⚫:
- Read states (status queries)
- Stop operations (emergency stop)
- Power off devices (safe shutdown)
- View data (results, calibration)

### How to Use

```
1. Operator receives prompt: "Click ENABLE button to proceed"
2. Operator physically presses ENABLE button on control panel
3. Button activates for 20 seconds
4. Within 20 seconds, operator clicks "Confirm" in UI
5. Operation proceeds if all safety interlocks satisfied
```

### Troubleshooting

#### "Enable button not active" error persists
**Cause**: Button expired before command sent (20-second timeout)

**Solution**: 
1. Click ENABLE button again
2. Immediately click "Confirm" in UI
3. Ensure minimal delay between physical button and UI confirmation

#### Can't initialize devices even with button active
**Possible Causes**:
1. Key switch not ON → Turn key switch to ON position
2. Emergency stop pressed → Release emergency stop
3. Door open → Close safety door
4. Interlocks not satisfied → Check all safety interlocks in UI

#### Button timeout too short for workflow
**Solution**: Contact system administrator to increase timeout in hardware server config (requires service restart)

---

## Interactive Session Guide

### Starting a Session

```bash
# With engineer certificates
omni-orch interactive \
  --cert <cert-path> \
  --key <key-path> \
  --ca-cert <ca-path>

# With USB maintenance stub
omni-orch interactive --usb-path <usb-directory>
```

### Available Commands

#### Maintenance Operations
```
omniscan> status                    # Get server status
omniscan> enter-maintenance 1800    # Enter maintenance (30 min)
omniscan> renew 900                 # Renew maintenance lease (15 min)
omniscan> exit-maintenance          # Exit maintenance mode
```

#### Configuration Management
```
omniscan> get-config               # View current configuration
omniscan> set-config config.json   # Load new configuration
omniscan> patch-config {"key": "value"}  # Update specific setting
omniscan> validate-config config.json    # Validate configuration file
```

#### Device Control
```
omniscan> device-state             # Check device states
omniscan> device-power on          # Power on devices
omniscan> device-power off         # Power off devices
omniscan> measure-start 120 calibrant   # Start calibration (120s)
omniscan> measure-status           # Check measurement progress
omniscan> measure-stop             # Stop measurement
omniscan> measure-result           # Get measurement results
```

#### Session Management
```
omniscan> help                     # Show all commands
omniscan> clear                    # Clear screen
omniscan> quit                     # Exit session (or 'exit')
```

### Keyboard Shortcuts

- **↑/↓**: Navigate command history
- **Tab**: Auto-complete commands
- **Ctrl+C**: Cancel current input
- **Ctrl+D**: Exit session
- **Ctrl+L**: Clear screen

### Tips

1. **Use History**: Press ↑ to quickly repeat commands
2. **Tab Completion**: Type partial commands and press Tab
3. **Keep Sessions Short**: Certificates typically expire after 1 day
4. **Check Status First**: Run `status` before operations

---

## Certificate Management

### For Engineers

#### Generating Certificates

```bash
# Generate 1-day certificate (default)
omni-orch cert generate \
  --engineer-id ENG001 \
  --device-uuid ABC123

# Generate 8-hour certificate
omni-orch cert generate \
  --engineer-id ENG001 \
  --device-uuid ABC123 \
  --validity-days 0.33

# Generate 7-day certificate (emergency)
omni-orch cert generate \
  --engineer-id ENG001 \
  --device-uuid ABC123 \
  --validity-days 7
```

#### Viewing Certificates

```bash
# List all certificates
omni-orch cert list

# View certificate details
omni-orch cert info --cert-path <path-to-cert>
```

#### Using Certificates

```bash
# Single command with certificate
omni-orch status \
  --cert <cert-path> \
  --key <key-path> \
  --ca-cert <ca-path>

# Use environment variables
$env:OMNISCAN_CERT = "<cert-path>"
$env:OMNISCAN_KEY = "<key-path>"
$env:OMNISCAN_CA = "<ca-path>"

omni-orch status --cert $env:OMNISCAN_CERT --key $env:OMNISCAN_KEY --ca-cert $env:OMNISCAN_CA
```

### Certificate Lifecycle

```
Day 1, 8:00 AM: Generate certificate (valid 24 hours)
Day 1, 8:00 AM - Day 2, 8:00 AM: Certificate valid for operations
Day 2, 8:00 AM: Certificate expires
Day 2, 8:05 AM: Generate new certificate
```

### Security Best Practices

✅ **DO**:
- Generate certificates daily
- Delete old certificates after use
- Store private keys securely
- Use device-specific certificates

❌ **DON'T**:
- Share certificates between engineers
- Commit certificates to version control
- Use expired certificates
- Store certificates on network shares

---

## Calibration Workflow

### Daily Calibration Procedure

```
1. Start of Day
   - Ensure system powered on
   - Check all interlocks satisfied
   - Verify key switch ON

2. Prepare Calibration Sample
   - Load calibration material (LaB6 or CeO2)
   - Position in beam path
   - Verify sample alignment

3. Start Calibration
   - Click "Start Calibration" in UI
   - Click ENABLE button (physical)
   - Wait for completion (~5-10 seconds)

4. Review QC Results
   - Total intensity: > 10,000 counts (pass)
   - Goodness of fit: > 0.95 (pass)
   - SNR: > 30 (pass)
   - Ring quality: > 0.90 (pass)

5. If QC Failed
   - Review error messages
   - Check sample positioning
   - Verify detector temperature stable
   - Retry calibration
   - Contact maintenance if repeated failures

6. If QC Passed
   - Calibration valid for 24 hours
   - All subsequent measurements linked to this calibration
   - Remove calibration sample
   - Proceed with patient measurements
```

### Calibration Validity

- **Valid for**: 24 hours from calibration time
- **Status check**: GET /api/calibration/status
- **Expiry warning**: System warns 1 hour before expiry
- **Expired calibration**: System enters LOCKED state, requires new calibration

---

## Patient Measurement Workflow

### Complete Workflow

```
1. Search Patient
   - Enter medical record number (MRN)
   - Verify patient details displayed
   - If not found: Create new patient record

2. Create Patient (if needed)
   - First name
   - Last name
   - Date of birth
   - Medical record number
   - Click "Create Patient"

3. Select Sample
   - Choose sample ID (e.g., "left_breast_1", "right_breast_2")
   - Or create custom sample ID
   - Set exposure time (default: 5000ms)

4. Start Measurement
   - Click "Start Measurement"
   - System checks:
     • Calibration valid ✓
     • Detector initialized ✓
     • Safety interlocks satisfied ✓
   - Click ENABLE button (20-second window)
   - Click "Confirm" in UI

5. Monitor Progress
   - Progress bar shows exposure duration
   - Real-time detector temperature
   - Beam intensity graph
   - Estimated time remaining

6. Review Results
   - QC status: pass/fail
   - Diffraction pattern displayed
   - Signal-to-noise ratio (SNR)
   - Detector temperature at measurement
   - Calibration context shown

7. Add Notes
   - Clinical observations
   - Any abnormalities
   - QC notes (if failed)
   - Click "Save"

8. Next Patient
   - Search for next patient
   - Repeat measurement workflow
```

### QC Checks

#### Pass Criteria:
- SNR > 30
- Detector temperature: 20-30°C
- Beam intensity within range
- No detector errors
- Calibration valid

#### Fail Criteria:
- Low SNR (< 30)
- Detector temperature out of range
- Beam intensity too low/high
- Detector errors detected
- Expired calibration

### Troubleshooting Measurements

#### Low Signal-to-Noise Ratio (SNR < 30)
**Causes**:
- Sample positioning incorrect
- Exposure time too short
- Beam intensity low
- Detector temperature unstable

**Solutions**:
- Reposition sample
- Increase exposure time
- Check beam source
- Allow detector to stabilize

#### Measurement Timeout
**Cause**: Hardware server unresponsive (>13 seconds)

**Solutions**:
1. Check hardware server status
2. Restart hardware server if needed
3. Check network connection
4. Retry measurement

#### Enable Button Errors
See "Enable Button Workflow Guide" section above

---

## System Monitoring

### Health Checks

#### For Operators (via UI)
- **System State**: Displayed in header (IDLE, RUNNING, LOCKED, etc.)
- **Device Status**: Detector and motion status indicators
- **Calibration Status**: Valid/expired indicator with time remaining
- **Safety Interlocks**: All interlock states visible

#### For Engineers (via CLI)
```bash
# Overall system health
omni-orch status

# GPIO and safety interlocks
omni-rest gpio state

# Device states
omni-rest system state

# Enable button status
omni-rest gpio enable-button
```

### Database Monitoring

```bash
# Check database size
dir C:\dev\Omniscan\omniscan-orchestrator\data\orchestrator.db

# List recent backups
dir C:\dev\Omniscan\omniscan-orchestrator\data\backups\

# Check backup age
$backup = Get-Item "C:\dev\Omniscan\omniscan-orchestrator\data\backups\*.db" | 
          Sort-Object LastWriteTime -Descending | 
          Select-Object -First 1
$age = (Get-Date) - $backup.LastWriteTime
Write-Host "Latest backup age: $($age.TotalHours) hours"
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Cannot Login
**Symptoms**: Login fails with "Authentication failed"

**Checks**:
- Verify username and password correct
- Check session not already active
- Verify database accessible

**Solutions**:
- Reset password (contact administrator)
- Close other browser tabs/sessions
- Restart orchestrator service

#### 2. System State LOCKED
**Symptoms**: Cannot start measurements, state shows LOCKED

**Cause**: Calibration expired or invalid

**Solution**:
1. Check calibration status
2. Perform new calibration
3. Verify QC checks pass
4. System automatically unlocks

#### 3. Enable Button Timeout
**Symptoms**: "Enable button not active" errors

**Cause**: 20-second timeout expired

**Solution**:
1. Click ENABLE button
2. Immediately click UI confirmation
3. Reduce delay between physical button and UI

#### 4. Detector Not Responding
**Symptoms**: Initialize detector fails

**Checks**:
- Enable button active?
- Key switch ON?
- Safety interlocks satisfied?
- Detector powered?

**Solutions**:
1. Press ENABLE button
2. Turn key switch ON
3. Close safety door
4. Check detector power supply
5. Contact maintenance if persists

#### 5. gRPC Connection Error
**Symptoms**: "Hardware server unavailable"

**Cause**: Hardware server not running or network issue

**Solutions**:
1. Check hardware server status (should be running)
2. Restart hardware server service
3. Check firewall allows port 50051
4. Verify network connectivity

#### 6. Database Locked
**Symptoms**: Operations fail with "database locked"

**Cause**: Multiple connections or crashed process

**Solutions**:
1. Close all UI sessions
2. Restart orchestrator service
3. Check for orphaned processes
4. Contact administrator if persists

---

## Maintenance Procedures

### Daily Tasks (Automated)

- ✅ **Database backup**: Automatic at midnight
- ✅ **Audit log rotation**: Automatic daily
- ✅ **Session cleanup**: Expired sessions removed hourly

### Weekly Tasks (Manual)

- [ ] **Review audit logs**: Check for anomalies
- [ ] **Check backup status**: Verify daily backups completing
- [ ] **Test measurements**: Perform test calibration and measurement
- [ ] **Review error logs**: Check system_event_log for errors

### Monthly Tasks (Administrator)

- [ ] **Backup restore test**: Verify backup can be restored
- [ ] **Database vacuum**: Optimize database performance
- [ ] **Certificate review**: Check for expiring certificates
- [ ] **Security audit**: Review access logs
- [ ] **Software updates**: Install patches and updates

### Annual Tasks (Administrator)

- [ ] **Encryption key rotation**: Rotate database encryption keys
- [ ] **Server certificate renewal**: Renew TLS/mTLS certificates
- [ ] **Compliance audit**: Full regulatory compliance review
- [ ] **Disaster recovery test**: Test full system recovery

---

## Operator Best Practices

### Safety

1. **Always check interlocks** before starting measurement
2. **Never bypass safety mechanisms** (door, emergency stop, key switch)
3. **Use enable button** as designed - physical confirmation required
4. **Monitor detector temperature** during measurements
5. **Follow calibration schedule** - daily calibration required

### Quality

1. **Perform daily calibration** at start of day
2. **Verify QC checks pass** before patient measurements
3. **Document anomalies** in clinical notes
4. **Review results before saving** - check SNR, temperature, QC status
5. **Report equipment issues** immediately to maintenance

### Privacy

1. **Never share login credentials**
2. **Logout when leaving workstation**
3. **Verify patient identity** before starting measurement
4. **No patient names in notes** transmitted to other systems
5. **Report data breaches** immediately

### Efficiency

1. **Use search shortcuts** - MRN search fastest
2. **Batch similar samples** - reduce setup time
3. **Pre-position next sample** during measurement
4. **Review previous measurements** for baseline comparison
5. **Keep calibration samples ready** for morning calibration

---

## Support & Contact

### For Operators

- **Equipment Issues**: Contact on-site maintenance engineer
- **Software Issues**: Restart orchestrator service, contact IT if persists
- **Emergency**: Press emergency stop, call safety officer

### For Engineers

- **Technical Documentation**: See IMPLEMENTATION_GUIDE.md
- **API Documentation**: See API_REFERENCE.md
- **Security & Compliance**: See SECURITY_COMPLIANCE.md
- **Architecture**: See ARCHITECTURE.md

### Emergency Contacts

- **Safety Officer**: [Contact Info]
- **IT Support**: [Contact Info]
- **Maintenance Engineer**: [Contact Info]
- **System Administrator**: [Contact Info]

---

## Glossary

- **Enable Button**: Physical safety confirmation button (20-second timeout)
- **Calibration**: Daily procedure to calibrate detector geometry (LaB6/CeO2)
- **QC**: Quality Control - automated checks on measurements and calibrations
- **MRN**: Medical Record Number - unique patient identifier
- **SNR**: Signal-to-Noise Ratio - measurement quality metric
- **PONI**: Point of Normal Incidence - detector geometry parameters
- **mTLS**: Mutual TLS - both client and server authenticate with certificates
- **gRPC**: Remote Procedure Call protocol used between orchestrator and hardware
- **UUID**: Universally Unique Identifier - used for measurements (no PII)
- **PII**: Personally Identifiable Information - patient names, DOB, MRN (never transmitted)

---

## Summary

The Omniscan Orchestrator provides:

✅ **Safety-first operation** with enable button workflow  
✅ **Privacy-protected** patient data management  
✅ **Complete audit trail** for regulatory compliance  
✅ **Daily calibration** with automated QC checks  
✅ **Real-time monitoring** via WebSocket updates  
✅ **Certificate-based authentication** for engineers  
✅ **Interactive CLI** for rapid development  
✅ **Automated backups** with 30-day retention  

For additional technical details, see companion documentation files.
