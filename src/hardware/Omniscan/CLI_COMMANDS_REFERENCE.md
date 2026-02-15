# Omniscan CLI Commands Reference

## Overview
The `omni-orch` CLI provides command-line access to hardware initialization, power control, and device diagnostics for the Omniscan XRD system.

---

## Prerequisites

### Environment Variables Setup
Before running any commands, set up the required certificate environment variables:

```powershell
# PowerShell (Windows)
$env:OMNISCAN_CERT = "C:\path\to\omniscan-cert.pem"
$env:OMNISCAN_KEY = "C:\path\to\omniscan-key.pem"
$env:OMNISCAN_CA = "C:\path\to\ca-cert.pem"
$env:OMNISCAN_URL = "https://localhost:50051"  # or your orchestrator URL
```

Or in `.env` file:
```bash
OMNISCAN_CERT=/path/to/omniscan-cert.pem
OMNISCAN_KEY=/path/to/omniscan-key.pem
OMNISCAN_CA=/path/to/ca-cert.pem
OMNISCAN_URL=https://localhost:50051
```

### Certificate Files
- **OMNISCAN_CERT**: Client certificate (PEM format)
- **OMNISCAN_KEY**: Client private key (PEM format)
- **OMNISCAN_CA**: CA certificate (PEM format)
- **OMNISCAN_URL**: gRPC server address

---

## Detector Commands

### Initialize Detector

**Command:**
```powershell
omni-orch initialize-detector `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL
```

**What It Does:**
1. Powers on the detector subsystem
2. Runs initialization sequence
3. Performs self-checks
4. Returns detector status and health

**Preconditions (All Required):**
- ✅ **Key switch** = ON
- ✅ **Enable/Activation button** = PRESSED and ACTIVE (within 20s window)
- ✅ **Radiation** = SAFE (beam physically blocked)
- ✅ **Cooling** = OK (system running, no blockages)
- ✅ **Power** = OK (PDU providing power)
- ⊙ **Door** = Any state (open, closed, doesn't matter)

**Success Response:**
```json
{
  "status": "initialized",
  "powered": true,
  "initialized": true,
  "detector_status": "IDLE",
  "temperature": 22.5,
  "voltage": 48.0,
  "total_exposures": 0
}
```

**Error Response (Enable Button Not Active):**
```
Error: Enable/Activation button not active - required for detector initialization
Detail: Click ACTIVATE/ENABLE button and retry within 20 seconds
```

**Error Response (Other Issues):**
```
Error: Radiation is NOT SAFE (beam not blocked) - block beam before initializing detector
Error: Cooling is NOT OK - check cooling system before initializing detector
Error: Power is NOT OK - check power before initializing detector
Error: Key switch must be ON for detector initialization
```

**Typical Duration:** 2-3 seconds

**When to Use:**
- System startup
- After power cycle
- Before running measurements
- After maintenance

---

### Power Off Detector

**Command:**
```powershell
omni-orch power-off-detector `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL
```

**What It Does:**
1. Powers down the detector subsystem
2. Disables cooling
3. Performs safe shutdown sequence

**Preconditions:**
- None - this is a SAFE operation
- **No enable button required**
- Can be run anytime

**Success Response:**
```json
{
  "status": "powered_off",
  "detector_status": "OFF"
}
```

**Typical Duration:** 1-2 seconds

**When to Use:**
- End of measurement session
- System shutdown
- Emergency power down
- Maintenance

---

## Motion Commands

### Initialize Motion

**Command:**
```powershell
omni-orch initialize-motion `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL
```

**What It Does:**
1. Powers on motion control system
2. Homes all axes
3. Performs calibration
4. Initializes position tracking
5. Returns status and current position

**Preconditions (Same as Detector):**
- ✅ **Key switch** = ON
- ✅ **Enable/Activation button** = PRESSED and ACTIVE (within 20s window)
- ✅ **Radiation** = SAFE (beam physically blocked)
- ✅ **Cooling** = OK
- ✅ **Power** = OK
- ⊙ **Door** = Any state

**Success Response:**
```json
{
  "status": "initialized",
  "powered": true,
  "initialized": true,
  "is_homed": true,
  "motion_status": "IDLE",
  "position": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
  },
  "total_moves": 0
}
```

**Error Response Examples:**
```
Error: Enable/Activation button not active - required for motion initialization
Error: Radiation is NOT SAFE (beam not blocked) - block beam before initializing motion
```

**Typical Duration:** 3-5 seconds (includes homing routine)

**When to Use:**
- System startup
- After power cycle
- Before positioning operations
- After maintenance

---

### Power Off Motion

**Command:**
```powershell
omni-orch power-off-motion `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL
```

**What It Does:**
1. Powers down motion control system
2. De-energizes stepper motors
3. Performs safe shutdown sequence

**Preconditions:**
- None - SAFE operation, no enable button required

**Success Response:**
```json
{
  "status": "powered_off",
  "motion_status": "OFF"
}
```

**Typical Duration:** 1 second

**When to Use:**
- End of session
- System shutdown
- Emergency stop
- Maintenance

---

## Diagnostics Commands

### Get Device State

**Command:**
```powershell
omni-orch get-device-state `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL
```

**What It Does:**
1. Queries detector health and status
2. Queries motion health and status
3. Returns comprehensive device information
4. Reports statistics and diagnostics

**Preconditions:**
- None - READ-ONLY, no enable button required

**Success Response:**
```json
{
  "detector": {
    "powered": true,
    "status": "IDLE",
    "initialized": true,
    "temperature": 22.5,
    "voltage": 48.0,
    "total_exposures": 42,
    "last_exposure_time_ms": 5000,
    "uptime_seconds": 3600
  },
  "motion": {
    "powered": true,
    "status": "IDLE",
    "initialized": true,
    "is_homed": true,
    "position": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0
    },
    "target_position": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0
    },
    "total_moves": 128,
    "uptime_seconds": 3600
  },
  "timestamp": "2025-10-28T10:38:00Z"
}
```

**Field Descriptions:**

**Detector Fields:**
- `powered`: True if detector subsystem has power
- `status`: IDLE, INIT, EXPOSING, READING, ERROR
- `initialized`: True if detector is calibrated and ready
- `temperature`: Detector temperature in Celsius
- `voltage`: Detector operating voltage
- `total_exposures`: Number of exposures since power-on
- `last_exposure_time_ms`: Duration of last exposure
- `uptime_seconds`: Seconds since detector powered on

**Motion Fields:**
- `powered`: True if motion control has power
- `status`: IDLE, INIT, MOVING, HOMING, ERROR, LIMIT_HIT
- `initialized`: True if axes are homed and ready
- `is_homed`: True if all axes have been homed
- `position`: Current position (x, y, z in mm)
- `target_position`: Commanded position
- `total_moves`: Number of moves since power-on
- `uptime_seconds`: Seconds since motion powered on

**Typical Duration:** <100ms (read-only query)

**When to Use:**
- Status checks before/after operations
- Troubleshooting
- Monitoring
- Logging system state
- Pre-measurement verification

---

## Certificate Parameters

All commands require the same certificate parameters:

| Parameter | Environment Variable | Purpose |
|-----------|----------------------|---------|
| `--cert` | `$env:OMNISCAN_CERT` | Client certificate for mTLS |
| `--key` | `$env:OMNISCAN_KEY` | Client private key for mTLS |
| `--ca-cert` | `$env:OMNISCAN_CA` | CA certificate to verify server |
| `--base-url` | `$env:OMNISCAN_URL` | gRPC server address |

---

## Quick Reference: Command Summary

| Command | Requires Enable Button | Typical Use |
|---------|------------------------|-------------|
| `initialize-detector` | ✅ YES | System startup |
| `initialize-motion` | ✅ YES | System startup |
| `power-off-detector` | ❌ NO | Shutdown |
| `power-off-motion` | ❌ NO | Shutdown |
| `get-device-state` | ❌ NO | Diagnostics |

---

## Safety Workflow: Step-by-Step

### Scenario: Initialize Full System

**Step 1: Verify Physical Prerequisites**
```powershell
# Check key switch is ON on GPIO panel
# Verify beam block is installed and closed
# Confirm cooling system running (listen for pump)
# Check power distribution unit is on
```

**Step 2: Check Current State**
```powershell
omni-orch get-device-state `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL

# Output shows: detector OFF, motion OFF
```

**Step 3: Initialize Detector**
```powershell
# Press ENABLE button on GPIO panel (20s timer starts)
# Quickly return to computer and run:

omni-orch initialize-detector `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL

# Output: detector initialized, status IDLE, temp 22.5°C
```

**Step 4: Initialize Motion**
```powershell
# Press ENABLE button on GPIO panel again
# Quickly return and run:

omni-orch initialize-motion `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL

# Output: motion initialized, homed, ready
```

**Step 5: Verify All Systems Ready**
```powershell
omni-orch get-device-state `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL

# Output shows: both detector and motion powered, initialized, ready
# Now ready for calibration or measurements
```

---

## Error Handling

### Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Enable button not active" | Didn't press ENABLE button | Go to GPIO panel, press ENABLE, retry within 20s |
| "Radiation is NOT SAFE" | Beam not blocked | Install/close beam block, try again |
| "Cooling is NOT OK" | Cooling system offline | Check cooling pump power, verify no blockages |
| "Power is NOT OK" | PDU not providing power | Check PDU power switch and connections |
| "Key switch must be ON" | Physical key switch OFF | Turn key switch ON on GPIO panel |
| "Connection timeout" | Server unreachable | Verify OMNISCAN_URL is correct |
| "Certificate error" | Invalid certs or wrong path | Verify cert files exist and are valid |

---

## Scripting Examples

### PowerShell Initialization Script

```powershell
# init-system.ps1
# Full system initialization with error checking

param(
    [string]$Cert = $env:OMNISCAN_CERT,
    [string]$Key = $env:OMNISCAN_KEY,
    [string]$CaCert = $env:OMNISCAN_CA,
    [string]$BaseUrl = $env:OMNISCAN_URL
)

Write-Host "🚀 Starting system initialization..." -ForegroundColor Green

# Check preconditions
Write-Host "📋 Checking system state..." -ForegroundColor Cyan
$state = omni-orch get-device-state `
  --cert $Cert `
  --key $Key `
  --ca-cert $CaCert `
  --base-url $BaseUrl

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to read device state" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Device state OK" -ForegroundColor Green

# Initialize detector
Write-Host "`n🔧 Initializing detector..." -ForegroundColor Cyan
Write-Host "   Press ENABLE button on GPIO panel, then press ENTER..."
Read-Host

$detector = omni-orch initialize-detector `
  --cert $Cert `
  --key $Key `
  --ca-cert $CaCert `
  --base-url $BaseUrl

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Detector initialization failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Detector initialized" -ForegroundColor Green
Write-Host $detector

# Initialize motion
Write-Host "`n🔧 Initializing motion system..." -ForegroundColor Cyan
Write-Host "   Press ENABLE button on GPIO panel, then press ENTER..."
Read-Host

$motion = omni-orch initialize-motion `
  --cert $Cert `
  --key $Key `
  --ca-cert $CaCert `
  --base-url $BaseUrl

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Motion initialization failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Motion initialized" -ForegroundColor Green
Write-Host $motion

# Verify
Write-Host "`n📋 Verifying system..." -ForegroundColor Cyan
$final = omni-orch get-device-state `
  --cert $Cert `
  --key $Key `
  --ca-cert $CaCert `
  --base-url $BaseUrl

Write-Host "✓ System ready!" -ForegroundColor Green
Write-Host $final
```

**Usage:**
```powershell
.\init-system.ps1
```

### Shutdown Script

```powershell
# shutdown-system.ps1
# Safe system shutdown

param(
    [string]$Cert = $env:OMNISCAN_CERT,
    [string]$Key = $env:OMNISCAN_KEY,
    [string]$CaCert = $env:OMNISCAN_CA,
    [string]$BaseUrl = $env:OMNISCAN_URL
)

Write-Host "🛑 Shutting down system..." -ForegroundColor Yellow

# Power off motion
Write-Host "Powering off motion..." -ForegroundColor Cyan
omni-orch power-off-motion `
  --cert $Cert `
  --key $Key `
  --ca-cert $CaCert `
  --base-url $BaseUrl
Write-Host "✓ Motion powered off" -ForegroundColor Green

# Power off detector
Write-Host "Powering off detector..." -ForegroundColor Cyan
omni-orch power-off-detector `
  --cert $Cert `
  --key $Key `
  --ca-cert $CaCert `
  --base-url $BaseUrl
Write-Host "✓ Detector powered off" -ForegroundColor Green

Write-Host "✓ System shutdown complete" -ForegroundColor Green
```

**Usage:**
```powershell
.\shutdown-system.ps1
```

---

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success | Operation completed successfully |
| 1 | General error | Check error message and fix issue |
| 2 | Invalid arguments | Verify command syntax and parameters |
| 3 | Connection error | Verify server address and certificates |
| 4 | Authorization error | Verify certificates are valid |
| 5 | Precondition not met | Check enable button, interlocks, etc. |

---

## Tips & Best Practices

### Before Initializing
- ✅ Always check `get-device-state` first
- ✅ Verify physical preconditions (key switch, beam block, cooling)
- ✅ Ensure certificates are in correct location
- ✅ Test ENABLE button on GPIO panel

### During Initialization
- ✅ Press ENABLE button just before running command
- ✅ Work quickly (20-second window)
- ✅ Don't interrupt the process
- ✅ Watch for error messages

### After Initialization
- ✅ Always verify with `get-device-state`
- ✅ Check temperatures are within range
- ✅ Confirm detector and motion statuses are IDLE
- ✅ Wait for any cooling cycle to complete before measurements

### Troubleshooting
- ✅ Use `get-device-state` to diagnose issues
- ✅ Check physical indicators (LEDs, cooling pump sound)
- ✅ Verify environment variables are set correctly
- ✅ Review certificate file paths and permissions

---

## Monitoring & Logging

### Log Device State Periodically
```powershell
# Check status every 5 minutes
while ($true) {
    $state = omni-orch get-device-state `
      --cert $env:OMNISCAN_CERT `
      --key $env:OMNISCAN_KEY `
      --ca-cert $env:OMNISCAN_CA `
      --base-url $env:OMNISCAN_URL
    
    Write-Host "$(Get-Date): Detector=$(($state | ConvertFrom-Json).detector.status), Motion=$(($state | ConvertFrom-Json).motion.status)"
    
    Start-Sleep -Seconds 300
}
```

### Log to File
```powershell
omni-orch get-device-state `
  --cert $env:OMNISCAN_CERT `
  --key $env:OMNISCAN_KEY `
  --ca-cert $env:OMNISCAN_CA `
  --base-url $env:OMNISCAN_URL | `
  Out-File -Append "device-state-$(Get-Date -Format 'yyyyMMdd').log"
```

---

## Related Documentation

- **Dashboard UI:** Dashboard has "Init Detector" and "Init Motion" buttons as UI alternative
- **Safety Spec:** Detailed safety requirements in safety documentation
- **Orchestrator API:** REST API alternative (uses same backend)
- **Hardware Server:** gRPC interface for low-level control

---

## Support

For issues or questions:
1. Check error message and consult Common Errors section
2. Verify all preconditions are met
3. Review certificate configuration
4. Check orchestrator server is running
5. Review system logs for details
