# Calibration System

⚠️ **QC Implementation Status**: Quality control framework implemented with stub validation. Image analysis algorithms pending integration. NOT VALIDATED for clinical use.

Comprehensive documentation for the omniscan-hw-server calibration system, including quality control, safety requirements, and implementation details.

---

## Table of Contents

1. [Overview](#overview)
2. [Safety Requirements](#safety-requirements)
3. [Quality Control System](#quality-control-system)
4. [QC Report Structure](#qc-report-structure)
5. [Background Measurements](#background-measurements)
6. [Implementation Status](#implementation-status)

---

## Overview

The calibration system validates detector geometry through a 5-stage quality control cascade. Two types of measurements are supported:

1. **Calibration Measurement** - Diffraction pattern from calibration standard (requires X-ray beam OPEN)
2. **Background Measurement** - Dark frame acquisition (requires X-ray beam CLOSED)

### Calibration Requirements
- **Frequency**: Daily (24-hour validity period)
- **Enforcement**: System transitions to LOCKED state when calibration expires
- **Standards Supported**: LaB6, CeO2, Si, Al2O3
- **Automatic Validation**: 5-stage QC cascade with acceptance criteria

---

## Safety Requirements

### Key Safety Components

#### Key Switch
- General system activation switch
- Must be ON for all operations
- Physical confirmation of operator presence

#### Activation Button
- Opens/closes X-ray beam
- 20-second timeout window
- Required for non-safe operations

#### Radiation Safety Flag
- `true` = Beam CLOSED (safe for personnel)
- `false` = Beam OPEN (X-rays active)

### Calibration Measurement Safety

**Location**: `src/grpc/services.rs` lines 406-437

**Required Conditions**:
- ✅ Key switch: ON
- ✅ Radiation safety: FALSE (beam OPEN)
- ✅ Emergency stop: Not pressed
- ✅ Door: Closed
- ✅ Cooling: OK
- ✅ Power: OK

#### Calibration Bootstrap Exception

**Special Case**: Calibration exposure is the ONLY operation that may proceed without a prior valid calibration.

**Rationale**: 
- Normal measurements require valid calibration (within 24 hours)
- Calibration itself IS a measurement operation
- Therefore, the first calibration (or after expiry) must bypass the "valid calibration required" check
- This establishes the initial calibration baseline

**Authority**: Hardware Server enforces this exception - calibration requests are accepted even when no valid calibration exists.

**Safety Checks**:
```rust
// Check 1: Key switch must be ON
if !key_switch_on {
    return Err("Key switch must be ON to activate the system");
}

// Check 2: X-ray beam must be OPEN (radiation_safe = false)
if interlocks.radiation_safe {
    return Err("X-ray beam is closed. Click 'Activation' button to open the beam.");
}

// Check 3: Other interlocks must be safe
if !interlocks.emergency_stop || !interlocks.door_closed 
   || !interlocks.cooling_ok || !interlocks.power_ok {
    return Err("Safety interlocks not satisfied");
}
```

**Error Messages**:
- Beam closed: `"Calibration blocked: X-ray beam is closed. Click 'Activation' button to open the beam."`
- Key switch off: `"Calibration blocked: Key switch must be ON to activate the system"`

---

## Quality Control System

### Implementation Status

✅ **Framework Implemented** - All 5 QC stages implemented as stubs  
⚠️ **Integration Pending** - Actual image analysis algorithms need integration

### 5-Stage QC Cascade

⚠️ **All stages below are DESIGN SPECIFICATIONS. Current implementation uses stubs that auto-pass.**

#### Stage 1: Total Intensity Validation

**Purpose**: Ensures calibration image has sufficient signal

**Current Status**: STUB - Returns passing result (no actual validation)

**Implementation Needed**:
- Sum all pixel intensities in the image
- Compare against minimum threshold (e.g., 1e6 counts)
- Account for exposure time normalization

**Acceptance Criteria**:
- Measured: Total counts across entire image
- Threshold: 1.0e6 counts (typical)
- Pass: `measured_value > threshold`

**Example Output**:
```
Total intensity 5.20e6 counts exceeds threshold 1.00e6
```

**Failure Guidance**:
- Increase exposure time
- Check X-ray source power
- Verify calibrant is in beam path

---

#### Stage 2: Goodness of Fit Validation

**Purpose**: Measures how well observed rings match expected pattern

**Current Status**: STUB - Returns passing result

**Implementation Needed**:
- Detect ring positions using peak finding
- Fit ellipses to detected rings
- Compare fitted positions to theoretical positions for calibrant
- Calculate chi-square or R-squared goodness of fit metric

**Acceptance Criteria**:
- Measured: Chi-square or R² metric (0-1 scale)
- Threshold: 0.85 (typical)
- Pass: `measured_value > threshold`

**Example Output**:
```
Ring fitting goodness 0.950 exceeds threshold 0.850
```

**Failure Guidance**:
- Check calibrant material (LaB6 vs Si vs CeO2)
- Verify calibrant is not contaminated
- Check detector alignment

---

#### Stage 3: Signal-to-Noise Ratio (SNR) Validation

**Purpose**: Ensures adequate image quality for accurate calibration

**Current Status**: STUB - Returns passing result

**Implementation Needed**:
- Calculate mean signal intensity in ring regions
- Calculate standard deviation of background regions
- SNR = mean_signal / std_background
- Typical threshold: SNR > 10 for good calibration

**Acceptance Criteria**:
- Measured: SNR = mean_signal / std_background
- Threshold: 10.0 (typical)
- Pass: `measured_value > threshold`

**Example Output**:
```
SNR 25.4 exceeds threshold 10.0
```

**Failure Guidance**:
- Increase exposure time
- Reduce detector noise (cooling, dark current)
- Check for background sources

---

#### Stage 4: Ring Quality Validation

**Purpose**: Validates that calibration rings are complete and well-defined

**Current Status**: STUB - Returns passing result

**Implementation Needed**:
- Check ring completeness (% of ring visible)
- Check ring sharpness/width (FWHM - Full Width Half Maximum)
- Verify expected number of rings for calibrant
- Check for ring distortions or artifacts
- Quality score: 0-1, where 1 is perfect rings

**Acceptance Criteria**:
- Measured: Quality score (0-1) based on completeness, count, sharpness
- Threshold: 0.80 (typical)
- Pass: `measured_value > threshold`

**Example Output**:
```
Ring quality score 0.92 exceeds threshold 0.80. All expected rings detected.
```

**Failure Guidance**:
- Check for beam clipping (apertures, sample holder)
- Verify calibrant is polycrystalline (not single crystal)
- Check for preferred orientation

---

#### Stage 5: PONI File Generation

**Purpose**: Extracts geometric calibration parameters from calibration image

**Current Status**: STUB - Returns simulated PONI parameters

**Implementation Needed**:
- Integrate with pyFAI or equivalent calibration library
- Use detected ring positions to calculate geometry
- Determine beam center (PONI1, PONI2)
- Calculate sample-to-detector distance
- Determine detector tilt angles (rot1, rot2, rot3)
- Generate PONI file format compatible with pyFAI

**Results**:
- **distance_mm**: Sample-to-detector distance (e.g., 100.0 mm)
- **beam_center**: Pixel coordinates of direct beam (e.g., 512.5, 512.5)
- **rotations**: Detector tilt angles (usually ~0 for aligned systems)
- **pixel_size**: Detector pixel physical size
- **wavelength**: X-ray wavelength from source

**Example Output**:
```
PONI file calculated - Distance: 100.00 mm, Center: (512.5, 512.5) pixels

# PONI file generated by OMNIScan calibration
# Calibrant: LaB6
# Date: 2025-10-28T09:15:00Z
Distance: 0.100000
Poni1: 0.038438
Poni2: 0.038438
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1.54e-10
PixelSize1: 75e-6
PixelSize2: 75e-6
```

---

## QC Report Structure

### gRPC Messages

#### CalibrateDetectorResponse
```protobuf
message CalibrateDetectorResponse {
  bool success = 1;                             // Overall success status
  optional CalibrationQcReport qc_report = 2;   // Detailed QC report (if successful)
  string error_message = 3;                     // Error message (if failed)
}
```

#### CalibrationQcReport
```protobuf
message CalibrationQcReport {
  string calibrant_material = 1;              // "LaB6", "Si", "CeO2", etc.
  bool overall_pass = 2;                      // True if ALL checks passed
  google.protobuf.Timestamp timestamp = 3;    // When calibration was performed
  
  // Individual QC checks
  QcCheckResult total_intensity_check = 4;
  QcCheckResult goodness_check = 5;
  QcCheckResult snr_check = 6;
  QcCheckResult ring_quality_check = 7;
  PoniCalculationResult poni_result = 8;
  
  // Summary
  string formatted_report = 9;                // Human-readable text report
}
```

#### QcCheckResult
```protobuf
message QcCheckResult {
  bool passed = 1;             // Did this check pass?
  string check_name = 2;       // Check name
  double measured_value = 3;   // Actual measured value
  double threshold = 4;        // Required threshold
  string details = 5;          // Human-readable explanation
}
```

### Operator Report Format

```
═══════════════════════════════════════════════
  CALIBRATION QUALITY CONTROL REPORT
  Calibrant: LaB6
═══════════════════════════════════════════════

Overall Status: ✅ PASS

Individual Checks:
  1. Total Intensity - PASS
     Total intensity 5.20e6 counts exceeds threshold 1.00e6

  2. Goodness of Fit - PASS
     Ring fitting goodness 0.950 exceeds threshold 0.850

  3. Signal-to-Noise Ratio - PASS
     SNR 25.4 exceeds threshold 10.0

  4. Ring Quality - PASS
     Ring quality score 0.92 exceeds threshold 0.80. All expected rings detected.

  5. PONI File Generation - SUCCESS
     Distance: 100.00 mm
     Beam Center: (512.5, 512.5) pixels

═══════════════════════════════════════════════
```

---

## Background Measurements

### Purpose

Background measurements (dark frames) capture detector noise without X-rays. Required for:
- Dark current correction
- Noise characterization
- Image quality improvement

### Safety Requirements

**Location**: `src/grpc/services.rs` lines 583-693

**Required Conditions**:
- ✅ Key switch: ON
- ✅ Radiation safety: TRUE (beam CLOSED)
- ✅ Emergency stop: Not pressed
- ✅ Door: Closed
- ✅ Cooling: OK
- ✅ Power: OK

**Safety Checks**:
```rust
// Check 1: Key switch must be ON
if !key_switch_on {
    return Err("Key switch must be ON to activate the system");
}

// Check 2: X-ray beam must be CLOSED (radiation_safe = true)
if !interlocks.radiation_safe {
    return Err("X-ray beam is open. Close the beam before taking dark frames.");
}

// Check 3: Other interlocks must be safe
if !interlocks.emergency_stop || !interlocks.door_closed 
   || !interlocks.cooling_ok || !interlocks.power_ok {
    return Err("Safety interlocks not satisfied");
}
```

**Error Messages**:
- Beam open: `"Background measurement blocked: X-ray beam is open. Close the beam before taking dark frames."`
- Key switch off: `"Background measurement blocked: Key switch must be ON to activate the system"`

### Proto Definition

```protobuf
message StartBackgroundMeasurementRequest {
  CommandContext ctx = 1;
  uint32 exposure_time_ms = 2;   // Exposure time for dark frame
}

message BackgroundMeasurementResponse {
  bool success = 1;
  string error_message = 2;
  optional ExposureResult result = 3;
}
```

### Service Extension

```protobuf
service Acquisition {
  // ... existing methods ...
  rpc StartBackgroundMeasurement(StartBackgroundMeasurementRequest) 
      returns (BackgroundMeasurementResponse);
}
```

---

## Testing Scenarios

### Test 1: Calibration with Beam Closed (Should FAIL)
```
1. Turn key switch ON
2. Keep radiation_safe = true (beam closed)
3. Call calibrate_detector
Expected: Error "X-ray beam is closed. Click 'Activation' button to open the beam."
```

### Test 2: Calibration with Beam Open (Should SUCCEED)
```
1. Turn key switch ON
2. Click activation button
3. Set radiation_safe = false (beam open)
4. Call calibrate_detector
Expected: Success, calibration proceeds
```

### Test 3: Background with Beam Open (Should FAIL)
```
1. Turn key switch ON
2. Click activation button
3. Set radiation_safe = false (beam open)
4. Call start_background_measurement
Expected: Error "X-ray beam is open. Close the beam before taking dark frames."
```

### Test 4: Background with Beam Closed (Should SUCCEED)
```
1. Turn key switch ON
2. Keep radiation_safe = true (beam closed)
3. Call start_background_measurement
Expected: Success, dark frame captured
```

### Test 5: Key Switch Off (Both Should FAIL)
```
1. Keep key switch OFF
2. Try calibrate_detector
Expected: Error "Key switch must be ON to activate the system"
3. Try start_background_measurement
Expected: Error "Key switch must be ON to activate the system"
```

---

## Implementation Status

### Completed
- ✅ Calibration manager with 24-hour enforcement
- ✅ 5-stage QC cascade framework (stubs)
- ✅ Safety checks for calibration measurement
- ✅ Safety checks for background measurement
- ✅ Proto definitions for both measurement types
- ✅ QC report structure and generation
- ✅ Operator-friendly error messages

### Pending
- ⏳ Actual image analysis integration (pyFAI or native Rust)
- ⏳ Ring detection algorithms
- ⏳ Calibrant material database with d-spacings
- ⏳ Visual QC reports with ring overlays
- ⏳ Calibration trending and historical analysis

---

## Usage Example

```rust
use omniscan_hw_server::calibration_qc::{CalibrationQualityController, CalibrationImageData};

// Create QC controller for LaB6 calibrant
let qc = CalibrationQualityController::new("LaB6".to_string());

// Prepare calibration image data
let image_data = CalibrationImageData {
    raw_data: detector_image_bytes,
    width: 1024,
    height: 1024,
    exposure_time_ms: 1000,
    timestamp: chrono::Utc::now(),
};

// Run complete QC cascade
let results = qc.run_quality_control(&image_data).await?;

// Check overall pass/fail
if results.overall_pass {
    println!("✅ Calibration QC PASSED");
    
    // Generate PONI file
    if let Some(poni_content) = results.poni_calculation.poni_file_content {
        std::fs::write("calibration.poni", poni_content)?;
    }
} else {
    println!("❌ Calibration QC FAILED");
    
    // Generate detailed report for operator
    let report = qc.generate_report(&results);
    println!("{}", report);
}
```

---

## Build Requirements

After modifying proto files, regenerate Rust code:

```powershell
cd omniscan-hw-server
cargo build
```

The build process automatically regenerates proto files via `build.rs`.

---

## Related Files

- `src/calibration.rs` - Calibration management system
- `src/calibration_qc.rs` - Quality control cascade
- `src/grpc/services.rs` - Calibration and background measurement services
- `proto/hub/v1/hub.proto` - Protocol buffer definitions
- `src/safety/mod.rs` - Safety state machine integration
- `src/devices/gpio/demo.rs` - GPIO demo implementation
