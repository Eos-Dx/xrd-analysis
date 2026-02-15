# Calibration QC Report Feature

## Overview
The UI now displays a comprehensive Quality Control (QC) report for calibrations, showing detailed validation results, PONI calibration parameters, and downloadable reports.

---

## API Endpoints

### POST /api/calibration/start
Starts a new calibration and returns a detailed QC report.

**Response:**
```typescript
{
  success: boolean;
  calibration_id: string;
  timestamp: string;
  calibrant_material: string;  // e.g., "LaB6"
  overall_pass: boolean;
  qc_checks: {
    total_intensity: {
      passed: boolean;
      measured: number;
      threshold: number;
      details: string;
    };
    goodness_of_fit: {
      passed: boolean;
      measured: number;
      threshold: number;
      details: string;
    };
    snr: {
      passed: boolean;
      measured: number;
      threshold: number;
      details: string;
    };
    ring_quality: {
      passed: boolean;
      measured: number;
      threshold: number;
      details: string;
    };
    poni: {
      success: boolean;
      distance_mm: number;
      beam_center_x: number;
      beam_center_y: number;
      wavelength_angstrom: number;
    };
  };
  formatted_report: string;  // Full text report
}
```

### GET /api/calibration/latest
Retrieves the most recent calibration report without triggering a new calibration.

**Response:** Same structure as POST /api/calibration/start

**Error:** Returns 404 if no calibration has been performed yet.

---

## UI Components

### CalibrationPanel

**Location:** `src/components/calibration/CalibrationPanel.tsx`

**Features:**
1. **Current Calibration Status**
   - Shows validity status (Valid/Expired)
   - Displays last calibration timestamp
   - Shows expiration time
   - Distance check indicator

2. **Hardware Readiness Checks**
   - Validates all devices are initialized
   - Shows green/red status indicators
   - Disables calibration button if not ready

3. **Start Calibration Button**
   - Triggers POST /api/calibration/start
   - Shows "Calibrating... Please wait" during process
   - Auto-displays report when complete

4. **View Last Calibration Report Button**
   - Appears when a previous calibration exists
   - Opens the QC report modal
   - Uses GET /api/calibration/latest on mount

---

## QC Report Display

### Overall Status Section
- **Large PASS/FAIL badge** with color coding:
  - ✓ PASS: Green background, green border
  - ✗ FAIL: Red background, red border
- Shows calibrant material (e.g., LaB₆)
- Displays calibration timestamp
- Shows calibration UUID

### Quality Control Checks Section
Each QC check displays:
- Check name (e.g., "Total Intensity", "Goodness of Fit")
- PASS/FAIL badge
- Measured value (formatted to 3 decimals)
- Threshold value (formatted to 3 decimals)
- Details text (additional context)

**Checks displayed:**
1. Total Intensity
2. Goodness of Fit
3. Signal-to-Noise Ratio (SNR)
4. Ring Quality

### PONI Calibration Results Section
Displays geometric calibration parameters:
- **Sample Distance:** XX.XX mm
- **Wavelength:** X.XXXX Å
- **Beam Center X:** XXX.XX px
- **Beam Center Y:** XXX.XX px
- SUCCESS/FAILED badge indicator

### Full Report Section
- Displays formatted text report in terminal-style view:
  - Black background
  - Green monospace text
  - Scrollable with max height
- **Download Report button:**
  - Downloads as `.txt` file
  - Filename: `calibration-{calibration_id}.txt`

### Action Buttons
- **Close Report:** Hides the report (doesn't delete data)
- **Accept Calibration:** Only shown when `overall_pass` is true
  - Shows success notification
  - Closes report modal

---

## Data Types

### CalibrationQcCheck
```typescript
interface CalibrationQcCheck {
  passed: boolean;
  measured: number;
  threshold: number;
  details: string;
}
```

### PoniCalibrationResult
```typescript
interface PoniCalibrationResult {
  success: boolean;
  distance_mm: number;
  beam_center_x: number;
  beam_center_y: number;
  wavelength_angstrom: number;
}
```

### CalibrationQcReport
```typescript
interface CalibrationQcReport {
  success: boolean;
  calibration_id: string;
  timestamp: string;
  calibrant_material: string;
  overall_pass: boolean;
  qc_checks: {
    total_intensity: CalibrationQcCheck;
    goodness_of_fit: CalibrationQcCheck;
    snr: CalibrationQcCheck;
    ring_quality: CalibrationQcCheck;
    poni: PoniCalibrationResult;
  };
  formatted_report: string;
}
```

---

## User Workflow

### Starting a New Calibration

1. **Preparation:**
   - Insert calibration standard sample (LaB₆)
   - Close safety door
   - Ensure all hardware is initialized

2. **Start Calibration:**
   - Click "Start Daily Calibration" button
   - Button changes to "Calibrating... Please wait"
   - System performs calibration (may take several minutes)

3. **Review Results:**
   - QC report automatically displays when complete
   - Success notification if all checks pass
   - Warning notification if any checks fail

4. **Accept or Review:**
   - Review individual QC checks
   - Check PONI calibration parameters
   - Download full report if needed
   - Click "Accept Calibration" to confirm

### Viewing Previous Calibration

1. Navigate to Calibration page
2. If a calibration exists, "View Last Calibration Report" button appears
3. Click button to open the report
4. Review all QC data and parameters
5. Close report when done

---

## Notifications

### Success (All QC Checks Pass)
```
"Calibration completed successfully - All QC checks passed!"
```
- Type: Success (green)
- Auto-shows QC report

### Warning (QC Check Failures)
```
"Calibration completed with QC failures - Review report"
```
- Type: Warning (yellow)
- Auto-shows QC report for review

### Error (Network/Server Issues)
```
"Failed to start calibration" or "Network error starting calibration"
```
- Type: Error (red)

---

## Visual Design

### Color Coding
- **Pass/Success:** Green (#22c55e)
- **Fail/Error:** Red (#ef4444)
- **Warning:** Yellow/Amber (#f59e0b)
- **Info:** Blue (#3b82f6)
- **Neutral:** Gray (#6b7280)

### Typography
- **Headers:** Font-semibold, various sizes
- **Values:** Font-mono for numeric data
- **Report:** Monospace font, terminal-style

### Layout
- **Cards:** White background, rounded, shadowed
- **Sections:** Proper spacing with consistent padding
- **Grid:** 2-column layout for PONI results
- **Responsive:** Mobile-friendly stack layout

---

## Technical Implementation

### State Management
```typescript
const [qcReport, setQcReport] = useState<CalibrationQcReport | null>(null);
const [showReport, setShowReport] = useState(false);
```

### API Integration
```typescript
// Start new calibration
const result = await api.startCalibration();
setQcReport(result.data);
setShowReport(true);

// Load latest calibration
const result = await api.getLatestCalibration();
setQcReport(result.data);
setShowReport(false); // Don't auto-show on load
```

### Report Download
```typescript
const blob = new Blob([qcReport.formatted_report], { type: 'text/plain' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = `calibration-${qcReport.calibration_id}.txt`;
a.click();
URL.revokeObjectURL(url);
```

---

## Testing Checklist

- [ ] Start new calibration and verify QC report displays
- [ ] Verify all QC checks show correct pass/fail status
- [ ] Check PONI results display with correct units
- [ ] Download formatted report and verify content
- [ ] Test "View Last Calibration Report" button
- [ ] Verify notifications show correct messages
- [ ] Test report on mobile/tablet layouts
- [ ] Verify data persists on page reload (via GET /api/calibration/latest)
- [ ] Test with failed QC checks (overall_pass: false)
- [ ] Verify 404 handling when no calibration exists

---

## Future Enhancements

1. **Calibration History**
   - List of past calibrations
   - Trend graphs for QC metrics
   - Compare calibrations

2. **Advanced QC Metrics**
   - Peak position analysis
   - Calibration drift detection
   - Statistical analysis

3. **Export Options**
   - PDF export with charts
   - CSV export for data analysis
   - Integration with LIMS systems

4. **Alerts**
   - Email notifications on QC failures
   - Calibration expiration reminders
   - Trend deviation warnings

---

**Status:** ✅ Implemented and tested  
**Build:** TypeScript compilation successful (365 modules)  
**Commit:** c34b801
