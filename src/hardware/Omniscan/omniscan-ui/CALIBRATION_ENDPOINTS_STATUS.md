# Calibration Endpoints - Implementation Status

## Current Status

### ✅ UI Implementation (Complete)
The omniscan-ui is **fully implemented** and ready to consume calibration QC reports.

**Commits:**
- `c34b801` - Add calibration QC report display and GET /api/calibration/latest support
- `b83cb30` - Add resilient error handling for missing calibration endpoint

**Features:**
- Detailed QC report display with all checks
- PONI calibration results visualization
- Formatted report text with download
- Graceful error handling for missing endpoints
- Works even if orchestrator endpoints are not yet implemented

---

## Required Orchestrator Endpoints

### 1. POST /api/calibration/start ⚠️ Needs Update

**Current Status:** Endpoint exists but needs to return detailed QC report

**Expected Response Structure:**
```json
{
  "success": true,
  "calibration_id": "uuid-string",
  "timestamp": "2025-10-28T15:00:00Z",
  "calibrant_material": "LaB6",
  "overall_pass": true,
  "qc_checks": {
    "total_intensity": {
      "passed": true,
      "measured": 12500.5,
      "threshold": 10000.0,
      "details": "Intensity within acceptable range"
    },
    "goodness_of_fit": {
      "passed": true,
      "measured": 0.95,
      "threshold": 0.90,
      "details": "Excellent fit quality"
    },
    "snr": {
      "passed": true,
      "measured": 25.3,
      "threshold": 20.0,
      "details": "Signal-to-noise ratio acceptable"
    },
    "ring_quality": {
      "passed": true,
      "measured": 0.92,
      "threshold": 0.85,
      "details": "Ring definition good"
    },
    "poni": {
      "success": true,
      "distance_mm": 150.5,
      "beam_center_x": 512.3,
      "beam_center_y": 511.8,
      "wavelength_angstrom": 1.5406
    }
  },
  "formatted_report": "=== Calibration QC Report ===\nCalibration ID: uuid-string\n..."
}
```

**Implementation Notes:**
- Call hardware server calibration routine
- Collect all QC check results
- Run PONI calibration
- Generate formatted text report
- Return complete structure to UI

---

### 2. GET /api/calibration/latest ❌ Not Implemented

**Status:** Endpoint does not exist yet

**Purpose:** Allow UI to retrieve the most recent calibration report without triggering a new calibration.

**Expected Response:**
- Same structure as POST /api/calibration/start
- Returns 404 if no calibration has been performed yet
- Returns 200 with calibration data if available

**Expected Behavior:**
```json
// Success case (200)
{
  "success": true,
  "calibration_id": "...",
  "timestamp": "...",
  // ... same structure as POST response
}

// No calibration case (404)
{
  "success": false,
  "error": "No calibration found"
}
```

**Implementation Notes:**
- Store calibration results in database with timestamp
- Query most recent calibration by timestamp
- Return full QC report structure
- Handle case where no calibration exists

---

## Error Handling

### UI Behavior

**When Endpoint is Missing (503/ECONNRESET):**
- ✅ UI logs informative message to console
- ✅ No error displayed to user
- ✅ UI continues to function normally
- ✅ "Start Daily Calibration" button still works

**When No Calibration Exists (404):**
- ✅ UI logs message: "No previous calibration found"
- ✅ "View Last Calibration Report" button hidden
- ✅ User can start new calibration

**When Calibration Fails QC (overall_pass: false):**
- ✅ UI shows warning notification
- ✅ QC report auto-displays with failed checks highlighted
- ✅ "Accept Calibration" button hidden
- ✅ User can review and retry

---

## Current Console Output

```
Proxying GET /api/calibration/latest to http://localhost:8081/api/calibration/latest
Proxy error for /api/calibration/latest: fetch failed
Full error: TypeError: fetch failed
  [cause]: Error: read ECONNRESET
```

**This is expected** when the orchestrator doesn't have the endpoint yet. The UI handles it gracefully.

---

## Testing After Orchestrator Implementation

### Test Cases

1. **First Time Setup (No Previous Calibration)**
   - [ ] Navigate to Calibration page
   - [ ] Verify no ECONNRESET errors in console
   - [ ] Verify "View Last Calibration Report" button does not appear
   - [ ] Start new calibration
   - [ ] Verify QC report displays after completion

2. **Subsequent Visits (With Previous Calibration)**
   - [ ] Navigate away and return to Calibration page
   - [ ] Verify GET /api/calibration/latest is called
   - [ ] Verify "View Last Calibration Report" button appears
   - [ ] Click button to view previous report
   - [ ] Verify all QC data displays correctly

3. **Failed QC Checks**
   - [ ] Create calibration with failed checks
   - [ ] Verify overall_pass: false
   - [ ] Verify warning notification appears
   - [ ] Verify failed checks highlighted in red
   - [ ] Verify "Accept Calibration" button hidden

4. **Successful QC**
   - [ ] Create calibration with all checks passing
   - [ ] Verify success notification
   - [ ] Verify all checks show green PASS badges
   - [ ] Verify "Accept Calibration" button appears

5. **PONI Calibration Results**
   - [ ] Verify distance displays in mm
   - [ ] Verify wavelength displays in Å
   - [ ] Verify beam center coordinates in pixels
   - [ ] Verify SUCCESS/FAILED badge

6. **Report Download**
   - [ ] Click "Download Report" button
   - [ ] Verify file downloads as `calibration-{id}.txt`
   - [ ] Verify formatted_report content is complete

---

## Integration Steps for Orchestrator

### Step 1: Update POST /api/calibration/start
1. Import new response structure types
2. Call hardware server calibration
3. Collect QC check results:
   - Total intensity from detector
   - Goodness of fit from pyFAI
   - SNR from image analysis
   - Ring quality metric
4. Run PONI calibration with pyFAI
5. Generate formatted text report
6. Return CalibrationQcReport structure

### Step 2: Implement GET /api/calibration/latest
1. Add database table for calibrations:
   ```sql
   CREATE TABLE calibrations (
     id TEXT PRIMARY KEY,
     timestamp TIMESTAMP,
     calibrant_material TEXT,
     overall_pass BOOLEAN,
     qc_checks_json TEXT,
     formatted_report TEXT
   );
   ```
2. Store calibration results after POST
3. Implement GET endpoint:
   ```python
   @app.get("/api/calibration/latest")
   async def get_latest_calibration():
       result = db.query(
           "SELECT * FROM calibrations ORDER BY timestamp DESC LIMIT 1"
       )
       if not result:
           return JSONResponse(
               status_code=404,
               content={"success": False, "error": "No calibration found"}
           )
       return result
   ```

### Step 3: Test Integration
1. Run UI against updated orchestrator
2. Verify GET /api/calibration/latest works
3. Verify POST /api/calibration/start returns full report
4. Test all UI features listed above

---

## Data Flow

```
┌─────────────┐
│    UI       │
└──────┬──────┘
       │ POST /api/calibration/start
       ▼
┌─────────────┐
│ Orchestrator│
└──────┬──────┘
       │ gRPC: CalibrateDetector
       ▼
┌─────────────┐
│  Hardware   │
│   Server    │
└──────┬──────┘
       │ Calibration data
       ▼
┌─────────────┐
│ Orchestrator│ Process QC checks
│             │ Run PONI calibration
│             │ Generate report
└──────┬──────┘
       │ CalibrationQcReport
       ▼
┌─────────────┐
│    UI       │ Display QC report
└─────────────┘
       │
       │ GET /api/calibration/latest
       ▼
┌─────────────┐
│ Orchestrator│ Query database
└──────┬──────┘
       │ Last calibration
       ▼
┌─────────────┐
│    UI       │ Show previous report
└─────────────┘
```

---

## Summary

**UI Status:** ✅ **READY** - Fully implemented with resilient error handling

**Orchestrator Status:** ⚠️ **NEEDS IMPLEMENTATION**
- Update POST /api/calibration/start to return full QC report
- Implement GET /api/calibration/latest endpoint
- Store calibration results in database

**Deployment:** UI can be deployed now and will work with partial orchestrator implementation. It will gracefully handle missing endpoints until they're ready.

---

**Last Updated:** 2025-10-28  
**Commits:**
- b83cb30 - Error handling
- c34b801 - QC report display
- 561bef1 - Type alignment
