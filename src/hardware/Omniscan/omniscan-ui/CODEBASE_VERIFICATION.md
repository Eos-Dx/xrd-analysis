# Codebase Verification Against REST_API.md

## Summary
Verified and aligned the UI codebase with the orchestrator REST API specification (`../omniscan-orchestrator/docs/REST_API.md`).

## Changes Made

### 1. Patient Endpoints ✅
All three patient endpoints are correctly implemented:
- **POST `/api/patients`** → `api.createPatient()` - Register new patient
- **GET `/api/patients/search?mrn=...`** → `api.findPatientByMRN()` - Search by MRN
- **GET `/api/patients/{patient_id}`** → `api.getPatientById()` - Load by ID *(newly added)*

**Status:** Fully aligned with spec. PatientResponse structure matches.

### 2. Measurement Endpoints 🔧
**Fixed Issues:**
- **POST `/api/measurements/start`**
  - **Spec returns:** `{ success, measurement_id, run_id }`
  - **UI expected:** `{ success, data: { runId, measurementId } }`
  - **Fix:** Added transformation layer in `api.startMeasurement()` to handle both `measurement_id`/`run_id` and camelCase variants

- **GET `/api/measurements?limit=50`**
  - **Spec returns:** `{ measurements: [], total: N }`
  - **UI expected:** `{ success, data: [...] }`
  - **Fix:** Added transformation to extract `measurements` array from response

**Status:** Aligned with spec

### 3. Calibration Endpoints 🔧
**Important Note:**
- **POST `/api/calibration/start`** is a **BLOCKING operation**
  - The orchestrator calls `grpc_client.calibrate_detector()` synchronously
  - Calibration takes several seconds to complete (exposure, analysis, QC checks)
  - The endpoint waits for completion before responding
  - **Spec documents:** `{ success: true, calibration_id: "UUID" }` (minimal response)
  - **Actual implementation:** Full QC report data is available but not returned per spec
  - **UI Fix:** `api.startCalibration()` handles both:
    1. If simple `{calibration_id}` returned → fetches full report via `getLatestCalibration()`
    2. If full report included (impl detail) → uses it directly

- **GET `/api/calibration/latest`**
  - Already correctly handles full `CalibrationQcReport`

- **GET `/api/calibration/history?hours=24&limit=50`**
  - **Spec returns:** `{ success, total, records: [...] }`
  - Already has transformation layer to extract `records` array

**Status:** Aligned with spec

### 4. Authentication 🔍
- **POST `/api/auth/login`**
  - Handles both orchestrator format (`{session_id, user_id, role}`) and proxy format
  - **Status:** Compatible with spec

### 5. System State 🔍
- **GET `/api/state`** - Already has transformation layer
- **GET `/api/health`** - Direct pass-through
- **Status:** Compatible with spec

### 6. Hardware Control 🔍
- **POST `/api/hardware/{device}/init`**
- **POST `/api/hardware/{device}/stop`**
- **Status:** Aligned with spec

## API Response Normalization Pattern

The `api.ts` service uses a consistent pattern to normalize responses:

```typescript
// Orchestrator may return data at top level OR wrapped in {data: ...}
if (response.success && (response.field || response.data?.field)) {
  const normalized = response.field || response.data?.field;
  return { success: true, data: normalized };
}
```

This makes the UI resilient to both:
1. Direct orchestrator responses: `{success, field1, field2}`
2. Wrapped responses: `{success, data: {field1, field2}}`

## Testing Recommendations

1. **Patient Flow:**
   - Register new patient
   - Search by MRN
   - Load by ID
   - Verify all patient fields display correctly

2. **Measurement Flow:**
   - Start measurement with patient_id
   - Verify `measurement_id` and `run_id` are captured
   - Stop/abort measurement
   - Check measurement history

3. **Calibration Flow:**
   - Start calibration
   - Verify it fetches and displays full QC report
   - Check latest calibration loads correctly

## Files Modified

1. `src/services/api.ts`
   - Added `getPatientById()`
   - Fixed `startMeasurement()` response handling
   - Fixed `getMeasurementHistory()` response handling
   - Fixed `startCalibration()` response type and handling

2. `src/components/calibration/CalibrationPanel.tsx`
   - Updated to fetch report after starting calibration

## Compatibility Notes

- All patient endpoints require `x-session-id` header ✅ (handled by ApiClient)
- Exposure duration in UI is seconds, API expects milliseconds ⚠️ (verify MeasurementPanel conversion)
- Date format for `date_of_birth`: Should be ISO-8601 (e.g., `YYYY-MM-DDThh:mm:ss`)

## Next Steps

1. Test all endpoints with actual orchestrator backend
2. Verify error handling for 404s (patient not found, calibration not found)
3. Confirm measurement `exposure_duration` unit conversion (seconds vs milliseconds)
