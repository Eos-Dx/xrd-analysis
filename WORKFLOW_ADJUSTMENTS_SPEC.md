# DIFRA Workflow Adjustments Specification

## Overview
This document specifies the required adjustments to the measurement workflow based on user requirements.

---

## 1. Sample ID Field Rename

### Current State
- UI field labeled: "file name"
- Located in zone measurements panel

### Required Change
- **Rename to**: "Sample ID"
- **Reasoning**: Clearer semantic meaning - this identifies the sample, not a file
- **Location**: Zone measurements UI field

### Implementation
```python
# In zone measurements UI setup:
# OLD: QLabel("file name:")
# NEW: QLabel("Sample ID:")
```

---

## 2. Session Replacement Workflow

### Current State
- User loads image → session created
- No handling of existing sessions

### Required Workflow

#### 2.1 Load Image Triggers Session Decision

When user clicks "Load Image":

```
1. Check if active session exists
   ↓
2. If YES: Show dialog
   ┌─────────────────────────────────────────────────┐
   │  Replace Current Session?                       │
   │                                                  │
   │  Current session: session_abc123.h5             │
   │  Sample ID: SAMPLE_001                          │
   │  Status: ⚠️  Not uploaded to cloud              │
   │  Points measured: 15/20                         │
   │                                                  │
   │  ⚠️  WARNING: Creating a new session will:      │
   │  • Lock the current session (if unlocked)       │
   │  • Archive the current session                  │
   │  • Mark it as 'not uploaded'                    │
   │                                                  │
   │  ⚠️  RECOMMENDATION:                             │
   │  It is strongly recommended to complete and     │
   │  upload the current session before starting     │
   │  a new one. Incomplete sessions may lose        │
   │  valuable measurement data.                     │
   │                                                  │
   │  Options:                                       │
   │  [ Continue Current Session ]  (default)        │
   │  [ Create New Session & Archive Current ]       │
   │  [ Mark Current as Error & Archive ]            │
   │  [ Cancel ]                                     │
   └─────────────────────────────────────────────────┘
   ↓
3. User selects action:
   
   a) Continue Current Session:
      - Keep existing session active
      - Allow adding new images to same session
      - Cancel load operation
   
   b) Create New Session & Archive Current:
      - Lock current session (if unlocked)
      - Add attribute: uploaded_to_cloud = False
      - Archive to: session_archive/<session_id>_<timestamp>/
      - Log: "Session archived (not uploaded)"
      - Create new session
      - Load new image
   
   c) Mark Current as Error & Archive:
      ┌─────────────────────────────────────────────┐
      │  Mark Session as Error                      │
      │                                             │
      │  Reason for error (optional):               │
      │  [____________________________________]     │
      │                                             │
      │  [ OK ]  [ Cancel ]                        │
      └─────────────────────────────────────────────┘
      - Lock current session (if unlocked)
      - Add attributes:
          created_by_error = True
          error_reason = "<user input or 'Not specified'>"
          uploaded_to_cloud = False
      - Archive to: session_archive/<session_id>_<timestamp>_ERROR/
      - Log: "Session marked as error and archived"
      - Create new session
      - Load new image
   
   d) Cancel:
      - No changes
      - Return to measurement view

4. If NO existing session:
   - Proceed directly to load image
   - Create new session
```

#### 2.2 Session Status Tracking

Add to session container attributes:
```python
uploaded_to_cloud: bool = False  # Set True after successful upload
created_by_error: bool = False   # Set True if marked as error
error_reason: str = ""           # User-provided reason if error
archived_timestamp: str = ""     # When archived
```

#### 2.3 Archive Structure

```
session_archive/
├── session_abc123_20260211_184500/
│   ├── session_abc123.h5
│   │   [attributes]
│   │   uploaded_to_cloud: False
│   │   created_by_error: False
│   │   archived_timestamp: "2026-02-11 18:45:00"
│   └── metadata.json  (optional summary)
│
└── session_def456_20260211_190000_ERROR/
    ├── session_def456.h5
    │   [attributes]
    │   uploaded_to_cloud: False
    │   created_by_error: True
    │   error_reason: "Wrong sample loaded"
    │   archived_timestamp: "2026-02-11 19:00:00"
    └── metadata.json
```

---

## 3. Image Data Storage

### Current Proposal
- Store as numpy array: `image_data: np.ndarray`

### Confirmed Approach
✅ **Approved**: Store image data as numpy arrays in HDF5

```python
# In /images/img_001/
image_data: np.ndarray  # Shape: (height, width) or (height, width, channels)
                        # Dtype: uint8, uint16, or float32
```

**Benefits:**
- Efficient compression in HDF5
- Native format for processing
- No external file dependencies

---

## 4. Pixel-to-MM Conversion from Sample Holder

### Current Approach
- Pixel-to-mm conversion provided as dict

### Clarification Required
**Conversion calculated from real size of sample holder zone**

### Proposed Implementation

```python
def calculate_pixel_to_mm(
    holder_zone_geometry_px: dict,
    holder_diameter_mm: float
) -> dict:
    """Calculate pixel-to-mm conversion from sample holder geometry.
    
    Args:
        holder_zone_geometry_px: Zone geometry in pixels
            For circle: {"center": [x, y], "radius": r}
            For rectangle: {"x": x1, "y": y1, "width": w, "height": h}
        holder_diameter_mm: Real physical diameter/size in mm
    
    Returns:
        Conversion dict with scale factors
    """
    if "radius" in holder_zone_geometry_px:
        # Circular holder
        radius_px = holder_zone_geometry_px["radius"]
        diameter_px = radius_px * 2
        scale = holder_diameter_mm / diameter_px  # mm per pixel
        
        return {
            "scale_x": scale,
            "scale_y": scale,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "method": "circular_holder",
            "holder_diameter_mm": holder_diameter_mm,
            "holder_diameter_px": diameter_px
        }
    
    elif "width" in holder_zone_geometry_px:
        # Rectangular holder - use width as reference
        width_px = holder_zone_geometry_px["width"]
        scale = holder_diameter_mm / width_px  # mm per pixel
        
        return {
            "scale_x": scale,
            "scale_y": scale,  # Assume square pixels
            "offset_x": 0.0,
            "offset_y": 0.0,
            "method": "rectangular_holder",
            "holder_width_mm": holder_diameter_mm,
            "holder_width_px": width_px
        }
```

### Usage in Session
```python
# When sample holder zone defined:
holder_zone = session_manager.add_zone(
    zone_index=1,
    geometry_px={"center": [500, 500], "radius": 200},
    shape="circle",
    zone_role="sample_holder",
    holder_diameter_mm=50.0  # Real physical size
)

# Automatically calculate conversion:
conversion = calculate_pixel_to_mm(
    holder_zone_geometry_px={"center": [500, 500], "radius": 200},
    holder_diameter_mm=50.0
)
# Result: scale = 50.0 / 400 = 0.125 mm/pixel

# Store in mapping:
session_manager.add_image_mapping(
    sample_holder_zone_id=holder_zone,
    pixel_to_mm_conversion=conversion,
    orientation="standard"
)
```

---

## 5. Raw Data Storage Reorganization

### Current Structure
```
/technical/tech_evt_001/det_001/
├── raw_signal (2D array)
├── raw_blob_txt (compressed)
└── raw_blob_dsc (compressed)
```

### Required Structure
```
/technical/tech_evt_001/det_001/
├── raw_signal (2D array)
└── blob/
    ├── raw_txt (compressed .txt file)
    └── raw_dsc (compressed .dsc file)
```

### Rationale
- **Clearer organization**: All raw/original files under `blob/`
- **Extensibility**: Easy to add more raw file types
- **Consistency**: Same structure in technical and session containers

### Apply to Both Containers

#### Technical Container
```
/technical/tech_evt_001/det_001/
├── raw_signal (2D array - processed)
└── blob/
    ├── raw_txt (original detector .txt)
    └── raw_dsc (original descriptor .dsc)
```

#### Session Container - Measurements
```
/measurements/pt_001/meas_000000003/det_001/
├── raw_signal (2D array - processed)
└── blob/  (optional - if raw files saved)
    ├── raw_txt
    └── raw_dsc
```

#### Session Container - Analytical Measurements
```
/analytical_measurements/ana_000000001/det_001/
├── raw_signal (2D array)
└── blob/  (optional)
    ├── raw_txt
    └── raw_dsc
```

### Implementation Changes

```python
# Old path:
tech_group.create_dataset("raw_blob_txt", data=compressed_txt)

# New path:
blob_group = tech_group.create_group("blob")
blob_group.create_dataset("raw_txt", data=compressed_txt)
blob_group.create_dataset("raw_dsc", data=compressed_dsc)
```

---

## 6. Updated Session Manager API

### New Methods

```python
class SessionManager:
    
    def check_and_handle_existing_session(
        self,
        action: str = "prompt"  # "prompt", "auto_archive", "error"
    ) -> bool:
        """Check for existing session and handle replacement.
        
        Args:
            action: How to handle existing session
                - "prompt": Show dialog to user
                - "auto_archive": Automatically archive current
                - "error": Require manual cleanup
        
        Returns:
            True if can proceed, False if user cancelled
        """
        pass
    
    def archive_current_session(
        self,
        mark_as_error: bool = False,
        error_reason: str = "",
        uploaded: bool = False
    ) -> Path:
        """Archive current session before creating new one.
        
        Args:
            mark_as_error: Mark session as created by error
            error_reason: Reason if error
            uploaded: Whether session was uploaded to cloud
        
        Returns:
            Path to archived session
        """
        pass
    
    def get_session_status(self) -> dict:
        """Get comprehensive session status.
        
        Returns:
            {
                "active": bool,
                "session_id": str,
                "sample_id": str,
                "total_points": int,
                "measured_points": int,
                "progress_percent": float,
                "uploaded_to_cloud": bool,
                "created_by_error": bool
            }
        """
        pass
```

---

## 7. Updated HDF5 Schema

### Session Container Attributes (Root Level)

```python
# Required attributes
sample_id: str           # From "Sample ID" field
session_id: str          # UUID
creation_timestamp: str
acquisition_date: str
operator_id: str
site_id: str
machine_name: str
beam_energy_keV: float

# Optional attributes
patient_id: str          # If applicable

# Status tracking (new)
uploaded_to_cloud: bool = False       # Updated after successful upload
created_by_error: bool = False        # True if session marked as error
error_reason: str = ""                # User-provided reason if error
archived_timestamp: str = ""          # When archived (if archived)
session_locked: bool = False          # Locked status
locked_by: str = ""                   # Operator who locked
locked_timestamp: str = ""            # When locked
```

### Updated Structure with blob/

```
session_<session_id>.h5
├── [attributes]
│   ├── sample_id: "SAMPLE_001"
│   ├── uploaded_to_cloud: False
│   ├── created_by_error: False
│   └── ...
│
├── /technical
│   └── /tech_evt_001
│       └── /det_001
│           ├── raw_signal
│           └── /blob
│               ├── raw_txt
│               └── raw_dsc
│
└── /measurements
    └── /pt_001
        └── /meas_000000003
            └── /det_001
                ├── raw_signal
                └── /blob  (optional)
                    ├── raw_txt
                    └── raw_dsc
```

---

## 8. Implementation Priority

### Phase 1: Core Changes (High Priority)
1. ⏳ Rename "file name" → "Sample ID" in UI (TODO)
2. ⏳ Implement session replacement workflow with dialog (TODO)
3. ⏳ Add session archiving with upload status (TODO)
4. ⏳ Add error session handling (TODO)

### Phase 2: Storage Changes (Medium Priority)
5. ✅ **Reorganize raw data to blob/ structure (COMPLETED)**
   - Technical containers: blob/ with raw_txt, raw_dsc datasets
   - Session measurements: blob/ group for optional raw files
   - Legacy technical containers: blob/ structure implemented
   - v0_1 schema updated: DATASET_BLOB replaces DATASET_RAW_FILES
   - All tests passing (session container tests verified)
6. ⏳ Update schema with new attributes (Partial - attributes defined in spec)
7. ⏳ Document pixel-to-mm calculation (TODO)

### Phase 3: Enhancement (Lower Priority)
8. Add session status dashboard
9. Implement session recovery tools
10. Add batch archive management

---

## 9. User Experience Flow

### Scenario 1: Normal Workflow
```
User loads image → No existing session → Session created → Measurements proceed
```

### Scenario 2: Replace Session (Complete Previous)
```
User loads image → Existing session found → Dialog shown → User chooses:
  "Continue Current Session" → User completes measurements → Upload → 
  Then load new image → New session created
```

### Scenario 3: Replace Session (Archive Incomplete)
```
User loads image → Existing session found → Dialog shown → User chooses:
  "Create New & Archive" → Warning shown → Confirm → 
  Previous session locked & archived (uploaded_to_cloud=False) →
  New session created → New image loaded
```

### Scenario 4: Error Session
```
User loads wrong sample → Realizes error → Loads correct image → Dialog shown →
  "Mark as Error & Archive" → Reason dialog → Enter reason →
  Session archived (created_by_error=True, error_reason="Wrong sample") →
  New session created → Correct workflow proceeds
```

---

## 10. Migration Plan

### Existing Containers
- Old containers remain compatible
- New attributes optional (backward compatible)
- No forced migration required

### New Features Active
- All new sessions include tracking attributes
- Archive workflow available immediately
- blob/ structure used for new data

---

## Summary of Changes

| Item | Change | Impact |
|------|--------|--------|
| Sample ID field | Rename from "file name" | UI clarity |
| Session replacement | Add dialog & workflow | Prevent data loss |
| Session archiving | Auto-archive with status | Better tracking |
| Error handling | Mark & archive errors | Data integrity |
| Image storage | Confirm numpy arrays | Implementation detail |
| Pixel-to-mm | Calculate from holder size | Accuracy |
| Raw data storage | Move to blob/ folder | Organization |
| Session attributes | Add upload/error tracking | Status visibility |

---

## Questions for User

1. **Archive Location**: Should archived sessions go to:
   - `session_archive/` (in measurement folder)
   - `difra/archive/sessions/` (centralized)

2. **Auto-Upload**: Should system attempt auto-upload before archiving?

3. **Session Limits**: Should there be a limit on incomplete sessions before warning?

4. **Recovery**: Should there be a "recover archived session" feature?

---

This specification provides a complete blueprint for implementing all requested workflow adjustments.
