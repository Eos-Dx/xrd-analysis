# Operator and Distance Selection Implementation

## Overview

Implemented comprehensive operator management and distance selection for DIFRA HDF5 container system with three key integration points:

1. **Startup**: Operator selection dialog
2. **Technical Container**: Operator + Distance selection when generating H5
3. **Session Container**: Operator + Distance selection with smart defaults

## Components Created

### 1. Operator Management System (`operator_manager.py`)

**OperatorManager** class:
- JSON-based storage: `src/hardware/difra/resources/config/operators.json`
- Stores operator information:
  - Operator ID (unique identifier)
  - Name and Surname
  - Email address
  - Phone (optional)
  - Institution (optional)
- Current operator tracking
- Auto-creates default operator if not exists
- Cloud-ready: JSON format allows easy sync when cloud integration added

**OperatorSelectionDialog**:
- Shows on DIFRA startup
- Displays all operators in dropdown
- Shows operator details (name, email, institution)
- "Create New Operator..." button
- Pre-selects last used operator

**NewOperatorDialog**:
- Add new operators with validation
- Required fields: ID, Name, Surname, Email
- Optional fields: Phone, Institution
- Checks for duplicate IDs

### 2. Technical Container Dialog (`technical_container_dialog.py`)

**TechnicalContainerDialog**:
- Shown when clicking "Generate H5" in Technical Measurements tab
- **Distance Selection**:
  - Shows PONI distance if available
  - Validates ±5% tolerance against PONI
  - Allows override with warning
  - Defaults to PONI distance or 17.0 cm
- **Operator Selection**:
  - Dropdown of all operators
  - Pre-selects current operator
  - "Add New Operator..." button inline
  - Shows operator details
- Stores operator info in technical container metadata

### 3. Session Container Dialog (`session_mixin.py`)

**Updated NewSessionDialog**:
- **Sample ID**: Required
- **Distance**: Required, with tooltip explaining it must match technical container
- **Operator Selection**:
  - Dropdown of all operators
  - Pre-selects current operator (smart default - usually same as technical)
  - "Add New Operator..." button inline
  - Shows operator details
  - Different operators allowed for technical vs session
- **Beam Energy**: Read from global config (not user input)

### 4. Session Mixin Updates (`session_mixin.py`)

**init_session_manager()**:
- Initializes OperatorManager
- Shows OperatorSelectionDialog on startup
- Passes current operator to SessionManager config
- Handles cancellation gracefully

**show_operator_selection_dialog()**:
- Called once on DIFRA startup
- Sets current operator for session
- Warns if cancelled (uses default)

## Workflow

### Startup Workflow

```
1. DIFRA starts
2. OperatorManager loads operators.json
3. OperatorSelectionDialog shows:
   - List of operators
   - Pre-select last used
   - Option to create new
4. User selects operator
5. Current operator stored for future defaults
```

### Technical Container Generation Workflow

```
1. User fills auxiliary measurements table
2. User clicks "Generate H5"
3. System reads PONI distance (if available)
4. TechnicalContainerDialog shows:
   - PONI distance displayed
   - Distance input (defaulted to PONI)
   - Operator selection (defaulted to current)
   - Option to add new operator
5. Validates:
   - Distance is number
   - Distance within ±5% of PONI (warning if not)
   - Operator selected
6. Creates technical container with:
   - Distance metadata
   - Operator ID metadata
7. Container stored with operator info
```

### Session Container Creation Workflow

```
1. User clicks "New Session..."
2. NewSessionDialog shows:
   - Sample ID input
   - Distance input (smart default from technical if available)
   - Operator selection (defaulted to current)
   - Option to add new operator
3. User can:
   - Keep current operator (common case)
   - Select different operator
   - Add new operator inline
4. Creates session container with:
   - Sample ID
   - Distance (must match technical)
   - Operator ID (can differ from technical)
   - Beam energy from config
```

## Data Storage

### Operator JSON Format

```json
{
  "operators": {
    "john_doe": {
      "name": "John",
      "surname": "Doe",
      "email": "john.doe@example.com",
      "phone": "+1-555-0100",
      "institution": "Research Institute"
    },
    "jane_smith": {
      "name": "Jane",
      "surname": "Smith",
      "email": "jane.smith@example.com",
      "phone": "",
      "institution": "University Lab"
    }
  },
  "current_operator_id": "john_doe"
}
```

### Technical Container Metadata

```
Root attributes:
├── operator_id: "john_doe"
├── distance_cm: 17.0
├── poni_distance_cm: 17.05  (if from PONI)
├── acquisition_date: "2026-02-11"
└── ... other metadata
```

### Session Container Metadata

```
Root attributes:
├── sample_id: "SAMPLE_001"
├── operator_id: "jane_smith"  (can differ from technical)
├── distance_cm: 17.0
├── beam_energy_keV: 17.5  (from global config)
├── session_id: "..."
└── ... other metadata
```

## Smart Defaults and UX

### Current Operator Concept
- Set on startup
- Used as default in all dialogs
- Can be changed per-container
- Persistent across sessions

### Distance Intelligence
- **Technical Container**:
  - Reads from PONI if available
  - Validates ±5% tolerance
  - Allows override with warning
- **Session Container**:
  - Can read from active technical container
  - Tooltip reminds user to match technical
  - Validation when linking to technical

### Operator Reuse
- **Common case**: Same operator for technical and session
  - Pre-selected automatically
  - One click to confirm
- **Different operators**:
  - Dropdown allows selection
  - E.g., technician does calibration, researcher does sample measurements
- **Add new operator**:
  - Inline in all dialogs
  - No need to exit and reconfigure

## Cloud Integration Ready

### JSON Format Benefits
- Easy to sync to cloud
- Human-readable
- Standard format
- Can be updated by cloud service

### Future Cloud Features
```python
# Planned integration:
operator_manager.sync_from_cloud()  # Download operator list
operator_manager.push_to_cloud()     # Upload new operators
operator_manager.set_cloud_url(url)  # Configure cloud endpoint
```

## Files Modified/Created

### Created
1. `src/hardware/difra/gui/operator_manager.py` (523 lines)
   - OperatorManager
   - OperatorSelectionDialog
   - NewOperatorDialog

2. `src/hardware/difra/gui/technical_container_dialog.py` (274 lines)
   - TechnicalContainerDialog

3. `src/hardware/difra/resources/config/operators.json` (auto-generated)
   - Operator database

### Modified
1. `src/hardware/difra/gui/main_window_ext/session_mixin.py`
   - Added operator_manager initialization
   - Added show_operator_selection_dialog()
   - Updated NewSessionDialog with operator selection
   - Updated dialog calls to pass operator_manager

## Integration Points

### For Technical Measurements Tab

```python
# In technical_measurements.py, when generating H5:
from hardware.difra.gui.technical_container_dialog import TechnicalContainerDialog

# Replace existing distance prompt with:
dialog = TechnicalContainerDialog(
    operator_manager=self.operator_manager,  # From main window
    poni_distance_cm=poni_distance_cm,  # From PONI file
    parent=self
)

if dialog.exec_() == QDialog.Accepted:
    distance_cm, operator_id = dialog.get_parameters()
    
    # Use these in generate_from_aux_table():
    container_id, file_path = technical_container.generate_from_aux_table(
        folder=folder,
        aux_measurements=aux_measurements,
        pony_data=pony_data,
        detector_config=detector_config,
        active_detector_ids=active_detector_ids,
        distance_cm=distance_cm,
        operator_id=operator_id,  # Add to container metadata
    )
```

### For Main Window

```python
# In main_window.py or main_window_basic.py:
class MainWindow(..., SessionMixin):
    def __init__(self):
        super().__init__()
        
        # Initialize session management (includes operator selection)
        self.init_session_manager()
        
        # operator_manager is now available as self.operator_manager
```

## Testing Checklist

- [ ] Operator JSON creation on first run
- [ ] Operator selection dialog on startup
- [ ] Add new operator from startup dialog
- [ ] Technical container dialog with PONI distance
- [ ] Technical container dialog without PONI
- [ ] Distance validation (±5% tolerance)
- [ ] Override distance warning
- [ ] Session container with operator selection
- [ ] Add new operator from session dialog
- [ ] Add new operator from technical dialog
- [ ] Pre-select current operator
- [ ] Different operators for technical vs session
- [ ] Operator details display
- [ ] Cancel operator selection
- [ ] Operator persistence across restarts

## Next Steps

1. **Integrate TechnicalContainerDialog**:
   - Update technical_measurements.py to use new dialog
   - Remove old distance prompt code
   - Pass operator_id to generate_from_aux_table()

2. **Update Technical Container Writer**:
   - Add operator_id to container metadata
   - Store in root attributes

3. **Test Full Workflow**:
   - Start DIFRA → Select operator
   - Generate technical container → Select operator + distance
   - Create session → Select operator + distance
   - Verify metadata in both containers

4. **Cloud Integration (Future)**:
   - Add cloud sync methods to OperatorManager
   - Add settings for cloud URL
   - Auto-sync on startup (optional)
   - Upload new operators to cloud

## Benefits

✅ **User Experience**:
- One-time operator selection on startup
- Smart defaults (current operator pre-selected)
- Inline operator creation (no need to exit dialogs)
- Clear distance validation with PONI
- Flexible: different operators allowed

✅ **Data Quality**:
- Operator info stored with each container
- Distance validation prevents errors
- Full audit trail (who created what)
- Contact information preserved

✅ **Workflow Efficiency**:
- Common case (same operator): 1 click
- Different operator: 2 clicks
- New operator: inline creation
- No interruption to measurement workflow

✅ **Cloud Ready**:
- JSON format for easy sync
- Extensible operator schema
- Future multi-site support
- Centralized operator database (future)
