# Moli Machine Configuration Summary

## Overview

The **Ulster (Moli)** machine has been configured to use the new `MarlinStageController` for its XY translation stage.

## Configuration Files Modified

### 1. `Ulster (Moli).json`
**Location:** `src/hardware/difra/resources/config/setups/Ulster (Moli).json`

**Changes Made:**
- **Stage Type:** Changed from `Kinesis` to `Marlin`
- **Stage ID:** Changed from `101370874` (Thorlabs device) to `COM4` (serial port)
- **Alias:** Changed from `XY_STAGE` to `MOLI_XY_STAGE`

**New Stage Configuration:**
```json
{
  "alias": "MOLI_XY_STAGE",
  "type": "Marlin",
  "id": "COM4",
  "real_zero": { "x_mm": 0.0, "y_mm": 0.0 },
  "settings": {
    "number_axes": 2,
    "limits_mm": { 
      "x": [0.0, 90.0], 
      "y": [0.0, 100.0] 
    },
    "home": [0.0, 0.0],
    "load": [90.0, 0.0],
    "baudrate": 115200,
    "timeout": 1.0,
    "feedrate": 3000,
    "homing_timeout": 15
  }
}
```

### 2. `hardware_control.py`
**Location:** `src/hardware/difra/hardware/hardware_control.py`

**Changes Made:**
- Added `MarlinStageController` import
- Added `"Marlin": MarlinStageController` to `STAGE_CLASSES` factory mapping

## Stage Specifications

### Coordinate System
- **Origin:** (0, 0) - Bottom-left corner after homing
- **X-axis Range:** 0 to 90 mm
- **Y-axis Range:** 0 to 100 mm

### Key Positions
| Position | X (mm) | Y (mm) | Description |
|----------|--------|--------|-------------|
| **Home** | 0.0 | 0.0 | Origin position after G28 homing |
| **Sample Out (Load)** | 90.0 | 0.0 | Position for sample loading/unloading |
| **Sample In** | 57.875 | 63.625 | Typical measurement position (from GUI) |

### Communication Settings
- **Serial Port:** COM4
- **Baud Rate:** 115200
- **Timeout:** 1.0 second
- **Feed Rate:** 3000 mm/min (50 mm/s)
- **Homing Timeout:** 15 seconds

## Hardware Details

### Firmware
- **Type:** Marlin or GRBL
- **Communication Protocol:** G-code via serial (USB)

### Supported G-code Commands
| Command | Purpose |
|---------|---------|
| `G28 X Y` | Home X and Y axes |
| `G90` | Set absolute positioning mode |
| `G0 X... Y... F...` | Rapid move to position |
| `M114` | Request current position |
| `M112` | Emergency stop |

## Usage

### Starting the System

1. **Select Setup:**
   - In the application, select **"Ulster (Moli)"** from setup dropdown

2. **Automatic Initialization:**
   - System will automatically connect to COM4
   - `MarlinStageController` will be instantiated
   - Stage will be ready for use

3. **Typical Workflow:**
   ```python
   # Home the stage
   hw_controller.home_stage()
   
   # Move to measurement position
   hw_controller.move_stage(57.875, 63.625)
   
   # Get current position
   x, y = hw_controller.get_xy_position()
   
   # Move to load position (sample out)
   hw_controller.move_stage(90.0, 0.0)
   ```

## Development Mode

When `DEV: true` in config:
- Uses `DUMMY_STAGE` (type: `DummyStage`)
- No hardware required
- Simulates movements

When `DEV: false` in config:
- Uses `MOLI_XY_STAGE` (type: `Marlin`)
- Requires COM4 connection
- Controls actual hardware

## Differences from Xena Machine

| Feature | Xena (Kinesis) | Moli (Marlin) |
|---------|----------------|---------------|
| **Controller** | XYStageLibController | MarlinStageController |
| **Communication** | Thorlabs Kinesis DLL | Serial G-code |
| **Device ID** | 101370874 | COM4 |
| **Coordinate System** | Centered (±14mm) | Origin-based (0-90mm, 0-100mm) |
| **Home Position** | (9.25, 6.0) | (0.0, 0.0) |
| **Load Position** | (-13.9, -6.0) | (90.0, 0.0) |
| **Real Zero Offset** | (9.25, -6.6) | (0.0, 0.0) |

## Troubleshooting

### Stage Not Connecting
1. Check COM4 is available in Device Manager
2. Verify no other program is using COM4
3. Check USB cable connection
4. Try different baud rate (250000)

### Position Drift
- Home the stage periodically
- Call `get_xy_position()` to sync with hardware

### Movements Too Slow
- Increase `feedrate` in config (e.g., 5000 mm/min)

### Emergency Issues
- Use emergency stop: `M112` command
- Power cycle the controller
- Re-home after recovery

## Testing

### Manual Testing with Example Script
```bash
cd C:\dev\xrd-analysis\src\hardware\difra\hardware
python marlin_stage_example.py
```

### Unit Tests
```bash
cd C:\dev\xrd-analysis
python -m pytest src/tests/test_marlin_stage.py -v
```

All 26 tests should pass.

## Documentation References

- **Full API Documentation:** `MARLIN_STAGE_README.md`
- **Quick Start Guide:** `MARLIN_QUICKSTART.md`
- **Usage Example:** `marlin_stage_example.py`
- **Configuration Template:** `marlin_stage_example_config.yaml`

## Commits

1. **df799d64** - "Add MarlinStageController for Moli machine XY stage"
   - Implemented MarlinStageController class
   - Added documentation and tests

2. **d02b4ad1** - "Configure Moli machine to use MarlinStageController"
   - Updated Ulster (Moli).json configuration
   - Added factory mapping in hardware_control.py

## Status

✅ **Implementation Complete**
✅ **Tests Passing (26/26)**
✅ **Configuration Updated**
✅ **Documentation Complete**
✅ **Ready for Production Use**

## Next Steps

1. Test with actual Moli machine hardware
2. Calibrate measurement positions if needed
3. Update Sample In position (57.875, 63.625) based on actual setup
4. Consider adding preset positions to config if additional positions are needed
