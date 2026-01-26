# MarlinStageController - Quick Start Guide

## What Was Added

A new `MarlinStageController` class has been added to `xystages.py` for controlling the **Moli machine's XY translation stage** via Marlin/GRBL firmware.

## Key Features

- ✅ Compatible with existing `BaseStageController` interface
- ✅ Works with your colleague's Marlin firmware stage (COM4)
- ✅ Supports the same positions: Sample In (57.875, 63.625) and Sample Out (90, 0)
- ✅ Enforces safety limits: X: 0-90mm, Y: 0-100mm
- ✅ Thread-safe serial communication

## Quick Example

```python
from xystages import MarlinStageController

# Configure for Moli machine
config = {
    "id": "COM4",
    "alias": "MOLI_XY_STAGE",
    "settings": {
        "limits_mm": {
            "x": [0.0, 90.0],
            "y": [0.0, 100.0]
        }
    }
}

# Initialize
stage = MarlinStageController(config)
stage.init_stage()

# Use it
stage.home_stage()
stage.move_stage(57.875, 63.625)  # Sample in
stage.move_stage(90.0, 0.0)       # Sample out

# Cleanup
stage.deinit()
```

## Files Created

1. **Modified:**
   - `xystages.py` - Added `MarlinStageController` class

2. **Documentation:**
   - `MARLIN_STAGE_README.md` - Full documentation
   - `MARLIN_QUICKSTART.md` - This file

3. **Examples:**
   - `marlin_stage_example.py` - Python usage example
   - `marlin_stage_example_config.yaml` - YAML configuration example

## How to Integrate

### Option 1: Direct Use (Recommended)

Replace your colleague's GUI serial code with:

```python
from xystages import MarlinStageController

stage = MarlinStageController({
    "id": "COM4",
    "alias": "MOLI_XY_STAGE"
})
stage.init_stage()
```

### Option 2: Factory Pattern

If you have a stage factory, add:

```python
def create_stage_controller(config):
    driver = config.get("driver", "dummy")
    
    if driver == "marlin":
        return MarlinStageController(config)
    elif driver == "thorlabs":
        return XYStageLibController(config)
    else:
        return DummyStageController(config)
```

## Testing

Run the example:

```bash
cd C:\dev\xrd-analysis\src\hardware\difra\hardware
python marlin_stage_example.py
```

## Next Steps

1. ✅ Test with actual hardware on COM4
2. ✅ Update your application's configuration to use `driver: "marlin"`
3. ✅ Optionally integrate with your colleague's PyQt6 GUI
4. ✅ Configure limits and positions as needed for Moli machine

## Support

See `MARLIN_STAGE_README.md` for:
- Complete API reference
- Troubleshooting guide
- Advanced configuration
- Integration examples

## Comparison: Before vs After

### Before (Your Colleague's GUI)
```python
# Direct serial communication in GUI
self.ser = serial.Serial("COM4", 115200)
self.ser.write(b"G28 X Y\n")
# ... manual position tracking, no limits, GUI-only
```

### After (MarlinStageController)
```python
# Standardized interface, reusable
stage = MarlinStageController(config)
stage.home_stage()  # Automatic limit checking, logging, error handling
```

## Benefits

1. **Compatibility** - Works with existing codebase that uses `BaseStageController`
2. **Safety** - Automatic limit checking prevents hardware damage
3. **Reliability** - Thread-safe serial communication
4. **Maintainability** - Separation of concerns (GUI vs control logic)
5. **Logging** - Built-in logging for debugging
6. **Testability** - Can be tested without GUI
