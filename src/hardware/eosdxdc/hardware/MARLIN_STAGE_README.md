# MarlinStageController Documentation

## Overview

The `MarlinStageController` is a stage controller implementation for XY translation stages running **Marlin** or **GRBL** firmware. This controller is designed for the **Moli machine** and communicates via serial port using G-code commands.

## Features

- ✅ **Standard Interface**: Inherits from `BaseStageController` for compatibility with existing code
- ✅ **Serial Communication**: Communicates with Marlin/GRBL firmware via pyserial
- ✅ **Position Tracking**: Monitors real-time stage position using M114 commands
- ✅ **Safety Limits**: Enforces configurable axis limits to prevent collisions
- ✅ **Emergency Stop**: Supports M112 emergency stop command
- ✅ **Thread-Safe**: Uses background thread for serial reading
- ✅ **Configurable**: Supports YAML configuration for limits, positions, and serial settings

## Hardware Requirements

- XY translation stage with Marlin or GRBL firmware
- Serial connection (USB or RS-232)
- Windows COM port or Linux /dev/ttyUSB device

## Software Requirements

```bash
pip install pyserial
```

## Configuration

### Basic Configuration

```yaml
xystage:
  driver: "marlin"
  id: "COM4"  # Serial port
  alias: "MOLI_XY_STAGE"
  settings:
    limits_mm:
      x: [0.0, 90.0]
      y: [0.0, 100.0]
    home: [0.0, 0.0]
    load: [90.0, 0.0]
```

### Advanced Configuration

```yaml
xystage:
  driver: "marlin"
  id: "COM4"
  alias: "MOLI_XY_STAGE"
  settings:
    limits_mm:
      x:
        min: 0.0
        max: 90.0
      y:
        min: 0.0
        max: 100.0
    home: [0.0, 0.0]
    load: [90.0, 0.0]
    
    # Serial communication settings
    baudrate: 115200
    timeout: 1.0
    feedrate: 3000  # mm/min
    homing_timeout: 15  # seconds
```

## Usage Examples

### Basic Usage

```python
from xystages import MarlinStageController

# Create configuration
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

# Initialize controller
stage = MarlinStageController(config=config, baudrate=115200, feedrate=3000)

try:
    # Connect to stage
    stage.init_stage()
    
    # Home the stage
    x, y = stage.home_stage()
    print(f"Homed to: X={x:.3f}, Y={y:.3f}")
    
    # Move to a position
    x, y = stage.move_stage(50.0, 50.0)
    print(f"Moved to: X={x:.3f}, Y={y:.3f}")
    
    # Get current position
    x, y = stage.get_xy_position()
    print(f"Current: X={x:.3f}, Y={y:.3f}")
    
finally:
    # Always cleanup
    stage.deinit()
```

### Using Predefined Positions

```python
# Get home and load positions from config
positions = stage.get_home_load_positions()
home_x, home_y = positions["home"]
load_x, load_y = positions["load"]

# Move to load position (sample out)
stage.move_stage(load_x, load_y)

# Move to sample in position (specific to Moli machine)
stage.move_stage(57.875, 63.625)
```

### Error Handling

```python
from xystages import MarlinStageController, StageAxisLimitError

try:
    stage.move_stage(100.0, 50.0)  # Exceeds X limit (90mm)
except StageAxisLimitError as e:
    print(f"Limit exceeded: {e}")
    print(f"Axis: {e.axis}, Value: {e.value}, Limits: [{e.min_limit}, {e.max_limit}]")
```

### Emergency Stop

```python
# In case of emergency
stage.emergency_stop()
```

## API Reference

### Constructor

```python
MarlinStageController(
    config,                # Configuration dict
    port=None,            # Serial port (overrides config['id'])
    baudrate=115200,      # Serial baud rate
    timeout=1.0,          # Serial timeout (seconds)
    feedrate=3000,        # Movement speed (mm/min)
    homing_timeout=15     # Homing timeout (seconds)
)
```

### Methods

#### `init_stage() -> bool`
Initialize serial connection and configure stage.

**Returns:** `True` if successful, `False` otherwise

#### `home_stage(timeout_s=45) -> tuple[float, float]`
Home the stage using G28 command.

**Args:**
- `timeout_s`: Maximum time to wait for homing

**Returns:** `(x, y)` position after homing

#### `move_stage(x_mm, y_mm, move_timeout=20) -> tuple[float, float]`
Move stage to absolute position.

**Args:**
- `x_mm`: Target X position in mm
- `y_mm`: Target Y position in mm
- `move_timeout`: Maximum time to wait for move

**Returns:** `(x, y)` final position

**Raises:** `StageAxisLimitError` if position exceeds limits

#### `get_xy_position() -> tuple[float, float]`
Get current XY position via M114 command.

**Returns:** `(x, y)` current position

#### `emergency_stop()`
Send M112 emergency stop command to controller.

#### `deinit()`
Close serial connection and cleanup resources.

#### `get_limits() -> dict`
Get axis limits.

**Returns:** `{"x": (min, max), "y": (min, max)}`

#### `get_home_load_positions() -> dict`
Get predefined positions.

**Returns:** `{"home": (x, y), "load": (x, y)}`

## G-Code Commands Used

| Command | Purpose |
|---------|---------|
| `G90` | Set absolute positioning mode |
| `G0 X... Y... F...` | Rapid move to position |
| `G28 X Y` | Home X and Y axes |
| `M114` | Request current position |
| `M112` | Emergency stop |

## Coordinate System

The Marlin stage uses an absolute coordinate system with origin at (0, 0):

```
Y (100mm)
^
|
|  [Working Area]
|
|
0 ----------------> X (90mm)
```

### Moli Machine Positions

Based on your colleague's configuration:
- **Home**: (0.0, 0.0) - Stage origin after homing
- **Sample In**: (57.875, 63.625) - Measurement position
- **Sample Out**: (90.0, 0.0) - Loading position

## Differences from Thorlabs Stage

| Feature | Marlin Stage | Thorlabs Stage |
|---------|--------------|----------------|
| Communication | Serial (G-code) | DLL (Kinesis) |
| Coordinate Origin | (0, 0) | Device-specific |
| Default Limits | X: 0-90mm, Y: 0-100mm | ±14mm |
| Homing | G28 command | BDC_Home API |
| Position Query | M114 command | BDC_GetPosition API |

## Integration with Your Colleague's GUI

The existing PyQt6 GUI can be modified to use `MarlinStageController`:

```python
# Instead of direct serial communication in the GUI:
from xystages import MarlinStageController

# In MainWindow.__init__():
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
self.stage = MarlinStageController(config)
self.stage.init_stage()

# Replace serial commands with:
self.stage.move_stage(x, y)
self.stage.home_stage()
x, y = self.stage.get_xy_position()
```

## Troubleshooting

### Connection Issues

1. **Check COM port**: Verify correct port in Device Manager (Windows) or `ls /dev/tty*` (Linux)
2. **Check baudrate**: Ensure baudrate matches firmware (typically 115200 or 250000)
3. **Wait after connection**: Marlin resets on serial connection, wait 2 seconds
4. **Check permissions**: On Linux, user needs access to serial ports (`sudo usermod -a -G dialout $USER`)

### Position Tracking Issues

1. **Position drift**: Call `get_xy_position()` periodically to sync with hardware
2. **Lost position after power cycle**: Always home the stage after initialization

### Movement Issues

1. **Slow movements**: Increase feedrate parameter (default 3000 mm/min)
2. **Jerky movements**: Ensure firmware acceleration settings are configured
3. **Timeout errors**: Increase `move_timeout` for long movements

## Logging

Enable debug logging to see G-code commands:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Thread Safety

The controller uses a background thread for serial reading. However, the main API methods are **not** thread-safe. Always call methods from the same thread.

## See Also

- [xystages.py](xystages.py) - Source code
- [marlin_stage_example.py](marlin_stage_example.py) - Example usage
- [marlin_stage_example_config.yaml](marlin_stage_example_config.yaml) - Configuration example
