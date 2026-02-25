# DiFRA - Diffraction Analysis Software

DiFRA (Diffraction Analysis) is a PyQt5-based GUI application for controlling X-ray diffraction hardware and performing calibration measurements.

## Contact Information

**Developer & Support:**
- Sergey Denisov
- Email: sdenisov@matur.co.uk
- Phone: +33 6 25 25 21 59

For technical support, bug reports, or feature requests, please contact via email or phone.

---

## Quick Start

### Launching DiFRA

**Windows:**
```cmd
src\hardware\difra\bin\run_difra.bat
```

**macOS/Linux:**
```bash
./src/hardware/difra/bin/run_difra.sh
```

### Runtime Architecture (gRPC + Detector Sidecar)

The current launchers run DiFRA in a protocol-first layout:

1. GUI starts in modern env (`eosdx13`, Python 3.13).
2. DiFRA gRPC server starts (default: same env as GUI).
3. PIXet detector sidecar starts in legacy env (`ulster37`, Python 3.7).
4. GUI hardware client runs in strict gRPC mode (`HARDWARE_CLIENT_MODE=grpc`, enforced by launchers).
5. gRPC server handles stage directly and routes detector init/capture through sidecar (`DETECTOR_BACKEND=sidecar`).

Sidecar development rule:
- Any code added under the sidecar path must remain compatible with legacy Python `<=3.8` (target runtime is currently Python 3.7).

Default endpoints:
- gRPC: `127.0.0.1:50061`
- Detector sidecar: `127.0.0.1:51001`

Useful environment variables:
- `DIFRA_GUI_ENV`, `DIFRA_GRPC_ENV`, `DIFRA_SIDECAR_ENV`
- `DIFRA_GRPC_HOST`, `DIFRA_GRPC_PORT`
- `PIXET_SIDECAR_HOST`, `PIXET_SIDECAR_PORT`
- `HARDWARE_CLIENT_MODE` (launchers force `grpc`)
- `DIFRA_GRPC_CONFIG` (optional JSON config path for gRPC server)

### Installing Python Dependencies (pip)

DiFRA now includes separate pip requirements files per runtime:

- `src/hardware/difra/requirements-ulster37-38.txt` - legacy runtime (sidecar target: `ulster37`, Python 3.7)
- `src/hardware/difra/requirements-eosdx13.txt` - modern runtime (`eosdx13`, Python 3.13)

Install with:

```bash
pip install -r src/hardware/difra/requirements-ulster37-38.txt
# or
pip install -r src/hardware/difra/requirements-eosdx13.txt
```

### First Launch

1. When DiFRA starts, you'll see a welcome screen with two options:
   - **Ulster (Xena)** - For Xena detector setup
   - **Ulster (Moli)** - For Moli detector setup

2. Click on the setup you want to use. The software will load the corresponding configuration.

3. The main window will open with the image viewer and control panels.

---

## Configuration

### Global Settings vs Setup Settings

DiFRA uses a two-level configuration system:

1. **Global Settings** (`resources/config/global.json`) - Apply to all setups:
   - Conda environment name
   - DEV mode toggle
   - Default folders (Windows-specific paths)

2. **Setup Settings** (`resources/config/setups/*.json`) - Specific to each detector:
   - Detector configurations
   - Stage parameters
   - PONI and mask file paths

### Changing the Conda Environment

The conda environment name is used when launching PyFAI calibration and must match your system's conda environment.

**Via GUI (Recommended):**

1. Go to **Settings → Edit Global Settings**
2. Find the line: `"conda": "ulster38"`
3. Change to your environment name (e.g., `"conda": "ulster37"` on Windows)
4. Click **Save**
5. The configuration will reload automatically

**Manually:**

Edit `src/hardware/difra/resources/config/global.json`:
```json
{
    "conda": "your_environment_name",
    "default_setup": "Ulster (Xena)",
    "DEV": false
}
```

**Common Environment Names:**
- Windows: `ulster37`
- macOS: `ulster38`

### Changing Setup-Specific Settings

1. Go to **Settings → Edit Setup Config**
2. This opens the currently selected setup file (e.g., `Ulster (Xena).json`)
3. Modify detector, stage, or calibration settings
4. Click **Save**

**Example Setup Settings:**
- Detector aliases and sizes
- Stage controller parameters
- Default PONI files for calibration
- Mask file paths

---

## Working with DiFRA

### Main Workflow

1. **Select Setup** - Choose Xena or Moli on startup
2. **Create Session** - Set `sample_id`, `study`, distance, and operator
3. **Load Image** - Open an existing image or capture from camera
4. **Define Zones** - Draw `sample_holder`/`include`/`exclude` zones
5. **Run Measurements** - Execute point-based scans with configured detectors
6. **Finalize and Send** - Use Session tab queue to close+send selected/all containers

### Key Features

#### Zone Measurements
- Draw and manage holder/include/exclude zones on the sample image
- Configure exposure time and number of frames
- Generate points and run automated measurements
- Data is written to session HDF5 containers

#### Technical Measurements (Auxiliary)
- Capture calibration images
- Manage PONI files (PyFAI calibration data)
- Load existing technical containers or raw technical files
- Auto-assign technical types and primary rows from loaded containers
- Apply masks to detector images
- View real-time detector output

#### Session Queue and Archive
- Session tab lists all `session_*.h5` containers in measurements folder
- Close/send one, selected set, or all containers (fake cloud send in current dev mode)
- Sent containers are locked, moved to session archive, and removed from pending list
- Archive list shows sample/study/operator/created/archived metadata

#### PyFAI Calibration
1. Set the working folder where calibration images are stored
2. Click **PyFAI** button to launch PyFAI-calib2
3. A new terminal window opens with the calibration GUI
4. Perform calibration and save `.poni` files
5. Load `.poni` files back into DiFRA for measurements

### DEV/Demo Mode

Toggle between production and demo modes:
- **Production Mode**: Normal operation with hardware
- **Demo Mode** (DEV=true):
  - Loads default test image automatically
  - Visual indicator: gray background + "[DEMO]" in title
  - Use for testing without hardware

**Toggle via GUI:**
- Toolbar button: **Switch to Demo** / **Switch to Production**

**Toggle via Settings:**
- **Settings → Edit Global Settings**
- Change `"DEV": false` to `"DEV": true`

---

## File Locations

### Configuration Files
```
src/hardware/difra/resources/config/
├── global.json              # Global settings (conda env, defaults)
└── setups/
    ├── Ulster (Xena).json   # Xena detector configuration
    └── Ulster (Moli).json   # Moli detector configuration
```

### Launcher Scripts
```
src/hardware/difra/bin/
├── run_difra.bat            # Windows launcher
├── run_difra.sh             # macOS/Linux launcher
├── run_difra_dual_env.sh    # macOS/Linux dual-env launcher (gRPC + sidecar)
└── run_difra_embedded.bat   # Windows embedded version
```

### Resources
```
src/hardware/difra/resources/
├── config/                  # Configuration files
├── images/                  # UI icons and logos
└── motivation/              # Startup motivation phrases
```

---

## Troubleshooting

### PyFAI Won't Launch

**Symptom:** Clicking PyFAI button opens terminal but command doesn't run.

**Solutions:**

1. **Verify conda environment exists:**
   ```bash
   conda env list
   ```
   Ensure the environment name in global.json matches an existing environment.

2. **Check PyFAI is installed:**
   ```bash
   conda run -n ulster38 which pyfai-calib2
   # or on Windows:
   conda run -n ulster37 where pyfai-calib2
   ```

3. **Install PyFAI if missing:**
   ```bash
   conda activate ulster38
   conda install -c conda-forge pyfai
   ```

4. **Check conda environment name:**
   - **Settings → Edit Global Settings**
   - Verify `"conda"` matches your system

### macOS Permission Issues

**Symptom:** Terminal won't open when clicking PyFAI.

**Solution:** DiFRA uses `.command` files which don't require AppleScript permissions. If Terminal still doesn't open:
- Check System Preferences → Security & Privacy → Privacy → Automation
- Ensure Terminal has necessary permissions

### Hardware Not Detected

**Solutions:**

1. **Check hardware connections** (USB cables, power)
2. **Verify detector drivers installed:**
   - Timepix detectors: PIXet SDK installed at `C:\Program Files\PIXet Pro`
   - Stages: Thorlabs Kinesis at `C:\Program Files\Thorlabs\Kinesis`
3. **Check setup configuration:**
   - **Settings → Edit Setup Config**
   - Verify detector aliases and parameters

### Configuration Errors

**Symptom:** JSON syntax error when saving settings.

**Solution:** Validate JSON syntax:
- Ensure all quotes are paired: `"key": "value"`
- Check for missing commas between items
- Use online JSON validators if needed

---

## Advanced Usage

### Custom Detector Setups

To create a new detector setup:

1. Copy an existing setup file:
   ```bash
   cp "resources/config/setups/Ulster (Xena).json" \
      "resources/config/setups/My Setup.json"
   ```

2. Edit detector aliases, sizes, and parameters

3. **Note:** Currently, only Xena and Moli appear in the welcome dialog. To add more setups, modify:
   ```
   src/hardware/difra/gui/views/welcome_dialog.py
   ```
   Line 49: Add your setup name to `_setup_names` list

### Environment Variables

DiFRA checks these environment variables for SDK paths:

- `PIXET_SDK_PATH` - PIXet Pro SDK location (default: `C:\Program Files\PIXet Pro`)
- `KINESIS_SDK_PATH` - Thorlabs Kinesis SDK (default: `C:\Program Files\Thorlabs\Kinesis`)

Set custom paths if your SDKs are installed elsewhere:

**Windows:**
```cmd
set PIXET_SDK_PATH=D:\Custom\Path\PIXet
```

**macOS/Linux:**
```bash
export PIXET_SDK_PATH=/opt/pixet
```

### Logs

Application logs are stored in:
```
src/hardware/difra/logs/
```

Check logs for debugging hardware issues or crashes.

---

## Platform-Specific Notes

### Windows
- Uses `conda activate` for PyFAI launch
- Paths use backslashes (`\`)
- Hardware SDKs required for detector operation

### macOS
- Uses `conda run -n` for PyFAI launch (more reliable)
- Creates temporary `.command` files for Terminal
- Some hardware may not be available on macOS

### Linux
- Attempts multiple terminal emulators (gnome-terminal, konsole, xterm)
- May require additional setup for hardware drivers

---

## System Requirements

### Software
- Python 3.8+
- PyQt5
- Conda (Miniconda or Anaconda)
- PyFAI (for calibration)

### Hardware (Optional)
- Timepix detectors (via PIXet SDK)
- Thorlabs motorized stages (via Kinesis)

### Operating Systems
- Windows 10/11 (primary development platform)
- macOS 10.15+
- Linux (Ubuntu 20.04+)

---

## Version History

See commit history for detailed changes.

**Key Updates:**
- Split configuration: global.json + setup-specific configs
- Welcome dialog with setup selection
- GUI-based settings editors
- Cross-platform PyFAI launcher
- Improved error handling and logging

---

## License

Copyright © 2025 - All rights reserved.
Contact developer for licensing information.
