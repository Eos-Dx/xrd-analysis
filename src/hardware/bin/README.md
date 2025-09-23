# D2XC/EOSDxDc Launchers (Hardware/Bin)

This directory contains launcher scripts for the D2XC (EOSDxDc) software from within the hardware source tree.

## Available Launchers

### `run_eosdxdc.bat` (Original Batch Launcher)
- **Platform**: Windows
- **Usage**: Double-click or run from Command Prompt
- **Purpose**: Original EOSDxDc batch launcher
- **Status**: ✅ Working

### `run_d2xc.bat` (Updated Batch Launcher)
- **Platform**: Windows
- **Usage**: Double-click or run from Command Prompt
- **Purpose**: Enhanced D2XC batch launcher with better error messages
- **Status**: ✅ Working

### `launcher.py` (Python GUI Launcher)
- **Platform**: Cross-platform
- **Usage**: `python launcher.py`
- **Purpose**: Python launcher with GUI error dialogs
- **Features**:
  - GUI error messages using tkinter
  - Detached process launching
  - Cross-platform compatibility
- **Status**: ✅ Working

## Differences from Root /bin Directory

The launchers in this directory (`src/hardware/bin/`) are located within the hardware source tree and use different path calculations compared to the root `/bin` directory:

- **Root /bin launchers**: Calculate repository root as `../` (1 level up)
- **Hardware /bin launchers**: Calculate repository root as `../../../` (3 levels up: bin → hardware → src → root)

## Configuration

All launchers read the conda environment name from:
```
src/hardware/eosdxdc/resources/config/global.json
```

## Usage Examples

```batch
# Windows batch launchers
run_eosdxdc.bat
run_d2xc.bat

# Python launcher
python launcher.py
```

## Notes

- The `run_eosdxdc.exe` is a working IExpress self-extracting executable
- Uses embedded batch file with hardcoded repository path for proper temp directory execution
- The batch and Python launchers provide better reliability and error reporting
- All launchers launch the same application: `src/hardware/eosdxdc/gui/main_app.py`

## For Developers

To create a proper executable from `launcher.py`, you can use PyInstaller:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "D2XC_Launcher" launcher.py
```

This will create a standalone executable in the `dist/` folder.
