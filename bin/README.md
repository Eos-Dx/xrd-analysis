# D2XC Software Launchers

This directory contains launcher scripts for the D2XC software. These scripts automatically configure the environment and launch the GUI application.

## Available Launchers

### Windows Batch Script (`run_d2xc.bat`)
- **Platform**: Windows
- **Usage**: Double-click the file or run from Command Prompt
- **Features**:
  - Automatically reads conda environment from configuration
  - Provides error messages with pause for debugging
  - Supports command-line arguments

### Python Script (`run_d2xc.py`)
- **Platform**: Cross-platform (Windows, macOS, Linux)
- **Usage**: `python run_d2xc.py [arguments]`
- **Features**:
  - Platform-independent launcher
  - Detailed error reporting
  - Supports command-line arguments

### PowerShell Script (`run_d2xc.ps1`)
- **Platform**: Windows (PowerShell 5.1+)
- **Usage**: `.\run_d2xc.ps1 [arguments]`
- **Features**:
  - Modern PowerShell implementation
  - Colored output for better visibility
  - Parameter validation and help documentation

## Requirements

1. **Anaconda/Miniconda**: All launchers require conda to be installed and available in PATH
2. **Configuration**: The conda environment name is read from `src/hardware/eosdxdc/resources/config/global.json`
3. **Python Environment**: The specified conda environment must exist and contain all required dependencies

## Usage Examples

```batch
# Windows Batch
run_d2xc.bat

# Python (any platform)
python run_d2xc.py
python run_d2xc.py --debug

# PowerShell
.\run_d2xc.ps1
.\run_d2xc.ps1 --verbose
```

## Configuration

The launchers automatically read the conda environment name from the configuration file:
```
src/hardware/eosdxdc/resources/config/global.json
```

Make sure this file contains a `conda` field with the correct environment name:
```json
{
  "conda": "ulster37",
  ...
}
```

## Troubleshooting

### "conda not found" error
- Ensure Anaconda/Miniconda is installed
- Run the launcher from an Anaconda/Miniconda prompt
- Or add conda to your system PATH

### "Configuration file not found" error
- Verify the repository structure is intact
- Check that `global.json` exists in the expected location

### "conda environment not found" error
- Verify the environment name in `global.json` is correct
- Create the environment if it doesn't exist: `conda create -n <env_name>`

## Development

To modify the launchers:
1. Update the appropriate script file
2. Test with your local setup
3. Commit the changes to version control

The launchers are designed to be robust and provide clear error messages to help users diagnose and resolve issues.
