# DiFRA Launcher and Application Logging

## Overview

The DiFRA application now has comprehensive logging enabled at both the launcher and application levels. This helps with troubleshooting and debugging any issues that may occur.

## Log File Locations

### Windows

#### Launcher Logs
- **Location**: `C:\Users\<YourUsername>\AppData\Local\Ulster\launcher_logs\`
- **File format**: `launcher_YYYYMMDD_HHMMSS.log`
- **Example**: `launcher_20251120_114500.log`

#### Application Logs
- **Location**: `C:\Users\<YourUsername>\AppData\Local\Ulster\`
- **File name**: `ulster.log`

### Linux/Mac

#### Launcher Logs
- **Location**: `~/.local/state/Ulster/launcher_logs/`
- **File format**: `launcher_YYYYMMDD_HHMMSS.log`

#### Application Logs
- **Location**: `~/.local/state/Ulster/`
- **File name**: `ulster.log`

## What's Logged

### Launcher (`launcher.py`)
- Python version and platform information
- Script and repository paths
- Configuration file loading and validation
- Conda environment checking
- Application launch command
- All errors with full stack traces

### Application (`main_app.py`)
- Application startup sequence
- PyQt5 initialization
- Welcome dialog creation
- Main window creation and initialization
- All component initialization steps
- Runtime errors with full context

### Main Window (`main_window.py`)
- Detailed initialization steps
- Component creation (image view, docks, toolbars)
- Hardware controller setup
- State management operations
- All errors during window creation

## Log Levels

Logs are written at different levels:

- **DEBUG**: Detailed diagnostic information (file only)
- **INFO**: General informational messages (file and console)
- **WARNING**: Warning messages (file and console)
- **ERROR**: Error messages with context (file and console)
- **CRITICAL**: Critical errors that prevent operation (file and console)

## Console Output

In addition to log files, important messages are also displayed in the terminal/console:
- INFO level and above for launcher
- All error messages are shown with clear formatting

## Error Messages

When an error occurs, you will see:
1. **GUI Dialog**: A popup window with the error details
2. **Console Output**: Formatted error message in the terminal
3. **Log File**: Complete error with full traceback and context

## Viewing Logs

### Quick Access (Windows)

Open PowerShell and run:
```powershell
# View launcher logs
explorer $env:LOCALAPPDATA\Ulster\launcher_logs

# View application log
explorer $env:LOCALAPPDATA\Ulster
```

Or directly open the log file:
```powershell
# View latest launcher log
Get-Content (Get-ChildItem $env:LOCALAPPDATA\Ulster\launcher_logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

# View application log
Get-Content $env:LOCALAPPDATA\Ulster\ulster.log
```

### Quick Access (Linux/Mac)

```bash
# View launcher logs
ls -lt ~/.local/state/Ulster/launcher_logs/

# View application log
tail -f ~/.local/state/Ulster/ulster.log
```

## Log Rotation

- **Launcher logs**: Each launch creates a new timestamped log file
- **Application logs**: Rotates when reaching 20MB, keeping 10 backup files
- Old logs are automatically managed to prevent disk space issues

## Troubleshooting

If you experience an error:

1. Check the terminal/console output for immediate error details
2. Look at the GUI error dialog for the log file location
3. Open the log file for complete information
4. Share the relevant log entries when reporting issues

## Example Log Output

### Successful Launch
```
11:45:00 | INFO     | Launcher logging initialized. Log file: C:\Users\...\launcher_20251120_114500.log
11:45:00 | INFO     | DiFRA Launcher Starting
11:45:00 | INFO     | Python version: 3.9.7 ...
11:45:00 | INFO     | Platform: win32
11:45:01 | INFO     | Conda version: conda 23.1.0
11:45:02 | INFO     | Application launched successfully
```

### Error Example
```
11:45:00 | INFO     | Checking conda availability...
11:45:01 | ERROR    | Conda not found: [WinError 2] The system cannot find the file specified: 'conda'
11:45:01 | ERROR    | PATH: C:\Windows\system32;C:\Windows;...
```

## Support

When reporting issues, please include:
- The log file from `launcher_logs/` (latest timestamped file)
- The main application log (`ulster.log`)
- A description of what you were doing when the error occurred
- Any error dialogs that appeared
