@echo off
setlocal enabledelayedexpansion

REM Determine repository root (one level up from this script directory)
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..") do set REPO_ROOT=%%~fI

set CONFIG_PATH=%REPO_ROOT%\src\hardware\eosdxdc\resources\config\global.json

REM Read conda env name from JSON using PowerShell
for /f "usebackq delims=" %%E in (`powershell -NoProfile -Command "(Get-Content -Raw '%CONFIG_PATH%') | ConvertFrom-Json | Select-Object -ExpandProperty conda"`) do set CONDA_ENV=%%E

if "%CONDA_ENV%"=="" (
  echo [ERROR] Could not read 'conda' from %CONFIG_PATH%
  exit /b 1
)

where conda >nul 2>&1
if errorlevel 1 (
  echo [ERROR] 'conda' was not found on PATH. Please run from an Anaconda Prompt or add conda to PATH.
  exit /b 1
)

REM Launch the GUI using the specified conda environment
conda run -n %CONDA_ENV% python "%REPO_ROOT%\src\hardware\eosdxdc\gui\main_app.py" %*

endlocal
