@echo off
setlocal enabledelayedexpansion

REM Determine repository root from this script directory
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..") do set REPO_ROOT=%%~fI

set CONFIG_PATH=%REPO_ROOT%\src\hardware\eosdxdc\resources\config\global.json

REM Read conda env name from JSON using PowerShell
for /f "usebackq delims=" %%E in (`powershell -NoProfile -Command "(Get-Content -Raw '%CONFIG_PATH%') | ConvertFrom-Json | Select-Object -ExpandProperty conda"`) do set CONDA_ENV=%%E

if "%CONDA_ENV%"=="" (
  echo [ERROR] Could not read 'conda' from %CONFIG_PATH%
  pause
  exit /b 1
)

where conda >nul 2>&1
if errorlevel 1 (
  echo [ERROR] 'conda' was not found on PATH. Please run from an Anaconda Prompt or add conda to PATH.
  pause
  exit /b 1
)

echo Starting D2XC software...
echo Using conda environment: %CONDA_ENV%
echo Repository root: %REPO_ROOT%

REM Initialize conda for batch file usage
call conda activate base
if errorlevel 1 (
  echo [ERROR] Failed to initialize conda
  pause
  exit /b 1
)

REM Activate the specified environment and run the application
call conda activate %CONDA_ENV%
if errorlevel 1 (
  echo [ERROR] Failed to activate conda environment: %CONDA_ENV%
  pause
  exit /b 1
)

REM Change to repository root and launch the D2XC GUI
cd /d "%REPO_ROOT%"
python "%REPO_ROOT%\src\hardware\eosdxdc\gui\main_app.py" %*

endlocal
