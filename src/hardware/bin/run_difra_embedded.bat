@echo off
setlocal enabledelayedexpansion

REM Embedded repository root path (will be extracted to temp directory)
set REPO_ROOT=C:\dev\xrd-analysis

set CONFIG_PATH=%REPO_ROOT%\src\hardware\difra\resources\config\global.json

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

echo Starting D2XC/DiFRA software...
echo Using conda environment: %CONDA_ENV%
echo Repository root: %REPO_ROOT%

REM Launch the GUI using the specified conda environment
conda run -n %CONDA_ENV% python "%REPO_ROOT%\src\hardware\difra\gui\main_app.py" %*

endlocal
