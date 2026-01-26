@echo off
setlocal enabledelayedexpansion

REM Determine repository root from this script directory (3 levels up: bin -> hardware -> src -> root)
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..\..") do set REPO_ROOT=%%~fI

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

echo Starting D2XC software...
echo Using conda environment: %CONDA_ENV%
echo Repository root: %REPO_ROOT%

REM Check if CONDA_ENV is a path (contains backslash or colon) or a name
echo %CONDA_ENV% | findstr /C:":\" >nul
if %errorlevel% equ 0 (
  REM It's a path, use -p flag
  conda run -p "%CONDA_ENV%" python "%REPO_ROOT%\src\hardware\difra\gui\main_app.py" %*
) else (
  REM It's a name, use -n flag
  conda run -n %CONDA_ENV% python "%REPO_ROOT%\src\hardware\difra\gui\main_app.py" %*
)

endlocal
