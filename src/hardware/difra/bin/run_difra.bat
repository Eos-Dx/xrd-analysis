@echo off
setlocal enabledelayedexpansion

REM Determine repository root (four levels up: bin -> difra -> hardware -> src -> root)
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..\..\..") do set REPO_ROOT=%%~fI

set CONFIG_PATH=%REPO_ROOT%\src\hardware\difra\resources\config\global.json

REM Read conda env name from JSON using PowerShell
for /f "usebackq delims=" %%E in (`powershell -NoProfile -Command "(Get-Content -Raw '%CONFIG_PATH%') | ConvertFrom-Json | Select-Object -ExpandProperty conda"`) do set CONDA_ENV=%%E

if "%CONDA_ENV%"=="" (
  echo [ERROR] Could not read 'conda' from %CONFIG_PATH%
  exit /b 1
)

REM Try to find conda if not in PATH
set CONDA_CMD=conda
where conda >nul 2>&1
if errorlevel 1 (
  echo [INFO] 'conda' not found in PATH, searching common installation locations...

  REM Common conda installation paths
  set CONDA_PATHS[0]=%USERPROFILE%\anaconda3
  set CONDA_PATHS[1]=%USERPROFILE%\miniconda3
  set CONDA_PATHS[2]=C:\ProgramData\Anaconda3
  set CONDA_PATHS[3]=C:\ProgramData\Miniconda3
  set CONDA_PATHS[4]=C:\Anaconda3
  set CONDA_PATHS[5]=C:\Miniconda3
  set CONDA_PATHS[6]=C:\Users\Ulster\anaconda3

  set CONDA_FOUND=0
  for /L %%i in (0,1,6) do (
    if defined CONDA_PATHS[%%i] (
      set CONDA_PATH=!CONDA_PATHS[%%i]!
      if exist "!CONDA_PATH!\Scripts\conda.exe" (
        set CONDA_CMD="!CONDA_PATH!\Scripts\conda.exe"
        echo [INFO] Found conda at: !CONDA_PATH!
        set CONDA_FOUND=1
        goto :conda_found
      )
    )
  )

  :conda_found
  if !CONDA_FOUND!==0 (
    echo [ERROR] Could not find conda installation. Please:
    echo   1. Install Anaconda/Miniconda, or
    echo   2. Add conda to your PATH, or
    echo   3. Run this script from an Anaconda Prompt
    echo.
    echo Searched locations:
    for /L %%i in (0,1,6) do (
      if defined CONDA_PATHS[%%i] echo   - !CONDA_PATHS[%%i]!
    )
    exit /b 1
  )
)

REM Launch the GUI using the specified conda environment
echo [INFO] Starting DiFRA GUI with environment: %CONDA_ENV%
%CONDA_CMD% run -n %CONDA_ENV% python "%REPO_ROOT%\src\hardware\difra\gui\main_app.py" %*

endlocal
