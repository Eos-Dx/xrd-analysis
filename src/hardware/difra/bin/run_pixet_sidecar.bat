@echo off
setlocal enabledelayedexpansion

REM Determine repository root (four levels up: bin -> difra -> hardware -> src -> root)
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..\..\..") do set REPO_ROOT=%%~fI

set SIDECAR_ENV=%DIFRA_SIDECAR_ENV%
if "%SIDECAR_ENV%"=="" set SIDECAR_ENV=ulster37

set SIDECAR_HOST=%PIXET_SIDECAR_HOST%
if "%SIDECAR_HOST%"=="" set SIDECAR_HOST=127.0.0.1

set SIDECAR_PORT=%PIXET_SIDECAR_PORT%
if "%SIDECAR_PORT%"=="" set SIDECAR_PORT=51001

set CONDA_CMD=conda
where conda >nul 2>&1
if errorlevel 1 (
  echo [INFO] 'conda' not found in PATH, searching common installation locations...

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
    echo [ERROR] Could not find conda installation.
    exit /b 1
  )
)

cd /d %REPO_ROOT%
set PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%
set PYTHONUNBUFFERED=1

set SIDECAR_PY=
for /f "usebackq delims=" %%V in (`%CONDA_CMD% run --live-stream --no-capture-output -n %SIDECAR_ENV% python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul`) do set SIDECAR_PY=%%V
if "%SIDECAR_PY%"=="" (
  echo [ERROR] Sidecar env '%SIDECAR_ENV%' is not available.
  exit /b 1
)
if /I not "%SIDECAR_PY%"=="3.7" (
  echo [ERROR] Sidecar env '%SIDECAR_ENV%' must be Python 3.7, found %SIDECAR_PY%.
  exit /b 1
)

echo [INFO] Starting PIXet sidecar in env: %SIDECAR_ENV%
echo [INFO] Sidecar endpoint: %SIDECAR_HOST%:%SIDECAR_PORT%

%CONDA_CMD% run --live-stream --no-capture-output -n %SIDECAR_ENV% python -u "%REPO_ROOT%\src\hardware\difra\scripts\pixet_sidecar_server.py" --host %SIDECAR_HOST% --port %SIDECAR_PORT%

endlocal
