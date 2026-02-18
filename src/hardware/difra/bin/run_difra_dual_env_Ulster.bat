@echo off
setlocal enabledelayedexpansion

REM Determine repository root (four levels up: bin -> difra -> hardware -> src -> root)
set SCRIPT_DIR=%~dp0
for %%i in ("%SCRIPT_DIR%..\..\..\..") do set REPO_ROOT=%%~fi
cd /d "%REPO_ROOT%"

set CONFIG_PATH=%REPO_ROOT%\src\hardware\difra\resources\config\global.json
set MAIN_CONFIG_PATH=%REPO_ROOT%\src\hardware\difra\resources\config\main.json

where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 'conda' was not found on PATH. Please ensure conda is initialized in your shell.
    exit /b 1
)

set PYTHONUNBUFFERED=1
if defined PYTHONPATH (
    set PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%
) else (
    set PYTHONPATH=%REPO_ROOT%\src
)

REM Determine GUI environment
set GUI_ENV=%DIFRA_GUI_ENV%
if not defined GUI_ENV (
    for /f "delims=" %%i in ('python -c "import json; print(json.load(open('%CONFIG_PATH%'))['conda'])" 2^>nul') do set GUI_ENV=%%i
)
if not defined GUI_ENV set GUI_ENV=eosdx13

REM Set sidecar environment to ulster37 (instead of ulster38)
if not defined DIFRA_SIDECAR_ENV (
    set SIDECAR_ENV=ulster37
) else (
    set SIDECAR_ENV=%DIFRA_SIDECAR_ENV%
)

if not defined PIXET_SIDECAR_HOST (
    set SIDECAR_HOST=127.0.0.1
) else (
    set SIDECAR_HOST=%PIXET_SIDECAR_HOST%
)

if not defined PIXET_SIDECAR_PORT (
    set SIDECAR_PORT=51001
) else (
    set SIDECAR_PORT=%PIXET_SIDECAR_PORT%
)

if not defined DIFRA_GRPC_ENV (
    set GRPC_ENV=%GUI_ENV%
) else (
    set GRPC_ENV=%DIFRA_GRPC_ENV%
)

if not defined DIFRA_GRPC_HOST (
    set GRPC_HOST=127.0.0.1
) else (
    set GRPC_HOST=%DIFRA_GRPC_HOST%
)

if not defined DIFRA_GRPC_PORT (
    set GRPC_PORT=50061
) else (
    set GRPC_PORT=%DIFRA_GRPC_PORT%
)

set GRPC_CONFIG=%DIFRA_GRPC_CONFIG%
if defined HARDWARE_CLIENT_MODE (
    if not "%HARDWARE_CLIENT_MODE%"=="grpc" (
        echo [WARN] HARDWARE_CLIENT_MODE=%HARDWARE_CLIENT_MODE% overridden to grpc
    )
)
set CLIENT_MODE=grpc

REM Determine GRPC config if not set
if not defined GRPC_CONFIG (
    for /f "delims=" %%i in ('python - "%CONFIG_PATH%" "%MAIN_CONFIG_PATH%" ^< "%SCRIPT_DIR%get_config.py" 2^>nul') do set GRPC_CONFIG=%%i
)

REM Create helper Python script for config resolution if it doesn't exist
if not exist "%SCRIPT_DIR%get_config.py" (
    (
        echo import json
        echo import sys
        echo from pathlib import Path
        echo.
        echo global_cfg = Path^(sys.argv[1]^)
        echo main_cfg = Path^(sys.argv[2]^)
        echo chosen = main_cfg
        echo.
        echo try:
        echo     if global_cfg.exists^(^):
        echo         data = json.loads^(global_cfg.read_text^(^)^)
        echo         setup = str^(data.get^("default_setup"^) or ""^).strip^(^)
        echo         if setup:
        echo             setup_path = global_cfg.parent / "setups" / f"{setup}.json"
        echo             if setup_path.exists^(^):
        echo                 chosen = setup_path
        echo except Exception:
        echo     pass
        echo.
        echo print^(chosen^)
    ) > "%SCRIPT_DIR%get_config.py"
)

REM Check if sidecar port is already open
python -c "import socket; socket.create_connection(('%SIDECAR_HOST%', %SIDECAR_PORT%), timeout=0.25)" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Starting sidecar env=%SIDECAR_ENV% endpoint=%SIDECAR_HOST%:%SIDECAR_PORT%
    start "DiFRA Sidecar" /min conda run --live-stream --no-capture-output -n %SIDECAR_ENV% python -u "%REPO_ROOT%\src\hardware\difra\scripts\pixet_sidecar_server.py" --host %SIDECAR_HOST% --port %SIDECAR_PORT%
    timeout /t 2 /nobreak >nul
) else (
    echo [WARN] Detector sidecar port already in use at %SIDECAR_HOST%:%SIDECAR_PORT%; reusing existing process.
)

REM Check if gRPC port is already open
python -c "import socket; socket.create_connection(('%GRPC_HOST%', %GRPC_PORT%), timeout=0.25)" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Starting gRPC env=%GRPC_ENV% endpoint=%GRPC_HOST%:%GRPC_PORT% config=!GRPC_CONFIG!
    start "DiFRA gRPC" /min conda run --live-stream --no-capture-output -n %GRPC_ENV% python -u "%REPO_ROOT%\src\hardware\difra\grpc_server\server.py" --host %GRPC_HOST% --port %GRPC_PORT% --config "!GRPC_CONFIG!"
    timeout /t 2 /nobreak >nul
) else (
    echo [WARN] gRPC port already in use at %GRPC_HOST%:%GRPC_PORT%; reusing existing process.
)

REM Wait for ports to be ready
echo [INFO] Waiting for services to be ready...
call :wait_for_port %SIDECAR_HOST% %SIDECAR_PORT% "Detector sidecar"
call :wait_for_port %GRPC_HOST% %GRPC_PORT% "DiFRA gRPC server"

set PIXET_BACKEND=sidecar
set DETECTOR_BACKEND=sidecar
set PIXET_SIDECAR_HOST=%SIDECAR_HOST%
set PIXET_SIDECAR_PORT=%SIDECAR_PORT%
set HARDWARE_CLIENT_MODE=%CLIENT_MODE%
set DIFRA_GRPC_HOST=%GRPC_HOST%
set DIFRA_GRPC_PORT=%GRPC_PORT%

echo [INFO] Starting DiFRA GUI env=%GUI_ENV% mode=%HARDWARE_CLIENT_MODE% grpc=%DIFRA_GRPC_HOST%:%DIFRA_GRPC_PORT% detector_backend=sidecar
conda run --live-stream --no-capture-output -n %GUI_ENV% python -u "%REPO_ROOT%\src\hardware\difra\gui\main_app.py" %*

exit /b %errorlevel%

:wait_for_port
set _host=%~1
set _port=%~2
set _label=%~3
set _retries=100
:wait_loop
python -c "import socket; socket.create_connection(('%_host%', %_port%), timeout=0.25)" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] %_label% ready at %_host%:%_port%
    goto :eof
)
set /a _retries-=1
if %_retries% gtr 0 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)
echo [ERROR] %_label% did not become ready at %_host%:%_port%
exit /b 1
