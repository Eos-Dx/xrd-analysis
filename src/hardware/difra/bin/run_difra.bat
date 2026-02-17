@echo off
setlocal enabledelayedexpansion

REM Determine repository root (four levels up: bin -> difra -> hardware -> src -> root)
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..\..\..") do set REPO_ROOT=%%~fI

set CONFIG_PATH=%REPO_ROOT%\src\hardware\difra\resources\config\global.json
set MAIN_CONFIG_PATH=%REPO_ROOT%\src\hardware\difra\resources\config\main_win.json

set GUI_ENV=%DIFRA_GUI_ENV%
if "%GUI_ENV%"=="" (
  for /f "usebackq delims=" %%E in (`powershell -NoProfile -Command "(Get-Content -Raw '%CONFIG_PATH%') | ConvertFrom-Json | Select-Object -ExpandProperty conda"`) do set GUI_ENV=%%E
)
if "%GUI_ENV%"=="" set GUI_ENV=eosdx13

set SIDECAR_ENV=%DIFRA_SIDECAR_ENV%
if "%SIDECAR_ENV%"=="" set SIDECAR_ENV=ulster37

set SIDECAR_HOST=%PIXET_SIDECAR_HOST%
if "%SIDECAR_HOST%"=="" set SIDECAR_HOST=127.0.0.1

set SIDECAR_PORT=%PIXET_SIDECAR_PORT%
if "%SIDECAR_PORT%"=="" set SIDECAR_PORT=51001

set GRPC_ENV=%DIFRA_GRPC_ENV%
if "%GRPC_ENV%"=="" set GRPC_ENV=%GUI_ENV%

set GRPC_HOST=%DIFRA_GRPC_HOST%
if "%GRPC_HOST%"=="" set GRPC_HOST=127.0.0.1

set GRPC_PORT=%DIFRA_GRPC_PORT%
if "%GRPC_PORT%"=="" set GRPC_PORT=50061

set GRPC_CONFIG=%DIFRA_GRPC_CONFIG%
if "%GRPC_CONFIG%"=="" (
  for /f "usebackq delims=" %%C in (`powershell -NoProfile -Command "$globalPath='%CONFIG_PATH%'; $mainPath='%MAIN_CONFIG_PATH%'; if (-not (Test-Path $mainPath)) { $mainPath='%REPO_ROOT%\src\hardware\difra\resources\config\main.json' }; $out=$mainPath; if (Test-Path $globalPath) { try { $g=(Get-Content -Raw $globalPath | ConvertFrom-Json); $setup=[string]$g.default_setup; if ($setup) { $setupPath=Join-Path (Join-Path (Split-Path -Parent $globalPath) 'setups') ($setup + '.json'); if (Test-Path $setupPath) { $out=$setupPath } } } catch {} }; Write-Output $out"`) do set GRPC_CONFIG=%%C
)

if not "%HARDWARE_CLIENT_MODE%"=="" (
  if /I not "%HARDWARE_CLIENT_MODE%"=="grpc" (
    echo [WARN] HARDWARE_CLIENT_MODE=%HARDWARE_CLIENT_MODE% overridden to grpc
  )
)
set HARDWARE_CLIENT_MODE=grpc

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

set PYTHONUNBUFFERED=1

echo [INFO] Starting sidecar env=%SIDECAR_ENV% endpoint=%SIDECAR_HOST%:%SIDECAR_PORT%
start "DiFRA Sidecar" /B %CONDA_CMD% run --live-stream --no-capture-output -n %SIDECAR_ENV% python -u "%REPO_ROOT%\src\hardware\difra\scripts\pixet_sidecar_server.py" --host %SIDECAR_HOST% --port %SIDECAR_PORT%

echo [INFO] Starting gRPC env=%GRPC_ENV% endpoint=%GRPC_HOST%:%GRPC_PORT% config=%GRPC_CONFIG%
if "%GRPC_CONFIG%"=="" (
  start "DiFRA gRPC" /B %CONDA_CMD% run --live-stream --no-capture-output -n %GRPC_ENV% python -u "%REPO_ROOT%\src\hardware\difra\grpc_server\server.py" --host %GRPC_HOST% --port %GRPC_PORT%
) else (
  start "DiFRA gRPC" /B %CONDA_CMD% run --live-stream --no-capture-output -n %GRPC_ENV% python -u "%REPO_ROOT%\src\hardware\difra\grpc_server\server.py" --host %GRPC_HOST% --port %GRPC_PORT% --config "%GRPC_CONFIG%"
)

REM Wait for sidecar socket readiness
powershell -NoProfile -Command "$h='%SIDECAR_HOST%'; $p=[int]'%SIDECAR_PORT%'; $ok=$false; for($i=0;$i -lt 100;$i++){ try { $c=New-Object Net.Sockets.TcpClient; $c.Connect($h,$p); $c.Close(); $ok=$true; break } catch { Start-Sleep -Milliseconds 100 } }; if(-not $ok){ Write-Error \"Sidecar did not become ready at $h:$p\"; exit 1 }"
if errorlevel 1 exit /b 1

REM Wait for gRPC readiness
powershell -NoProfile -Command "$h='%GRPC_HOST%'; $p=[int]'%GRPC_PORT%'; $ok=$false; for($i=0;$i -lt 100;$i++){ try { $c=New-Object Net.Sockets.TcpClient; $c.Connect($h,$p); $c.Close(); $ok=$true; break } catch { Start-Sleep -Milliseconds 100 } }; if(-not $ok){ Write-Error \"gRPC did not become ready at $h:$p\"; exit 1 }"
if errorlevel 1 exit /b 1

set PIXET_BACKEND=sidecar
set DETECTOR_BACKEND=sidecar
set PIXET_SIDECAR_HOST=%SIDECAR_HOST%
set PIXET_SIDECAR_PORT=%SIDECAR_PORT%
set DIFRA_GRPC_HOST=%GRPC_HOST%
set DIFRA_GRPC_PORT=%GRPC_PORT%

echo [INFO] Starting DiFRA GUI env=%GUI_ENV% mode=%HARDWARE_CLIENT_MODE% grpc=%DIFRA_GRPC_HOST%:%DIFRA_GRPC_PORT% detector_backend=%DETECTOR_BACKEND%
%CONDA_CMD% run --live-stream --no-capture-output -n %GUI_ENV% python -u "%REPO_ROOT%\src\hardware\difra\gui\main_app.py" %*

endlocal
