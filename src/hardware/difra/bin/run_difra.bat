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

set SIDECAR_PY_EXE=
call :resolve_env_python "%SIDECAR_ENV%" SIDECAR_PY_EXE
if "%SIDECAR_PY_EXE%"=="" (
  echo [ERROR] Sidecar env '%SIDECAR_ENV%' is not available.
  echo [ERROR] Install/create legacy ulster37 or set DIFRA_SIDECAR_ENV.
  exit /b 1
)

set GRPC_PY_EXE=
call :resolve_env_python "%GRPC_ENV%" GRPC_PY_EXE
if "%GRPC_PY_EXE%"=="" (
  echo [ERROR] gRPC env '%GRPC_ENV%' is not available.
  exit /b 1
)

set GUI_PY_EXE=
call :resolve_env_python "%GUI_ENV%" GUI_PY_EXE
if "%GUI_PY_EXE%"=="" (
  echo [ERROR] GUI env '%GUI_ENV%' is not available.
  exit /b 1
)

for %%I in ("%SIDECAR_PY_EXE%") do set "SIDECAR_ENV_ROOT=%%~dpI"
for %%I in ("%GRPC_PY_EXE%") do set "GRPC_ENV_ROOT=%%~dpI"
for %%I in ("%GUI_PY_EXE%") do set "GUI_ENV_ROOT=%%~dpI"

set "ORIGINAL_PATH=%PATH%"
set "SIDECAR_LAUNCH_PATH=%SIDECAR_ENV_ROOT%;%SIDECAR_ENV_ROOT%Library\mingw-w64\bin;%SIDECAR_ENV_ROOT%Library\usr\bin;%SIDECAR_ENV_ROOT%Library\bin;%SIDECAR_ENV_ROOT%Scripts;%SIDECAR_ENV_ROOT%bin;%ORIGINAL_PATH%"
set "GRPC_LAUNCH_PATH=%GRPC_ENV_ROOT%;%GRPC_ENV_ROOT%Library\mingw-w64\bin;%GRPC_ENV_ROOT%Library\usr\bin;%GRPC_ENV_ROOT%Library\bin;%GRPC_ENV_ROOT%Scripts;%GRPC_ENV_ROOT%bin;%ORIGINAL_PATH%"
set "GUI_LAUNCH_PATH=%GUI_ENV_ROOT%;%GUI_ENV_ROOT%Library\mingw-w64\bin;%GUI_ENV_ROOT%Library\usr\bin;%GUI_ENV_ROOT%Library\bin;%GUI_ENV_ROOT%Scripts;%GUI_ENV_ROOT%bin;%ORIGINAL_PATH%"

cd /d %REPO_ROOT%
set PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%
set PYTHONUNBUFFERED=1

set SIDECAR_OWNER_WATCHDOG=%DIFRA_SIDECAR_OWNER_WATCHDOG%
if "%SIDECAR_OWNER_WATCHDOG%"=="" set SIDECAR_OWNER_WATCHDOG=0

set LAUNCHER_PID=
if "%SIDECAR_OWNER_WATCHDOG%"=="1" (
  for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process -Filter ('ProcessId=' + $PID); if($p -and $p.ParentProcessId){ $pp=Get-CimInstance Win32_Process -Filter ('ProcessId=' + $p.ParentProcessId); if($pp){ Write-Output $pp.ParentProcessId } }"`) do set LAUNCHER_PID=%%P
  if "%LAUNCHER_PID%"=="" (
    echo [WARN] Could not determine launcher PID; sidecar owner watchdog disabled.
  ) else (
    echo [INFO] Sidecar owner watchdog enabled ^(owner pid=%LAUNCHER_PID%^).
  )
) else (
  echo [INFO] Sidecar owner watchdog disabled on Windows ^(set DIFRA_SIDECAR_OWNER_WATCHDOG=1 to enable^).
)

if /I "%SIDECAR_HOST%"=="127.0.0.1" (
  powershell -NoProfile -Command "$p=[int]'%SIDECAR_PORT%'; try { $ids=Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique; foreach($id in $ids){ try { Stop-Process -Id $id -Force -ErrorAction Stop; Write-Host ('[INFO] Restarting detector sidecar: killed PID ' + $id + ' on port ' + $p) } catch {} } } catch {}"
) else if /I "%SIDECAR_HOST%"=="localhost" (
  powershell -NoProfile -Command "$p=[int]'%SIDECAR_PORT%'; try { $ids=Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique; foreach($id in $ids){ try { Stop-Process -Id $id -Force -ErrorAction Stop; Write-Host ('[INFO] Restarting detector sidecar: killed PID ' + $id + ' on port ' + $p) } catch {} } } catch {}"
) else (
  echo [WARN] Detector sidecar host is non-local ^(%SIDECAR_HOST%^) - skipping forced restart.
)

if /I "%GRPC_HOST%"=="127.0.0.1" (
  powershell -NoProfile -Command "$p=[int]'%GRPC_PORT%'; try { $ids=Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique; foreach($id in $ids){ try { Stop-Process -Id $id -Force -ErrorAction Stop; Write-Host ('[INFO] Restarting gRPC server: killed PID ' + $id + ' on port ' + $p) } catch {} } } catch {}"
) else if /I "%GRPC_HOST%"=="localhost" (
  powershell -NoProfile -Command "$p=[int]'%GRPC_PORT%'; try { $ids=Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique; foreach($id in $ids){ try { Stop-Process -Id $id -Force -ErrorAction Stop; Write-Host ('[INFO] Restarting gRPC server: killed PID ' + $id + ' on port ' + $p) } catch {} } } catch {}"
) else (
  echo [WARN] gRPC host is non-local ^(%GRPC_HOST%^) - skipping forced restart.
)

echo [INFO] Starting sidecar env=%SIDECAR_ENV% endpoint=%SIDECAR_HOST%:%SIDECAR_PORT%
set "PATH=%SIDECAR_LAUNCH_PATH%"
if "%LAUNCHER_PID%"=="" (
  start "DiFRA Sidecar" /B "%SIDECAR_PY_EXE%" -u "%REPO_ROOT%\src\hardware\difra\scripts\pixet_sidecar_server.py" --host %SIDECAR_HOST% --port %SIDECAR_PORT%
) else (
  start "DiFRA Sidecar" /B "%SIDECAR_PY_EXE%" -u "%REPO_ROOT%\src\hardware\difra\scripts\pixet_sidecar_server.py" --host %SIDECAR_HOST% --port %SIDECAR_PORT% --owner-pid %LAUNCHER_PID%
)
set "PATH=%ORIGINAL_PATH%"

echo [INFO] Starting gRPC env=%GRPC_ENV% endpoint=%GRPC_HOST%:%GRPC_PORT% config=%GRPC_CONFIG%
set "PATH=%GRPC_LAUNCH_PATH%"
if "%GRPC_CONFIG%"=="" (
  start "DiFRA gRPC" /B "%GRPC_PY_EXE%" -u "%REPO_ROOT%\src\hardware\difra\grpc_server\server.py" --host %GRPC_HOST% --port %GRPC_PORT%
) else (
  start "DiFRA gRPC" /B "%GRPC_PY_EXE%" -u "%REPO_ROOT%\src\hardware\difra\grpc_server\server.py" --host %GRPC_HOST% --port %GRPC_PORT% --config "%GRPC_CONFIG%"
)
set "PATH=%ORIGINAL_PATH%"

REM Wait for sidecar socket readiness
powershell -NoProfile -Command "$h='%SIDECAR_HOST%'; $p=[int]'%SIDECAR_PORT%'; $ok=$false; for($i=0;$i -lt 100;$i++){ try { $c=New-Object Net.Sockets.TcpClient; $c.Connect($h,$p); $c.Close(); $ok=$true; break } catch { Start-Sleep -Milliseconds 100 } }; if(-not $ok){ Write-Error \"Sidecar did not become ready at $($h):$p\"; exit 1 }"
if errorlevel 1 exit /b 1

REM Wait for gRPC readiness
powershell -NoProfile -Command "$h='%GRPC_HOST%'; $p=[int]'%GRPC_PORT%'; $ok=$false; for($i=0;$i -lt 100;$i++){ try { $c=New-Object Net.Sockets.TcpClient; $c.Connect($h,$p); $c.Close(); $ok=$true; break } catch { Start-Sleep -Milliseconds 100 } }; if(-not $ok){ Write-Error \"gRPC did not become ready at $($h):$p\"; exit 1 }"
if errorlevel 1 exit /b 1

set PIXET_BACKEND=sidecar
set DETECTOR_BACKEND=sidecar
set PIXET_SIDECAR_HOST=%SIDECAR_HOST%
set PIXET_SIDECAR_PORT=%SIDECAR_PORT%
set DIFRA_GRPC_HOST=%GRPC_HOST%
set DIFRA_GRPC_PORT=%GRPC_PORT%

echo [INFO] Starting DiFRA GUI env=%GUI_ENV% mode=%HARDWARE_CLIENT_MODE% grpc=%DIFRA_GRPC_HOST%:%DIFRA_GRPC_PORT% detector_backend=%DETECTOR_BACKEND%
set "PATH=%GUI_LAUNCH_PATH%"
"%GUI_PY_EXE%" -u "%REPO_ROOT%\src\hardware\difra\gui\main_app.py" %*
set "GUI_EXIT_CODE=%ERRORLEVEL%"
set "PATH=%ORIGINAL_PATH%"

call :stop_local_listener "%SIDECAR_HOST%" "%SIDECAR_PORT%" "Detector sidecar"
call :stop_local_listener "%GRPC_HOST%" "%GRPC_PORT%" "DiFRA gRPC server"

endlocal & exit /b %GUI_EXIT_CODE%

:resolve_env_python
setlocal
set "TARGET_ENV=%~1"
set "ENV_PATH="

for /f "tokens=1,2,3*" %%A in ('%CONDA_CMD% info --envs 2^>nul') do (
  if /I "%%A"=="%TARGET_ENV%" set "ENV_PATH=%%B"
  if /I "%%B"=="*" if /I "%%A"=="%TARGET_ENV%" set "ENV_PATH=%%C"
)

if defined ENV_PATH (
  if exist "%ENV_PATH%\python.exe" (
    endlocal & set "%~2=%ENV_PATH%\python.exe" & exit /b 0
  )
)

endlocal & set "%~2=" & exit /b 0

:stop_local_listener
setlocal
set "TARGET_HOST=%~1"
set "TARGET_PORT=%~2"
set "TARGET_LABEL=%~3"

if /I "%TARGET_HOST%"=="127.0.0.1" goto :do_stop_listener
if /I "%TARGET_HOST%"=="localhost" goto :do_stop_listener
if /I "%TARGET_HOST%"=="::1" goto :do_stop_listener
if /I "%TARGET_HOST%"=="0.0.0.0" goto :do_stop_listener
echo [WARN] %TARGET_LABEL% host is non-local ^(%TARGET_HOST%^) - skipping stop.
endlocal & exit /b 0

:do_stop_listener
powershell -NoProfile -Command "$p=[int]'%TARGET_PORT%'; try { $ids=Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique; foreach($id in $ids){ try { Stop-Process -Id $id -Force -ErrorAction Stop; Write-Host ('[INFO] Stopped process PID ' + $id + ' on port ' + $p) } catch {} } } catch {}"
endlocal & exit /b 0
