@echo off
setlocal

REM Keep embedded launcher protocol-consistent with main launcher:
REM GUI -> gRPC server -> legacy PIXet sidecar.
set SCRIPT_DIR=%~dp0
call "%SCRIPT_DIR%run_difra.bat" %*
set EXIT_CODE=%ERRORLEVEL%

endlocal & exit /b %EXIT_CODE%
