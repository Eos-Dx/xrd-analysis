@echo off
REM Start Omniscan Hardware Server in Demo Mode
REM This batch file starts the server in the current terminal window

echo ========================================
echo  Omniscan Hardware Server - Demo Mode
echo ========================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

echo Starting server with demo configuration...
echo Config: configs\config_demo_safe.json
echo.

REM Run the server
cargo run --bin omniscan-hw-server -- --config configs\config_demo_safe.json

REM If server exits, pause to see error messages
echo.
echo ========================================
echo Server stopped.
echo ========================================
pause
