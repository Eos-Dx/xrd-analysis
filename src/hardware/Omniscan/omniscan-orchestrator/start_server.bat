@echo off
echo Starting Omniscan Orchestrator REST API...
echo Press Ctrl+C to stop
echo.
echo Installing in editable mode (using local source code)...
call conda activate eosdx
pip install -e . >nul 2>&1
echo Starting server with --reload (auto-restart on code changes)...
echo.
python -m uvicorn omniscan_orchestrator.rest_server:app --host 0.0.0.0 --port 8081 --reload
