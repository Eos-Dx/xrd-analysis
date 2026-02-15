#!/bin/bash
echo "Starting Omniscan Orchestrator REST API..."
echo "Press Ctrl+C to stop"
echo ""
echo "Installing in editable mode (using local source code)..."
eval "$(conda shell.bash hook)"
conda activate eosdx
pip install -e . > /dev/null 2>&1
echo "Starting server with --reload (auto-restart on code changes)..."
echo ""
# Kill any existing process on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
python -m uvicorn omniscan_orchestrator.rest_server:app --host 0.0.0.0 --port 8080 --reload
