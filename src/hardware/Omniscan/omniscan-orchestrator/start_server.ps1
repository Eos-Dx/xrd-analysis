# Start Omniscan Orchestrator REST API Server
# This script starts the FastAPI server that bridges Web UI with hardware server

Write-Host "🚀 Starting Omniscan Orchestrator REST API Server..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "⚠️  Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Cyan
& .venv\Scripts\Activate.ps1

# Install/upgrade dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
pip install -e .

Write-Host ""
Write-Host "🌐 Starting REST API server on http://localhost:8080" -ForegroundColor Green
Write-Host "📡 WebSocket available on ws://localhost:8080/ws" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start uvicorn server
python -m uvicorn omniscan_orchestrator.rest_server:app --host 0.0.0.0 --port 8080 --reload
