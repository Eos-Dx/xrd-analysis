#!/usr/bin/env pwsh
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("37", "311", "313")]
    [string]$PythonVersion
)

Write-Host "Switching to Python $PythonVersion configuration..." -ForegroundColor Green

switch ($PythonVersion) {
    "37" {
        Copy-Item "pyproject-py37.toml" "pyproject.toml" -Force
        Copy-Item "poetry-py37.lock" "poetry.lock" -Force
        Write-Host "✅ Switched to Python 3.7 configuration" -ForegroundColor Green
        Write-Host "📝 Remember to activate your Python 3.7 conda environment:" -ForegroundColor Yellow
        Write-Host "   conda activate ulster37" -ForegroundColor Cyan
        Write-Host "   poetry env use python" -ForegroundColor Cyan
    }
    "311" {
        Copy-Item "pyproject-py311.toml" "pyproject.toml" -Force
        Copy-Item "poetry-py311.lock" "poetry.lock" -Force
        Write-Host "✅ Switched to the modern Python configuration (compatible with 3.11-3.13)" -ForegroundColor Green
        Write-Host "📝 Recommended Conda environment:" -ForegroundColor Yellow
        Write-Host "   conda activate eosdx13" -ForegroundColor Cyan
        Write-Host "   poetry env use python" -ForegroundColor Cyan
    }
    "313" {
        Copy-Item "pyproject-py311.toml" "pyproject.toml" -Force
        Copy-Item "poetry-py311.lock" "poetry.lock" -Force
        Write-Host "✅ Switched to the modern Python configuration (compatible with 3.11-3.13)" -ForegroundColor Green
        Write-Host "📝 Recommended Conda environment:" -ForegroundColor Yellow
        Write-Host "   conda activate eosdx13" -ForegroundColor Cyan
        Write-Host "   poetry env use python" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "🚀 Now you can run:" -ForegroundColor Blue
Write-Host "   poetry install" -ForegroundColor Cyan
