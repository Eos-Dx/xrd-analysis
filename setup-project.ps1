#!/usr/bin/env pwsh

# Helper: Check last exit code and fail with message
function Assert-Success($Message) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ $Message" -ForegroundColor Red
        exit 1
    }
}

Write-Host "🔧 Project setup" -ForegroundColor Blue

# 1) Ensure Conda is available
Write-Host "\n🔍 Checking for Conda..." -ForegroundColor Blue
$condaCmd = $null
try {
    $null = conda --version 2>$null
    if ($LASTEXITCODE -eq 0) { $condaCmd = "conda" }
} catch {}
if (-not $condaCmd -and $env:CONDA_EXE) { $condaCmd = $env:CONDA_EXE }
if (-not $condaCmd) {
    Write-Host "❌ Conda not found. Please install Anaconda or Miniconda and ensure 'conda' is on PATH." -ForegroundColor Red
    exit 1
}
Write-Host "📍 Using Conda: $condaCmd" -ForegroundColor Green

# 2) Prompt for environment name
$defaultEnv = "ulster37"
$envName = Read-Host "\n🧪 Enter Conda environment name [default: $defaultEnv]"
if ([string]::IsNullOrWhiteSpace($envName)) { $envName = $defaultEnv }
Write-Host "📦 Environment will be: $envName" -ForegroundColor Cyan

# 3) Choose Python version (3.7 or 3.11)
Write-Host "\n🤔 Choose Python version:" -ForegroundColor Blue
Write-Host "1) Python 3.7 (Legacy/d2xc compatible)"
Write-Host "2) Python 3.11 (Modern)"
$pyChoice = $null
do {
    $pyChoice = Read-Host "Enter choice (1 or 2)"
} while ($pyChoice -notin @("1","2"))

switch ($pyChoice) {
    "1" { $pyVersion = "3.7"; $shortVersion = "37"; $configType = "Python 3.7 (Legacy/d2xc compatible)" }
    "2" { $pyVersion = "3.11"; $shortVersion = "311"; $configType = "Python 3.11 (Modern)" }
}

# 4) Create Conda environment (if missing)
Write-Host "\n🏗️  Creating Conda env '$envName' with Python $pyVersion (if it doesn't exist)..." -ForegroundColor Blue
# Check if env exists
$envsList = & $condaCmd env list --json 2>$null | Out-String
$exists = $false
try {
    $envs = ($envsList | ConvertFrom-Json).envs
    if ($envs -ne $null) {
        foreach ($p in $envs) { if ($p -match "[\\/]$envName$") { $exists = $true; break } }
    }
} catch {}
if (-not $exists) {
    & $condaCmd create -y -n $envName "python=$pyVersion"
    Assert-Success "Failed to create Conda environment '$envName' with Python $pyVersion."
} else {
    Write-Host "✅ Env '$envName' already exists." -ForegroundColor Green
}

# 5) Select pyproject/lock based on chosen Python
switch ($shortVersion) {
    "37" {
        Write-Host "⚙️  Configuring for Python 3.7..." -ForegroundColor Yellow
        Copy-Item "pyproject-py37.toml" "pyproject.toml" -Force
        Copy-Item "poetry-py37.lock" "poetry.lock" -Force
    }
    "311" {
        Write-Host "⚙️  Configuring for Python 3.11..." -ForegroundColor Yellow
        Copy-Item "pyproject-py311.toml" "pyproject.toml" -Force
        Copy-Item "poetry-py311.lock" "poetry.lock" -Force
    }
}
Write-Host "✅ Selected: $configType" -ForegroundColor Green

# 6) Ensure Poetry inside the Conda env
Write-Host "\n🔍 Ensuring Poetry is available in env '$envName'..." -ForegroundColor Blue
& $condaCmd run -n $envName python -m pip show poetry *> $null
if ($LASTEXITCODE -ne 0) {
    & $condaCmd run -n $envName python -m pip install "poetry==1.4.2"
    Assert-Success "Failed to install Poetry in env '$envName'."
}
& $condaCmd run -n $envName poetry --version
Assert-Success "Poetry not working inside env '$envName'."

# 7) Install dependencies with Poetry in that env
Write-Host "\n📦 Installing dependencies via Poetry (env: $envName)..." -ForegroundColor Blue
& $condaCmd run -n $envName poetry env use python
Assert-Success "Failed to set Poetry interpreter to env Python."
& $condaCmd run -n $envName poetry install
Assert-Success "Failed to install project dependencies with Poetry."

Write-Host "\n🎉 Setup complete!" -ForegroundColor Green
Write-Host "📋 Configuration: $configType" -ForegroundColor Cyan
Write-Host "\n🚀 You can now run your project with:" -ForegroundColor Blue
Write-Host "   conda run -n $envName poetry run python -c `"import xrdanalysis; import hardware; print('All packages loaded successfully!')`"" -ForegroundColor Cyan
