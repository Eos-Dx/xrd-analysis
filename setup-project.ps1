#!/usr/bin/env pwsh

Write-Host "🔍 Detecting Python version..." -ForegroundColor Blue

# Get Python version
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Python not found. Please install Python first." -ForegroundColor Red
        exit 1
    }

    Write-Host "📍 Found: $pythonVersion" -ForegroundColor Green

    # Extract major.minor version
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $majorVersion = [int]$matches[1]
        $minorVersion = [int]$matches[2]
        $shortVersion = "$majorVersion$minorVersion"

        Write-Host "🎯 Detected Python $majorVersion.$minorVersion" -ForegroundColor Cyan

        # Auto-select configuration based on Python version
        switch ($shortVersion) {
            "37" {
                Write-Host "⚙️  Configuring for Python 3.7..." -ForegroundColor Yellow
                Copy-Item "pyproject-py37.toml" "pyproject.toml" -Force
                Copy-Item "poetry-py37.lock" "poetry.lock" -Force
                $configType = "Python 3.7 (Legacy/d2xc compatible)"
            }
            "311" {
                Write-Host "⚙️  Configuring for Python 3.11..." -ForegroundColor Yellow
                Copy-Item "pyproject-py311.toml" "pyproject.toml" -Force
                Copy-Item "poetry-py311.lock" "poetry.lock" -Force
                $configType = "Python 3.11 (Modern)"
            }
            default {
                Write-Host "⚠️  Unsupported Python version: $majorVersion.$minorVersion" -ForegroundColor Yellow
                Write-Host "📋 Supported versions: 3.7, 3.11" -ForegroundColor Yellow

                # Prompt user to choose
                Write-Host "`n🤔 Which configuration would you like to use?"
                Write-Host "1) Python 3.7 configuration (Legacy/d2xc compatible)"
                Write-Host "2) Python 3.11 configuration (Modern)"

                do {
                    $choice = Read-Host "Enter choice (1 or 2)"
                } while ($choice -notin @("1", "2"))

                switch ($choice) {
                    "1" {
                        Copy-Item "pyproject-py37.toml" "pyproject.toml" -Force
                        Copy-Item "poetry-py37.lock" "poetry.lock" -Force
                        $configType = "Python 3.7 (Legacy/d2xc compatible)"
                    }
                    "2" {
                        Copy-Item "pyproject-py311.toml" "pyproject.toml" -Force
                        Copy-Item "poetry-py311.lock" "poetry.lock" -Force
                        $configType = "Python 3.11 (Modern)"
                    }
                }
            }
        }

        Write-Host "✅ Selected: $configType" -ForegroundColor Green

        # Check if Poetry is installed
        Write-Host "`n🔍 Checking Poetry installation..." -ForegroundColor Blue
        try {
            $poetryVersion = poetry --version 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw "Poetry not found"
            }
            Write-Host "📍 Found: $poetryVersion" -ForegroundColor Green
        }
        catch {
            Write-Host "⚠️  Poetry not found. Installing Poetry..." -ForegroundColor Yellow
            python -m pip install "poetry==1.4.2"
        }

        # Configure Poetry environment
        Write-Host "`n⚙️  Configuring Poetry environment..." -ForegroundColor Blue
        poetry env use python

        # Install dependencies
        Write-Host "`n📦 Installing dependencies..." -ForegroundColor Blue
        poetry install

        Write-Host "`n🎉 Setup complete!" -ForegroundColor Green
        Write-Host "📋 Configuration: $configType" -ForegroundColor Cyan
        Write-Host "`n🚀 You can now run your project with:" -ForegroundColor Blue
        Write-Host "   poetry run python -c `"import xrdanalysis; import hardware; print('All packages loaded successfully!')`"" -ForegroundColor Cyan

    } else {
        Write-Host "❌ Could not parse Python version: $pythonVersion" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "❌ Error detecting Python version: $_" -ForegroundColor Red
    exit 1
}
