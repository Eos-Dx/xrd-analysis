#!/usr/bin/env powershell
<#
.SYNOPSIS
    D2XC/EOSDxDc Software Launcher

.DESCRIPTION
    Launches the D2XC GUI application with proper environment setup.
    This script can be compiled to run_eosdxdc.exe using PS2EXE.

.EXAMPLE
    .\run_eosdxdc.ps1
#>

# Set error action preference for better error handling
$ErrorActionPreference = "Stop"

# Function to show error messages
function Show-ErrorDialog {
    param([string]$Title, [string]$Message)

    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show($Message, $Title, [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error)
    } catch {
        Write-Host "ERROR: $Message" -ForegroundColor Red
        Read-Host "Press Enter to exit"
    }
}

# Function to show info messages
function Show-InfoDialog {
    param([string]$Title, [string]$Message)

    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show($Message, $Title, [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information)
    } catch {
        Write-Host "INFO: $Message" -ForegroundColor Green
    }
}

try {
    # Hardcoded repository root (since this will be compiled to exe)
    $RepoRoot = "C:\dev\xrd-analysis"

    # Path to configuration file
    $ConfigPath = Join-Path $RepoRoot "src\hardware\eosdxdc\resources\config\global.json"

    if (!(Test-Path $ConfigPath)) {
        Show-ErrorDialog "Configuration Error" "Configuration file not found:`n$ConfigPath"
        exit 1
    }

    # Read conda environment from config
    try {
        $Config = Get-Content -Raw $ConfigPath | ConvertFrom-Json
        $CondaEnv = $Config.conda

        if (!$CondaEnv) {
            Show-ErrorDialog "Configuration Error" "'conda' field not found in $ConfigPath"
            exit 1
        }
    } catch {
        Show-ErrorDialog "Configuration Error" "Failed to read config: $($_.Exception.Message)"
        exit 1
    }

    # Check if conda is available
    try {
        $null = & conda --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Conda command failed"
        }
    } catch {
        Show-ErrorDialog "Conda Error" "'conda' was not found or not working properly.`n`nPlease ensure:`n• Anaconda/Miniconda is installed`n• Conda is available in your PATH`n• Or run this from an Anaconda Prompt"
        exit 1
    }

    # Path to main application
    $AppPath = Join-Path $RepoRoot "src\hardware\eosdxdc\gui\main_app.py"

    if (!(Test-Path $AppPath)) {
        Show-ErrorDialog "Application Error" "Application file not found:`n$AppPath"
        exit 1
    }

    # Show startup info
    Write-Host "Starting D2XC/EOSDxDc software..." -ForegroundColor Green
    Write-Host "Using conda environment: $CondaEnv" -ForegroundColor Cyan
    Write-Host "Repository root: $RepoRoot" -ForegroundColor Cyan
    Write-Host "Application path: $AppPath" -ForegroundColor Cyan

    # Launch the application
    try {
        $arguments = @('run', '-n', $CondaEnv, 'python', $AppPath) + $args

        # Start the process and wait briefly to check if it started successfully
        $process = Start-Process -FilePath 'conda' -ArgumentList $arguments -NoNewWindow -PassThru

        # Give it a moment to start
        Start-Sleep -Milliseconds 500

        if ($process.HasExited -and $process.ExitCode -ne 0) {
            Show-ErrorDialog "Launch Error" "Failed to start D2XC application.`nExit code: $($process.ExitCode)"
            exit 1
        }

        Write-Host "D2XC application launched successfully!" -ForegroundColor Green
        exit 0

    } catch {
        Show-ErrorDialog "Launch Error" "Failed to launch application: $($_.Exception.Message)"
        exit 1
    }

} catch {
    Show-ErrorDialog "Unexpected Error" "An unexpected error occurred: $($_.Exception.Message)"
    exit 1
}
