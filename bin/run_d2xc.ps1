#Requires -Version 5.1

<#
.SYNOPSIS
    D2XC Software Launcher

.DESCRIPTION
    Launches the D2XC GUI application with proper environment setup.
    Reads conda environment from configuration and launches the application.

.PARAMETER Arguments
    Additional arguments to pass to the application

.EXAMPLE
    .\run_d2xc.ps1

.EXAMPLE
    .\run_d2xc.ps1 --debug
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

# Set error action preference
$ErrorActionPreference = "Stop"

try {
    # Determine repository root
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Split-Path -Parent $ScriptDir

    # Path to configuration file
    $ConfigPath = Join-Path $RepoRoot "src\hardware\eosdxdc\resources\config\global.json"

    if (!(Test-Path $ConfigPath)) {
        throw "Configuration file not found: $ConfigPath"
    }

    # Read conda environment from config
    $Config = Get-Content -Raw $ConfigPath | ConvertFrom-Json
    $CondaEnv = $Config.conda

    if (!$CondaEnv) {
        throw "'conda' field not found in $ConfigPath"
    }

    # Check if conda is available
    try {
        conda --version | Out-Null
    } catch {
        throw "'conda' was not found. Please run from an Anaconda/Miniconda environment or add conda to PATH."
    }

    # Path to main application
    $AppPath = Join-Path $RepoRoot "src\hardware\eosdxdc\gui\main_app.py"

    if (!(Test-Path $AppPath)) {
        throw "Application file not found: $AppPath"
    }

    Write-Host "Starting D2XC software..." -ForegroundColor Green
    Write-Host "Using conda environment: $CondaEnv" -ForegroundColor Cyan
    Write-Host "Repository root: $RepoRoot" -ForegroundColor Cyan
    Write-Host "Application path: $AppPath" -ForegroundColor Cyan
    Write-Host ""

    # Build command arguments
    $CmdArgs = @('run', '-n', $CondaEnv, 'python', $AppPath) + $Arguments

    # Launch the application
    & conda @CmdArgs

} catch {
    Write-Error "Failed to launch D2XC software: $_"
    if ($Host.Name -eq "ConsoleHost") {
        Read-Host "Press Enter to exit"
    }
    exit 1
}
