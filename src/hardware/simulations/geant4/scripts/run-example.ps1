#!/usr/bin/env pwsh
# PowerShell script to build and run Geant4 examples

param(
    [string]$Example = "example1",
    [string]$BuildType = "Release",
    [string]$Macro = "",
    [switch]$Build = $true,
    [switch]$Interactive = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "Usage: ./run-example.ps1 [options] [example_name]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Example      Example to run (default: example1)"
    Write-Host "  -BuildType    Build configuration (Debug/Release, default: Release)"
    Write-Host "  -Macro        Macro file to execute (default: run1.mac for batch mode)"
    Write-Host "  -Build        Build before running (default: true)"
    Write-Host "  -Interactive  Run in interactive mode (default: false)"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  ./run-example.ps1 example1                    # Build and run in batch mode"
    Write-Host "  ./run-example.ps1 -Interactive               # Run in interactive mode with visualization"
    Write-Host "  ./run-example.ps1 -Macro run1.mac            # Run specific macro"
    Write-Host "  ./run-example.ps1 -Build:`$false example1      # Run without building"
    exit 0
}

# Set up paths
$RootDir = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RootDir "build\windows-$($BuildType.ToLower())"
$ExampleDir = Join-Path $RootDir "projects\$Example"

Write-Host "Running Geant4 Example: $Example" -ForegroundColor Green
Write-Host "Build Type: $BuildType" -ForegroundColor Cyan
Write-Host "Interactive Mode: $Interactive" -ForegroundColor Cyan
Write-Host ""

# Check if example exists
if (-not (Test-Path $ExampleDir)) {
    Write-Error "Example '$Example' not found at: $ExampleDir"
    Write-Host "Available examples:"
    Get-ChildItem -Path (Join-Path $RootDir "projects") -Directory | ForEach-Object {
        Write-Host "  - $($_.Name)"
    }
    exit 1
}

# Check if build directory exists
if (-not (Test-Path $BuildDir)) {
    Write-Warning "Build directory not found: $BuildDir"
    Write-Host "Please run ./setup-geant4.ps1 first to configure the build."
    exit 1
}

# Build if requested
if ($Build) {
    Write-Host "Building..." -ForegroundColor Green
    Push-Location $BuildDir
    try {
        & cmake --build . --config $BuildType --target $Example
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Build failed!"
            exit 1
        }
        Write-Host "Build successful!" -ForegroundColor Green
        Write-Host ""
    } finally {
        Pop-Location
    }
}

# Find the executable
$ExeDir = Join-Path $BuildDir "projects\$Example\$BuildType"
$ExePath = Join-Path $ExeDir "$Example.exe"

if (-not (Test-Path $ExePath)) {
    # Try without BuildType subdirectory (for single-config generators)
    $ExeDir = Join-Path $BuildDir "projects\$Example"
    $ExePath = Join-Path $ExeDir "$Example.exe"
}

if (-not (Test-Path $ExePath)) {
    Write-Error "Executable not found: $ExePath"
    Write-Host "Please ensure the build was successful."
    exit 1
}

# Change to executable directory (so macro files are found)
Push-Location $ExeDir
try {
    Write-Host "Executable: $ExePath" -ForegroundColor Cyan
    Write-Host "Working Directory: $ExeDir" -ForegroundColor Cyan
    Write-Host ""

    if ($Interactive) {
        Write-Host "Starting interactive mode..." -ForegroundColor Green
        Write-Host "Use 'exit' command to quit the simulation." -ForegroundColor Yellow
        Write-Host ""
        & $ExePath
    } else {
        # Batch mode
        if (-not $Macro) {
            $Macro = "run1.mac"
        }

        $MacroPath = Join-Path $ExeDir $Macro
        if (-not (Test-Path $MacroPath)) {
            Write-Error "Macro file not found: $MacroPath"
            Write-Host "Available macro files:"
            Get-ChildItem -Path $ExeDir -Filter "*.mac" | ForEach-Object {
                Write-Host "  - $($_.Name)"
            }
            exit 1
        }

        Write-Host "Running macro: $Macro" -ForegroundColor Green
        Write-Host ""
        & $ExePath $Macro
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Simulation completed successfully!" -ForegroundColor Green
    } else {
        Write-Error "Simulation failed with exit code: $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
