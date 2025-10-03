#!/usr/bin/env pwsh
# PowerShell script to configure and build Geant4 simulations on Windows

param(
    [string]$BuildType = "Release",
    [string]$Generator = "Visual Studio 17 2022",
    [switch]$Clean = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "Usage: ./setup-geant4.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -BuildType    Build configuration (Debug/Release, default: Release)"
    Write-Host "  -Generator    CMake generator (default: 'Visual Studio 17 2022')"
    Write-Host "  -Clean        Clean build directory before configuring"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
    Write-Host "Prerequisites:"
    Write-Host "  - Geant4 installed with GEANT4_DIR environment variable set"
    Write-Host "  - Visual Studio 2019 or 2022 with C++ workload"
    Write-Host "  - CMake 3.22 or later"
    exit 0
}

# Check prerequisites
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "CMake not found in PATH. Please install CMake 3.22 or later."
    exit 1
}

if (-not $env:GEANT4_DIR) {
    Write-Warning "GEANT4_DIR environment variable not set."
    Write-Host "Please set GEANT4_DIR to your Geant4 installation directory."
    Write-Host "Example: `$env:GEANT4_DIR = 'C:\geant4\11.1.0\lib\Geant4-11.1.0'"
    exit 1
}

Write-Host "Setting up Geant4 simulations build..." -ForegroundColor Green
Write-Host "Build Type: $BuildType" -ForegroundColor Cyan
Write-Host "Generator: $Generator" -ForegroundColor Cyan
Write-Host "Geant4 Dir: $env:GEANT4_DIR" -ForegroundColor Cyan
Write-Host ""

# Set up paths
$RootDir = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RootDir "build\windows-$($BuildType.ToLower())"
$InstallDir = Join-Path $RootDir "install\$($BuildType.ToLower())"

Write-Host "Root Directory: $RootDir"
Write-Host "Build Directory: $BuildDir"
Write-Host "Install Directory: $InstallDir"
Write-Host ""

# Clean build directory if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

# Create build directory
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# Configure with CMake
Write-Host "Configuring with CMake..." -ForegroundColor Green
Push-Location $BuildDir
try {
    $cmakeArgs = @(
        "-G", $Generator,
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_INSTALL_PREFIX=$InstallDir",
        $RootDir
    )

    Write-Host "cmake $($cmakeArgs -join ' ')" -ForegroundColor Gray
    & cmake @cmakeArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake configuration failed!"
        exit 1
    }

    Write-Host ""
    Write-Host "Configuration successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Build: cmake --build . --config $BuildType"
    Write-Host "  2. Or use: ./run-example.ps1 example1"
    Write-Host "  3. Or open: $(Get-ChildItem *.sln | Select-Object -First 1 -ExpandProperty Name)"

} finally {
    Pop-Location
}
