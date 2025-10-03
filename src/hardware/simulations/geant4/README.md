# Geant4 Monte Carlo Simulations

Geant4-based Monte Carlo simulations for X-ray detector hardware development.

## Prerequisites
- Geant4 11.1+ with Qt5/Qt6 visualization (recommended)
- CMake 3.22+
- Visual Studio 2019+ with C++17 support

## Structure
```
geant4/
├── projects/           # Individual simulation projects
│   └── example1/       # Basic detector simulation example
├── common/            # Shared utilities and base classes
├── data/              # Cross-section data, materials
├── macros/            # Common macro files
├── scripts/           # Build and run helper scripts
└── docs/              # Documentation
```

## Quick Start
1. Install Geant4 (see docs/INSTALL.md)
2. Run setup script:
   ```powershell
   .\scripts\setup-geant4.ps1
   ```
3. Build and run example:
   ```powershell
   .\scripts\run-example.ps1 example1
   ```

## Projects
- `example1/` — Basic water phantom with scoring volumes
