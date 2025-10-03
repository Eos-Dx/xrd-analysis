# Geant4 Installation Guide (Windows)

## Prerequisites

### Required Software
1. **Visual Studio 2019 or 2022** with C++ workload
2. **CMake 3.22 or later**
3. **Git** (if not already installed)

### Optional (Recommended)
4. **Qt5 or Qt6** for GUI visualization
5. **Python 3.8+** for data analysis scripts

## Installing Geant4

### Option 1: Pre-built Binaries (Recommended)
1. Download from: https://geant4.web.cern.ch/download
2. Choose Windows binaries with Qt visualization
3. Extract to `C:\geant4\11.1.0\` (or your preferred location)
4. Set environment variable:
   ```powershell
   $env:GEANT4_DIR = "C:\geant4\11.1.0\lib\Geant4-11.1.0"
   ```

### Option 2: Build from Source
If you need specific features or the latest version:

1. **Download source code**:
   ```powershell
   git clone https://github.com/Geant4/geant4.git
   cd geant4
   git checkout v11.1.0  # or desired version
   ```

2. **Create build directory**:
   ```powershell
   mkdir build-geant4
   cd build-geant4
   ```

3. **Configure with CMake**:
   ```powershell
   cmake -G "Visual Studio 17 2022" -A x64 `
         -DGEANT4_INSTALL_DATA=ON `
         -DGEANT4_USE_QT=ON `
         -DGEANT4_USE_OPENGL_WIN32=ON `
         -DGEANT4_USE_RAYTRACER_X11=OFF `
         -DCMAKE_INSTALL_PREFIX=C:\geant4\11.1.0 `
         ..
   ```

4. **Build and Install**:
   ```powershell
   cmake --build . --config Release --target install
   ```

5. **Set environment variable**:
   ```powershell
   $env:GEANT4_DIR = "C:\geant4\11.1.0\lib\Geant4-11.1.0"
   ```

## Setting Up Environment

### Permanent Environment Variables
Add to your PowerShell profile (`$PROFILE`):

```powershell
# Geant4 setup
$env:GEANT4_DIR = "C:\geant4\11.1.0\lib\Geant4-11.1.0"
$env:G4NEUTRONHPDATA = "C:\geant4\11.1.0\share\Geant4-11.1.0\data\G4NDL4.6"
# Add other G4 data variables as needed
```

Or set via Windows System Properties:
1. Right-click "This PC" → Properties → Advanced System Settings
2. Click "Environment Variables"
3. Add `GEANT4_DIR` with the path to your Geant4 CMake files

## Verification

Test your installation:

```powershell
# Check CMake can find Geant4
cmake --find-package -DNAME=Geant4 -DCOMPILER_ID=MSVC -DLANGUAGE=CXX -DMODE=EXIST

# Or check environment
Write-Host "Geant4 directory: $env:GEANT4_DIR"
Test-Path "$env:GEANT4_DIR\Geant4Config.cmake"
```

## Troubleshooting

### Common Issues

1. **"Could not find Geant4"**
   - Verify `GEANT4_DIR` points to the lib/Geant4-X.Y.Z directory
   - Check that `Geant4Config.cmake` exists in that directory

2. **"Qt not found" warnings**
   - Install Qt5/Qt6 development libraries
   - Or disable Qt: `-DGEANT4_USE_QT=OFF` during configuration

3. **Missing data files**
   - Download Geant4 data files separately if needed
   - Set G4 data environment variables (G4NEUTRONHPDATA, etc.)

4. **Visual Studio version mismatch**
   - Ensure CMake generator matches your VS version
   - Use `-G "Visual Studio 16 2019"` for VS 2019

### Getting Help

- **Geant4 Documentation**: https://geant4-userdoc.web.cern.ch/
- **Installation Guide**: https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/
- **Geant4 Forum**: https://geant4-forum.web.cern.ch/
