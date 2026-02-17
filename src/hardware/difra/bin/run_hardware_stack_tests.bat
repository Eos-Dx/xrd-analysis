@echo off
setlocal enabledelayedexpansion

REM Determine repository root (four levels up: bin -> difra -> hardware -> src -> root)
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..\..\..") do set REPO_ROOT=%%~fI

set GLOBAL_CONFIG=%REPO_ROOT%\src\hardware\difra\resources\config\global.json

set CONDA_CMD=conda
where conda >nul 2>&1
if errorlevel 1 (
  echo [INFO] 'conda' not found in PATH, searching common installation locations...
  set CONDA_PATHS[0]=%USERPROFILE%\anaconda3
  set CONDA_PATHS[1]=%USERPROFILE%\miniconda3
  set CONDA_PATHS[2]=C:\ProgramData\Anaconda3
  set CONDA_PATHS[3]=C:\ProgramData\Miniconda3
  set CONDA_PATHS[4]=C:\Anaconda3
  set CONDA_PATHS[5]=C:\Miniconda3
  set CONDA_PATHS[6]=C:\Users\Ulster\anaconda3

  set CONDA_FOUND=0
  for /L %%i in (0,1,6) do (
    if defined CONDA_PATHS[%%i] (
      set CONDA_PATH=!CONDA_PATHS[%%i]!
      if exist "!CONDA_PATH!\Scripts\conda.exe" (
        set CONDA_CMD="!CONDA_PATH!\Scripts\conda.exe"
        echo [INFO] Found conda at: !CONDA_PATH!
        set CONDA_FOUND=1
        goto :conda_found
      )
    )
  )

  :conda_found
  if !CONDA_FOUND!==0 (
    echo [ERROR] Could not find conda installation.
    exit /b 1
  )
)

set GUI_ENV=%DIFRA_GUI_ENV%
if "%GUI_ENV%"=="" (
  for /f "usebackq delims=" %%E in (`powershell -NoProfile -Command "$p='%GLOBAL_CONFIG%'; if (Test-Path $p) { try { $g=(Get-Content -Raw $p | ConvertFrom-Json); [string]$g.conda } catch { '' } } else { '' }"`) do set GUI_ENV=%%E
)
if "%GUI_ENV%"=="" set GUI_ENV=eosdx13

if "%DIFRA_LEGACY_PYTHON%"=="" (
  if "%DIFRA_LEGACY_ENV%"=="" (
    for /f "usebackq delims=" %%B in (`%CONDA_CMD% info --base`) do set CONDA_BASE=%%B
    if defined CONDA_BASE (
      if exist "%CONDA_BASE%\envs\ulster37" (
        set DIFRA_LEGACY_ENV=ulster37
      ) else (
        if exist "%CONDA_BASE%\envs\ulster38" (
          set DIFRA_LEGACY_ENV=ulster38
        )
      )
    )

    if "%DIFRA_LEGACY_ENV%"=="" (
      echo [WARN] ulster37/ulster38 not found; tests will use current Python unless DIFRA_LEGACY_PYTHON is set.
    ) else (
      echo [INFO] Using legacy env: %DIFRA_LEGACY_ENV%
    )
  ) else (
    echo [INFO] Using requested legacy env: %DIFRA_LEGACY_ENV%
  )
) else (
  echo [INFO] Using explicit legacy python: %DIFRA_LEGACY_PYTHON%
)

if "%DIFRA_EXPECT_STAGE_TYPE%"=="" set DIFRA_EXPECT_STAGE_TYPE=Kinesis
if "%DIFRA_EXPECT_STAGE_CLASS%"=="" set DIFRA_EXPECT_STAGE_CLASS=XYStageLibController
if "%DIFRA_EXPECT_DETECTOR_CLASS%"=="" set DIFRA_EXPECT_DETECTOR_CLASS=PixetSidecarDetectorController

cd /d %REPO_ROOT%
set PYTHONUNBUFFERED=1
set PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%

echo [INFO] Running hardware stack tests in GUI env: %GUI_ENV%
echo [INFO] Expected route: stage_type=%DIFRA_EXPECT_STAGE_TYPE% stage_class=%DIFRA_EXPECT_STAGE_CLASS% detector_class=%DIFRA_EXPECT_DETECTOR_CLASS%

%CONDA_CMD% run --live-stream --no-capture-output -n %GUI_ENV% python -m pytest -q -s ^
  "%REPO_ROOT%\src\hardware\difra\tests\test_detector_integration_timing_e2e.py" ^
  "%REPO_ROOT%\src\hardware\difra\tests\manual_hardware_real_legacy_e2e.py"

set EXIT_CODE=%ERRORLEVEL%
endlocal & exit /b %EXIT_CODE%

