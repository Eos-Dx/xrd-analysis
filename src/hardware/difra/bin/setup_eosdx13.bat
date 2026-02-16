@echo off
setlocal enabledelayedexpansion

REM Determine repository root (four levels up: bin -> difra -> hardware -> src -> root)
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..\..\..") do set REPO_ROOT=%%~fI

set DIFRA_DIR=%REPO_ROOT%\src\hardware\difra
set YAML_313=%DIFRA_DIR%\environment-eosdx13.yml
set ENV_NAME=eosdx13

if not exist "%YAML_313%" (
  echo [ERROR] Missing YAML file: %YAML_313%
  exit /b 1
)

call :find_conda
if errorlevel 1 exit /b 1

echo.
echo ======================================
echo    DiFRA EOSDX13 Environment Setup
echo ======================================
echo Environment: %ENV_NAME%
echo Spec file  : %YAML_313%
echo.

%CONDA_CMD% run -n %ENV_NAME% python -V >nul 2>&1
if errorlevel 1 (
  echo [INFO] Environment %ENV_NAME% not found. Creating...
  %CONDA_CMD% env create -f "%YAML_313%"
) else (
  echo [INFO] Environment %ENV_NAME% exists. Updating...
  %CONDA_CMD% env update -n %ENV_NAME% -f "%YAML_313%" --prune
)
if errorlevel 1 (
  echo [ERROR] Failed to create/update %ENV_NAME%.
  exit /b 1
)

echo.
echo [INFO] %ENV_NAME% is ready.
set /p RUN_TESTS=Run DiFRA test suite now? [y/N]:
if /I not "%RUN_TESTS%"=="Y" goto :done

set QT_QPA_PLATFORM=offscreen
set PYTHONPATH=%REPO_ROOT%\src
echo [INFO] Running tests in %ENV_NAME%...
%CONDA_CMD% run -n %ENV_NAME% python -m pytest "%REPO_ROOT%\src\hardware\difra\tests"
if errorlevel 1 (
  echo [ERROR] Tests failed.
  exit /b 1
)
echo [INFO] Tests finished successfully.

:done
echo [INFO] Completed.
exit /b 0

:find_conda
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
exit /b 0
