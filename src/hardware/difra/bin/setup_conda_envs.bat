@echo off
setlocal enabledelayedexpansion

REM Determine repository root (four levels up: bin -> difra -> hardware -> src -> root)
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..\..\..") do set REPO_ROOT=%%~fI

set DIFRA_DIR=%REPO_ROOT%\src\hardware\difra
set YAML_38=%DIFRA_DIR%\environment-ulster38.yml
set YAML_311=%DIFRA_DIR%\environment-ulster311.yml
set STUBS_SCRIPT=%REPO_ROOT%\src\hardware\protocol\scripts\generate_python_stubs.py

if not exist "%YAML_38%" (
  echo [ERROR] Missing YAML file: %YAML_38%
  exit /b 1
)
if not exist "%YAML_311%" (
  echo [ERROR] Missing YAML file: %YAML_311%
  exit /b 1
)
if not exist "%STUBS_SCRIPT%" (
  echo [ERROR] Missing stub generation script: %STUBS_SCRIPT%
  exit /b 1
)

call :find_conda
if errorlevel 1 exit /b 1

echo.
echo ======================================
echo   DiFRA Conda Environment Installer
echo ======================================
echo 1. Install/Update ulster38  (Python 3.8)
echo 2. Install/Update ulster311 (Python 3.11)
echo 3. Install/Update both
echo Q. Quit
echo.
set /p CHOICE=Choose option [1/2/3/Q]:

if /I "%CHOICE%"=="Q" (
  echo [INFO] Cancelled by user.
  exit /b 0
)

if "%CHOICE%"=="1" (
  call :apply_env ulster38 "%YAML_38%" || exit /b 1
  goto :ask_regen
)
if "%CHOICE%"=="2" (
  call :apply_env ulster311 "%YAML_311%" || exit /b 1
  goto :ask_regen
)
if "%CHOICE%"=="3" (
  call :apply_env ulster38 "%YAML_38%" || exit /b 1
  call :apply_env ulster311 "%YAML_311%" || exit /b 1
  goto :ask_regen
)

echo [ERROR] Invalid choice: %CHOICE%
exit /b 1

:ask_regen
echo.
set /p REGEN=Regenerate gRPC/protobuf stubs now? [y/N]:
if /I not "%REGEN%"=="Y" goto :done

if "%CHOICE%"=="1" (
  call :regen_stubs ulster38 || exit /b 1
  goto :done
)
if "%CHOICE%"=="2" (
  call :regen_stubs ulster311 || exit /b 1
  goto :done
)

echo.
echo Which env should regenerate stubs?
echo 1. ulster38
echo 2. ulster311
set /p REGEN_ENV=Choose [1/2]:
if "%REGEN_ENV%"=="1" (
  call :regen_stubs ulster38 || exit /b 1
  goto :done
)
if "%REGEN_ENV%"=="2" (
  call :regen_stubs ulster311 || exit /b 1
  goto :done
)
echo [ERROR] Invalid choice: %REGEN_ENV%
exit /b 1

:done
echo.
echo [INFO] Completed.
exit /b 0

:apply_env
set ENV_NAME=%~1
set ENV_FILE=%~2
echo.
echo [INFO] Processing %ENV_NAME% using %ENV_FILE%

%CONDA_CMD% run -n %ENV_NAME% python -V >nul 2>&1
if errorlevel 1 (
  echo [INFO] Environment %ENV_NAME% not found. Creating...
  %CONDA_CMD% env create -f "%ENV_FILE%"
) else (
  echo [INFO] Environment %ENV_NAME% exists. Updating...
  %CONDA_CMD% env update -n %ENV_NAME% -f "%ENV_FILE%" --prune
)
if errorlevel 1 (
  echo [ERROR] Failed to install/update %ENV_NAME%
  exit /b 1
)
echo [INFO] %ENV_NAME% ready.
exit /b 0

:regen_stubs
set ENV_NAME=%~1
echo.
echo [INFO] Regenerating protobuf/grpc stubs with %ENV_NAME%...
%CONDA_CMD% run -n %ENV_NAME% python "%STUBS_SCRIPT%"
if errorlevel 1 (
  echo [ERROR] Stub generation failed in %ENV_NAME%
  exit /b 1
)
echo [INFO] Stub generation completed in %ENV_NAME%.
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
