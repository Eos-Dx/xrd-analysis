# Warp guidance (xrd-analysis)

This document provides guidance and curated tasks for Warp when working on this repository.

Overview
- Python monorepo layout under `src/` with major areas:
  - `xrdanalysis`: core analysis library for X-ray scattering
  - `quality_control`: PyQt5 quality-control GUI (independent package)
  - `hardware/eosdxdc`: EOSDxDc hardware control and PyQt5 GUI (replaces legacy `Ulster`)
- Tests live under `src/**/tests`, including headless tests for the hardware GUI.

Environment setup
- Conda environment (recommended):
  ```pwsh
  conda env create -f environment.yml
  conda activate eosdx
  ```
- Editable install and hooks:
  ```pwsh
  pip install -e .
  pre-commit install
  ```

Common tasks
- Lint/format (pre-commit):
  ```pwsh
  pre-commit run --all-files
  ```
- Run all tests:
  ```pwsh
  pytest -q
  ```
- Run hardware GUI tests only:
  ```pwsh
  pytest src/hardware/eosdxdc/tests -q
  ```
- Build docs:
  ```pwsh
  sphinx-build docs _build
  # or
  make -C docs html
  ```

Hardware GUI (EOSDxDc)
- Launcher (Windows):
  - `bin\run_eosdxdc.bat`
  - The launcher reads the conda env name from `src/hardware/eosdxdc/resources/config/main.json` (`"conda"`) and executes:
    - `conda run -n <env> python src/hardware/eosdxdc/gui/main_app.py`
- Config: `src/hardware/eosdxdc/resources/config/main.json`
  - Define active detectors/stages and the conda env name

Notes
- Legacy `src/hardware/Ulster` has been retired in favor of `src/hardware/eosdxdc`.
- pytest is configured for importlib mode to avoid src layout conflicts.
