# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository: xrd-analysis (Python)

Overview
- Mono-repo style Python project under src/ with three major areas:
  - xrdanalysis: core analysis library for X-ray scattering data. Provides pyFAI-based azimuthal integration, feature extraction (Fourier, profiles), sklearn-compatible transformers and training/validation utilities.
  - quality_control (eosdx_quality_tool): a PyQt5 desktop GUI for quality control workflows. Packaged independently with its own console entrypoint.
  - hardware/Ulster: hardware control and a PyQt5 GUI (stage control, acquisition, visualization). Includes resources (configuration, PONI examples) and tests that stub heavy GUI dependencies.
- Tests live in src/tests (library) and src/hardware/Ulster/tests (hardware GUI), plus test_environment.py at repo root.
- Sphinx documentation under docs/. GitHub Actions builds docs and publishes to gh-pages on pushes to main.

Environment setup
- Conda environment (recommended):
  ```pwsh
  conda env create -f environment.yml
  conda activate eosdx
  ```
- Development install of the core packages:
  ```pwsh
  pip install -e .
  ```
- Enable pre-commit hooks (formatting, linting, doc coverage checks):
  ```pwsh
  pre-commit install
  # Run on entire repo when needed
  pre-commit run --all-files
  ```

Build, lint, and tests
- Lint (two options):
  - Via pre-commit (recommended, enforces black, isort, flake8, interrogate on staged files):
    ```pwsh
    pre-commit run --all-files
    ```
  - Direct flake8 (Makefile provides a lint target, but using flake8 directly is cross-platform):
    ```pwsh
    flake8 src
    ```
- Run all tests:
  ```pwsh
  pytest
  ```
- Run a single test file:
  ```pwsh
  pytest src/tests/data_processing/test_utility_functions.py
  ```
- Run a specific test function:
  ```pwsh
  pytest src/tests/data_processing/test_utility_functions.py::test_get_center
  ```
- Run only hardware GUI tests (these stub PyQt5, no UI needed):
  ```pwsh
  pytest src/hardware/Ulster/tests -q
  ```

Documentation
- Build Sphinx docs locally:
  ```pwsh
  sphinx-build docs _build
  ```
  Alternatively, if make is available:
  ```pwsh
  make -C docs html
  ```
- CI notes: .github/workflows/documentation.yml sets up the conda env (eosdx), runs pip install -e ., builds with sphinx-build docs _build, and publishes _build/ to gh-pages on pushes to main.

Subpackages and entrypoints
- Core analysis library (src/xrdanalysis)
  - data_processing.azimuthal_integration: wraps pyFAI AzimuthalIntegrator and related workflows; supports modes (1D, 2D, sigma_clip, rotating_angles) and PONI- or DataFrame-based calibration.
  - data_processing.transformers: sklearn TransformerMixin implementations for integration and deviation computation; integrates with pipelines.
  - data_processing.utility_functions: HDF5 → DataFrame loaders, mask builders, ROC/metrics utilities, plotting helpers, and PONI text/file helpers.
  - data_processing.fourier: FFT/feature extraction utilities (including real-space and Fourier-space beam masking and frequency-domain profiles).
  - data_processing.estimators: patient-level model training/stacking using RandomForest; exposes fit/predict/predict_proba and validation helpers.
  - data_processing.containers: dataclasses and enums describing limits and rules for data cleaning and q-value operations.
  - data_processing.pipeline: high-level MLPipeline that composes wrangling, preprocessing, and estimator steps with train/validate/predict flows.
- Quality control GUI (src/quality_control)
  - Packaged via its own setup.py with console_script entrypoint eosdx-quality-tool → eosdx_quality_tool.main:main.
  - To install and run the GUI locally:
    ```pwsh
    # Install the GUI package (entrypoint lives in the subpackage setup)
    pip install -e src/quality_control
    # Then launch
    eosdx-quality-tool
    ```
- Hardware/Ulster GUI (src/hardware/Ulster)
  - PyQt5-based GUI and hardware orchestration. Key areas:
    - gui/: UI composition and extensions for image view, points, state, and measurements.
    - technical/: acquisition workers and widgets (tests stub these for headless runs).
    - hardware/: detectors, XY stage control, capture dialog, and movement limits.
    - resources/: config (JSON), PONI examples, images; utils/ for logging setup.
  - Tests under src/hardware/Ulster/tests validate stage limits and metadata while stubbing PyQt5.
  - After editable install of the repo, the GUI can be run from its module script; see gui/Ulster.py for the entrypoint used by developers.

Conventions and tooling
- pytest configuration: pyproject.toml sets --import-mode=importlib (prevents test-time import shadowing with src layout).
- Linting/formatting: black (line length 88), isort (profile black), flake8 (max complexity 20; extended ignores for E203,W503,E501,F401,F841,C901,E402,F811 on src/xrdanalysis and src/quality_control), interrogate for doc coverage (non-blocking).
- Makefile exists but contains some cookiecutter-scaffold targets; prefer explicit commands above on Windows. The lint target delegates to flake8.

Quick start
1) Create and activate env, install packages and hooks:
   ```pwsh
   conda env create -f environment.yml
   conda activate eosdx
   pip install -e .
   pre-commit install
   ```
2) Run tests and lint:
   ```pwsh
   pytest
   pre-commit run --all-files
   ```
3) Build docs:
   ```pwsh
   sphinx-build docs _build
   ```
4) Optional GUIs:
   - Quality tool: pip install -e src/quality_control && eosdx-quality-tool
   - Ulster GUI: see src/hardware/Ulster/gui/Ulster.py for how it’s launched in development.
