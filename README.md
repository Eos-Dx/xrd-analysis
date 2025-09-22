# EOSDxDc (Data Collection) – xrd-analysis

EOSDxDc is the hardware GUI and data collection layer for EOS-Dx experiments. The core analysis library remains under `src/xrdanalysis`.

Highlights:
- New package path for hardware GUI: `src/hardware/eosdxdc` (Ulster naming removed)
- One-click launcher: `bin\run_eosdxdc.bat`
- Config-driven conda env selection (`resources/config/main.json`)


Quick start (Windows, Conda)
- Create and activate an environment (or update your existing one):
  ```pwsh
  conda env create -f environment.yml  # or: conda env update -f environment.yml --prune
  conda activate eosdx
  ```
- Install repo for development and enable hooks:
  ```pwsh
  pip install -e .
  pre-commit install
  ```
- Launch the hardware GUI:
  - Run:
    - `bin\run_eosdxdc.bat`
  - The script reads the conda env name from `src/hardware/eosdxdc/resources/config/main.json` (the `"conda"` field, e.g. `"ulster37"`) and runs:
    - `conda run -n <env> python src/hardware/eosdxdc/gui/main_app.py`


Configuration
- Main config: `src/hardware/eosdxdc/resources/config/main.json`
  - `conda`: Name of the conda environment the launcher should use
  - `detectors`, `translation_stages`: Active hardware and settings
  - DEV flags and demo assets for running without physical hardware


Repository layout (key parts)
- `src/xrdanalysis`: Core analysis library (integration, transformers, utilities)
- `src/hardware/eosdxdc`: Hardware GUI, controllers, and resources
  - `gui/`: PyQt5 GUI (views, extensions, technical and zone measurements)
  - `hardware/`: Detectors, stage controllers, and movement logic
  - `resources/`: `config/main.json`, PONI examples, images, faulty pixels
  - `tests/`: Headless tests that stub GUI where needed
- `bin/run_eosdxdc.bat`: Windows launcher (reads env from config and runs the GUI)


Development commands
- Lint/format via pre-commit (recommended):
  ```pwsh
  pre-commit run --all-files
  ```
- Run tests:
  ```pwsh
  pytest -q
  # hardware GUI-only tests
  pytest src/hardware/eosdxdc/tests -q
  ```
- Build docs (if needed):
  ```pwsh
  sphinx-build docs _build
  # or
  make -C docs html
  ```


Notes
- The legacy `src/hardware/Ulster` path has been removed in favor of `src/hardware/eosdxdc`.
- Future direction: multiple experimental setup profiles selectable at startup (per-setup configs under `resources/config/setups/`).
