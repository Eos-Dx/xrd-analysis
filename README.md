# DiFRA (Data Collection) – xrd-analysis

DiFRA is the hardware GUI and data collection layer for EOS-Dx experiments. The core analysis library remains under `src/xrdanalysis`.

Highlights:
- New package path for hardware GUI: `src/hardware/difra` (Ulster naming removed)
- One-click launcher: `bin\run_difra.bat`
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
- `src\\hardware\\bin\\run_difra.bat`
  - The script reads the conda env name from `src/hardware/difra/resources/config/global.json` (the `"conda"` field, e.g. `"ulster37"`) and runs:
    - `conda run -n <env> python src/hardware/difra/gui/main_app.py`


Configuration
- Global config: `src/hardware/difra/resources/config/global.json`
  - `conda`: Name of the conda environment the launcher should use
  - `default_setup`: Name of the default experimental setup to load
  - DEV flags and global defaults (paths)
- Per-setup configs: `src/hardware/difra/resources/config/setups/*.json`
  - Each file defines `detectors`, `translation_stages`, and active selections for a setup
  - Example setups included: `ulster.json`, `queen-mary.json`


Repository layout (key parts)
- `src/xrdanalysis`: Core analysis library (integration, transformers, utilities)
- `src/hardware/difra`: Hardware GUI, controllers, and resources
  - `gui/`: PyQt5 GUI (views, extensions, technical and zone measurements)
  - `hardware/`: Detectors, stage controllers, and movement logic
  - `resources/`: `config/global.json`, `config/setups/*.json`, PONI examples, images, faulty pixels
  - `tests/`: Headless tests that stub GUI where needed
- `bin/run_difra.bat`: Windows launcher (reads env from config and runs the GUI)


Development commands
- Lint/format via pre-commit (recommended):
  ```pwsh
  pre-commit run --all-files
  ```
- Run tests:
  ```pwsh
  pytest -q
  # hardware GUI-only tests
  pytest src/hardware/difra/tests -q
  ```
- Build docs (if needed):
  ```pwsh
  sphinx-build docs _build
  # or
  make -C docs html
  ```


Notes
- The legacy `src/hardware/Ulster` path has been removed in favor of `src/hardware/difra`.
- Multiple experimental setup profiles are supported. Use `--setup <name>` when launching (e.g., `--setup Ulster` or `--setup Queen-Mary`), or select from the dialog at startup.
