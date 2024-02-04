# xrd-analysis
Software for Eos Dx, Inc. sample preprocessing and analysis. Last updated for v2.77.5.

## Documentation

Open ``xrd-analysis/eosdxanalysis/docs/build/html/index.html`` in your web browser.

# Codebase
The python package is named ``eosdxanalysis``.

The ``eosdxanalysis`` main modules are ``calibration``, ``models``, ``preprocessing``, ``simulations``, and ``visualization``.

* ``calibration`` contains code to calibrate measurements based on machine parameters and calibration measurements.

* ``models`` contains learning models, such as logistic regression, K-means, as well as function fitting models and function transforms (Fourier).

* ``preprocessing`` contains image preprocessing code to process batches of raw x-ray diffraction measurement data.

* ``simulations`` contains code based on the physics of x-ray diffraction.

* ``visualization`` contains code to visualize measurements and results of learning models.

# Contributing

## Development Environment Setup
Use miniforge (https://github.com/conda-forge/miniforge) and create environment with environment.yml file from xrd-analysis repository:

```bash
conda env create --name eos --file environment.yaml
conda activate eos
```

Note: Remove the line containing ``sphinx-automodapi=0.41.1`` if install fails.

## Installation

First follow steps here to set up GitHub account and use SSH keys for access: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection

Install the ``eosdxanalysis`` python package as follows:
1. Clone this repository: ``git clone https://github.com/Eos-Dx/xrd-analysis``
2. Change into the ``xrd-analysis`` directory.
3. Install using pip: ``pip install -e .``

## GitHub Workflow
Use the GitHub commandline interface to speed up development. Make sure to follow best practices.
1. Create an issue, and optionally add a label: ``gh issue create``
2. Create a branch named according to the issue type, description, and issue number:
``git checkout -b feature/brilliant-new-feature/42``
3. Solve the issue and include tests and documentation if appropriate.
4. Make some commits: ``git add name_of_file_changed; git commit -m "adds new brilliant feature"``
5. Create a pull request and note its number: ``gh pr create``
6. Merge the pull request and reference the issue: ``gh pr merge 43 --body "Solves #42"``

## Preprocessing

### Preprocessing Raw Data
In the shell, run the following commands from the ``xrd-analysis`` directory to preprocess raw data,
```bash
python /path/to/eosdxanalysis/preprocessing/preprocess.py --input_path "INPUT_PATH" --data_dir "DATA_DIR" --params_file "PARAMETERS_FILE_PATH"
```
where ``/path/to/eosdxanalysis/preprocessing/preprocess.py`` is the path to ``preprocess.py``, ``INPUT_PATH`` contains the data directory ``DATA_DIR``, and ``PARAMETERS_FILE_PATH`` is the path to the parameters file. See full documentation for details.
