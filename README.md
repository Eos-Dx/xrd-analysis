# xrd-analysis
Software for Eos Dx, Inc. sample preprocessing and analysis.

# Codebase
The python package is named `eosdxanalysis`.

The `eosdxanalysis` main modules are `preprocessing` and `models`.

`preprocessing` contains image preprocessing classes and utilities to process batches of samples.

`models` contains learning models, such as logistic regression, K-means, function fitting models (polynomial), and function transforms (Fourier).

# Contributing

## Development Environment Setup
Use miniforge (https://github.com/conda-forge/miniforge) and create environment with environment.yml file from xrd-analysis repo:

`conda env create --name env_name --file environment.yaml`

## Installation
Install the `eosdxanalysis` python package as follows:
1. Clone this repository: `git clone https://github.com/Eos-Dx/xrd-analysis`
2. Change into the `eosdxanalysis` directory.
3. Install using pip: `pip install -e .`

## GitHub Workflow
Use the GitHub commandline interface to speed up development. Make sure to follow best practices.
1. Create an issue, and optionally add a label: `gh issue create`
2. Create a branch named according to the issue type, description, and issue number: `git checkout -b feature/brilliant-new-feature/42`
3. Solve the issue and include tests and documentation if appropriate.
4. some commits: `git add name_of_file_changed; git commit -m "adds new brilliant feature"`
5. Create a pull request and note its number: `gh pr create`
6. Merge the pull request and reference the issue: `gh pr merge 43 --body "Solves #42"`
