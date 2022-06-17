# xrd-analysis
Software for Eos Dx, Inc. sample preprocessing and analysis.

# Codebase
The main modules are `preprocessing` and `models`.

`preprocessing` contains image preprocessing classes and utilities to process batches of samples.

`models` contains learning models, such as logistic regression, and function fitting models, such as 2D Polar Discrete Fourier Transform.

# Installation
Use miniforge (https://github.com/conda-forge/miniforge) and create environment with environment.yml file from xrd-analysis repo:

`conda env create --name env_name --file environment.yaml`

# Contributing
1. Create or refer to an existing issue (bug or feature).
2. Create a branch to work on the issue with simple, descriptive name.
3. Create tests for the issue.
4. Commit each incremental change with a simple, descriptive commit message.
5. Create a pull request when the issue is resolved.
