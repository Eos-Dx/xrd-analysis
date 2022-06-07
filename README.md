# xrd-analysis
Software for Eos Dx, Inc. sample preprocessing and analysis.

# Codebase
The main modules are `preprocessing` and `models`.

`preprocessing` contains image preprocessing classes and utilities to process batches of samples.

`models` contains learning models, such as logistic regression, and function fitting models, such as 2D Polar Discrete Fourier Transform.

# Installation
Use miniforge (https://github.com/conda-forge/miniforge) and create environment with environment.yml file from xrd-analysis repo:

`conda env create --name env_name --file environment.yaml`
