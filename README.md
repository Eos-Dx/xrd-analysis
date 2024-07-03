# xrd-analysis
Repository for collaboration of EOSDX team

To start working with this repo:

Install the packages using conda:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate eosdx
```

Install environment for pre-commit:
```bash
pre-commit install
```

Install the editable package locally:
```bash
pip install -e .
```

When required you can update the environment using conda:
```bash
conda env update -f environment.yml --prune
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── requirements.txt   <- The environment file for reproducing the analysis environment, e.g.
    │                         generated with `conda env create -f environment.yml`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── xrdanalysis
    |   |     └── __init__.py   <- Initializes Python module
    |   |        │
    |   |        └── data_processing    <- Code used to process data.
    |   |
    │   └── tests          <- Tests for the codes. Mirrors xrdanalysis.
    |
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
