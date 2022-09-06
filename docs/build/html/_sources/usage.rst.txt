Usage
=====

Installation
------------

To use **eosdxanalysis**, first set up a **conda** environment with `miniforge <https://github.com/conda-forge/miniforge>`_.


1. Clone the repository from GitHub

.. code-block:: console

    $ git clone https://github.com/Eos-Dx/xrd-analysis

2. Change into the local repository directory

.. code-block:: console

    $ cd xrd-analysis

3. Set up a conda environment and activate

.. code-block:: console

    $ conda env create --name eos --file environment.yaml
    $ conda activate eos

4. The prompt should look like 

.. code-block:: console

    (eos) $ 

5. Finally, install eosdxanalysis Python library

.. code-block:: console

    (eos) $ pip install .

Preprocessing
-------------

In the shell, run the following commands from the **xrd-analysis** directory to preprocess raw data,

.. code-block:: console

    $ python eosdxanalysis/preprocessing/preprocess.py --parent_dir "PARENT_PATH" \
        --samples_dir "SAMPLES_DIR" --params_file "PARAMETERS_FILE_PATH"

where ``PARENT_PATH`` contains the directory ``SAMPLES_DIR``, and the parameters file path can be anywhere. See existing parameters files for details.
