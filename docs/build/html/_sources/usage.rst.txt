Usage
=====

Installation
------------

1. Install ``conda`` using `miniforge <https://github.com/conda-forge/miniforge>`_.

2. Follow all steps in `Connecting to GitHub with SSH <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`_.

3. Clone the repository from GitHub

.. code-block:: console

    $ git clone https://github.com/Eos-Dx/xrd-analysis

3. Change into the local repository directory

.. code-block:: console

    $ cd xrd-analysis

4. Set up a ``conda`` environment and activate (Note: very large download and high CPU usage)

.. code-block:: console

    $ conda env create --name eos --file environment.yaml
    $ conda activate eos

4. The prompt should look like 

.. code-block:: console

    (eos) $ 

5. Finally, install the ``eosdxanalysis`` Python library

.. code-block:: console

    (eos) $ pip install -e .

Preprocessing
-------------

In the shell, run the following commands from the ``xrd-analysis`` directory to preprocess raw data,

.. code-block:: console

    $ python eosdxanalysis/preprocessing/preprocess.py --parent_dir "PARENT_PATH" \
        --samples_dir "SAMPLES_DIR" --params_file "PARAMETERS_FILE_PATH"

where ``PARENT_PATH`` contains the directory ``SAMPLES_DIR``, and the parameters file path can be anywhere. See existing parameters files for details.
