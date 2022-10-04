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

    $ python eosdxanalysis/preprocessing/preprocess.py --input_path "INPUT_PATH" \
        --data_dir "DATA_DIR" --params_file "PARAMETERS_FILE_PATH"

where ``INPUT_PATH`` contains the directory ``DATA_DIR``, and the parameters file path can be anywhere. See existing parameters files for details.

Preprocessing requires specifying parameters. Preprocessing parameters can be specified as a text file in JSON-encoded format using the ``params_file`` flag as demonstrated above (preferred method). Alternatively, preprocessing parameters can be provided as a JSON-encoded string using the ``params`` flag, as demonstrated above.

The preprocessing parameters are as follows:

* ``h``: image height
* ``w``: image width
* ``beam_rmax``: defines circular region of interest of the beam
* ``rmin``: final image masking inner masking
* ``rmax``: final image masking outer masking
* ``eyes_rmin``: 9 A region of interest, inner radius of annulus
* ``eyes_rmax``: 9 A region of interest, outer radius of annulus
* ``eyes_blob_rmax``: defines a circle region of interest centered at the eye peak
* ``eyes_percentile``: defines the threshold for generating a binary blob for noise-robust 9 A peak finding
* ``local_thresh_block_size``: no longer used (future: specify filter type and size)
* ``crop_style``: choice of ``"both"``, ``"beam"``, or ``"outside"``. ``"beam"`` sets the inner circle of radius ``rmin`` to zero. ``"outside"`` sets values outside ``rmax`` to zero. ``"both"`` does both.
* ``plans``: a list of strings denoting the preprocessing plan(s) to perform. Choice of ``"original"``, ``"centerize"``, ``"centerize_rotate"``, and ``"centerize_rotate_quad_fold"``. (Note: JSON syntax does not allow for a spare comma at the end of a list, whereas Python does.)

A sample preprocessing parameters text file would containg the following content:

.. code-block:: javascript

    {
        "h": 256,
        "w": 256,
        "beam_rmax": 25,
        "rmin": 25,
        "rmax": 90,
        "eyes_rmin": 30,
        "eyes_rmax": 45,
        "eyes_blob_rmax": 20,
        "eyes_percentile": 99,
        "local_thresh_block_size": 21,
        "crop_style": "both",
        "plans": [
            "centerize",
            "centerize_rotate",
            "centerize_rotate_quad_fold"
        ]
    }
