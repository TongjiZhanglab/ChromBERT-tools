============
Installation
============

Overview
========

ChromBERT-tools is a lightweight GitHub toolkit that exposes core ChromBERT functionality through easy-to-use command-line tools (CLI). To use ChromBERT-tools, you need to install:

1. ChromBERT dependencies
2. ChromBERT datasets
3. ChromBERT-tools package

Prerequisites
=============

* Python 3.9 or higher
* CUDA-compatible GPU (recommended)
* Conda or pip package manager

Installing ChromBERT Dependencies
==================================

If you have already installed ChromBERT dependencies, you can skip this step and proceed to :ref:`installing-chrombert-dataset`.

Using Singularity Image (Recommended)
--------------------------------------

For direct use of these CLI tools, it is recommended to utilize the ChromBERT Singularity image. **These images include almost all packages needed by ChromBERT and ChromBERT-tools**, including:

* flash-attention-2
* transformers
* pytorch
* and other dependencies

Installing Singularity/Apptainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, install ``singularity`` (or ``Apptainer``):

.. code-block:: bash

   conda install -c conda-forge apptainer

Testing the Singularity Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Then you can test whether it was successfully installed:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif python -c "import chrombert; print('hello chrombert')"
   singularity exec --nv /path/to/chrombert.sif chrombert-tools

Installing from Source
----------------------

If you want to install from source and use development mode, you can follow the instructions in the `ChromBERT repository <https://github.com/TongjiZhanglab/ChromBERT>`_.

Key dependencies include:

* PyTorch >= 2.0
* transformers >= 4.30.0
* flash-attention-2
* numpy, pandas, scipy
* h5py, pyBigWig
* scikit-learn

.. _installing-chrombert-dataset:

Installing ChromBERT Dataset
=============================

ChromBERT requires pre-trained models and annotation data files. These files should be downloaded from Hugging Face to ``~/.cache/chrombert/data``.

Supported Genomes and Resolutions
----------------------------------

You can download datasets for:

* **hg38** (Human): 200bp, 1kb, 2kb, 4kb resolutions
* **mm10** (Mouse): 1kb resolution

Basic Installation
------------------

To download the required datasets:

.. code-block:: bash

   chrombert_prepare_env --genome hg38 --resolution 1kb

This command will download:

* Pre-trained ChromBERT model weights
* Region annotation files
* TF/regulator metadata
* HDF5 feature files

Using a Hugging Face Mirror
----------------------------

If you're experiencing connectivity issues with Hugging Face, you can use the ``--hf-endpoint`` option to connect to an available mirror:

.. code-block:: bash

   chrombert_prepare_env --genome hg38 --resolution 1kb --hf-endpoint <Hugging Face endpoint>

Multiple Resolutions
--------------------

To install multiple resolutions, run the command multiple times:

.. code-block:: bash

   chrombert_prepare_env --genome hg38 --resolution 1kb
   chrombert_prepare_env --genome hg38 --resolution 2kb
   chrombert_prepare_env --genome mm10 --resolution 1kb

.. _installing-chrombert-tools:

Installing ChromBERT-tools
===========================

Installation from GitHub
-------------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
   cd ChromBERT-tools
   pip install -e .

The ``-e`` flag installs the package in editable/development mode, which is useful if you want to modify the code or contribute to the project.

Verifying Installation
-----------------------

To verify that ChromBERT-tools was installed correctly, run:

.. code-block:: bash

   chrombert-tools

You should see a list of available commands and their descriptions.

To check the version:

.. code-block:: bash

   python -c "import chrombert_tools; print(chrombert_tools.__version__)"

Using with Singularity
======================

If you're using the Singularity image, you can run ChromBERT-tools commands inside the container:

.. code-block:: bash

   # Run a command inside the container
   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed-region -h
   
   # Bind mount your data directory
   singularity exec --nv --bind /your/data:/data /path/to/chrombert.sif \
       chrombert-tools embed-region -h

Important Notes
^^^^^^^^^^^^^^^

* Use ``--nv`` flag to enable NVIDIA GPU support
* Use ``--bind`` to mount host directories into the container
* Ensure your data paths are accessible within the container

Troubleshooting
===============

CUDA Out of Memory
------------------

If you encounter CUDA out of memory errors:

1. Reduce the batch size using ``--batch-size`` option

File Not Found: ChromBERT Data
-------------------------------

If ChromBERT cannot find data files, ensure:

1. You have run ``chrombert_prepare_env`` for your genome and resolution
2. The cache directory ``~/.cache/chrombert/data`` exists and contains the required files
3. You can specify a custom cache directory using ``--chrombert-cache-dir``

Next Steps
==========

Once installation is complete, check out the :doc:`usage` section to learn how to use ChromBERT-tools for your analysis.

