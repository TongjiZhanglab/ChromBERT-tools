============
Installation
============

ChromBERT-tools is a lightweight toolkit that exposes core ChromBERT functionality through easy-to-use command-line tools (CLI). To use ChromBERT-tools, you need to install:

1. ChromBERT dependencies
2. ChromBERT datasets
3. ChromBERT-tools package


Installing ChromBERT Dependencies
==================================

If you have already installed ChromBERT dependencies, you can skip this step and proceed to :ref:`installing-chrombert-dataset`.

Using Singularity Image (Recommended)
--------------------------------------

For direct use of these CLI tools, it is recommended to utilize the ChromBERT Singularity image. We provide a pre-built Singularity image available: `chrombert.sif <https://drive.google.com/file/d/1ePmDK6DANSq-zkRgVBTxSBnKBZk-cEzM/view?usp=sharing>`_. **These images include almost all packages needed by ChromBERT and ChromBERT-tools**, including: flash-attention-2, transformers, pytorch, and other dependencies.

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


.. _installing-chrombert-dataset:

Installing ChromBERT Dataset
=============================
If you have already installed ChromBERT datasets, you can skip this step and proceed to :ref:`installing-chrombert-tools`.

ChromBERT requires pre-trained models and annotation data files. These files should be downloaded from Hugging Face to ``~/.cache/chrombert/data``. Supported Genomes and Resolutions:

* **hg38** (Human): 200bp, 1kb, 2kb, 4kb resolutions
* **mm10** (Mouse): 1kb resolution

To download the required datasets:

.. code-block:: bash

   chrombert_prepare_env --genome <genome> --resolution <resolution>

Using a Hugging Face Mirror
----------------------------

If you're experiencing connectivity issues with Hugging Face, you can use the ``--hf-endpoint`` option to connect to an available mirror:

.. code-block:: bash

   chrombert_prepare_env --genome <genome> --resolution <resolution> --hf-endpoint <Hugging Face endpoint>

If you're using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert_prepare_env --genome <genome> --resolution <resolution> --hf-endpoint <Hugging Face endpoint>


.. _installing-chrombert-tools:

Installing ChromBERT-tools
===========================

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
   cd ChromBERT-tools
   pip install .

Verifying Installation
-----------------------

To verify that ChromBERT-tools was installed correctly, run:

.. code-block:: bash

   chrombert-tools

You should see a list of available commands and their descriptions.

Next Steps
==========

Once installation is complete, check out the :doc:`usage` section to learn how to use ChromBERT-tools for your analysis.

