Welcome to ChromBERT-tools Documentation!
==========================================

**ChromBERT** is a pre-trained deep learning model designed to capture genome-wide co-association patterns of ~1,000 transcription regulators and to learn context-specific transcriptional regulatory networks (TRNs).

**ChromBERT-tools** is a lightweight toolkit that exposes core ChromBERT functionality through easy-to-use command-line tools (CLI).

Features
--------

* **Easy-to-use CLI**: Simple command-line interface for common ChromBERT tasks
* **Flexible**: Works with hg38 (human) and mm10 (mouse) genomes, and different resolutions
* **Comprehensive**: Tools for embedding, imputation, inference, and driver factor analysis
* **Cell-specific**: Support for cell-type specific analysis

ChromBERT-tools CLI
---------------------
* :doc:`commands/embed_cistrome`: Extract cistrome embeddings on specified regions
* :doc:`commands/embed_gene`: Extract gene embeddings on specified regions
* :doc:`commands/embed_region`: Extract region embeddings on specified regions
* :doc:`commands/embed_regulator`: Extract regulator embeddings on specified regions
* :doc:`commands/infer_trn`: Infer transcriptional regulatory network (TRN) on specified regions
* :doc:`commands/infer_cell_trn`: Infer cell-specific transcriptional regulatory network (TRN) on specified regions and cell-specific key regulators
* :doc:`commands/impute_cistrome`: Impute cistrome data on specified regions
* :doc:`commands/find_driver_in_dual_region`: Find driver factors in dual-functional regions
* :doc:`commands/find_driver_in_transition`: Find driver factors in cell-state transition

Quick Start
-----------

Check out the :doc:`installation` section for setup instructions, and the :doc:`usage` section to learn how to use ChromBERT-tools.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage


Links
-----

* ChromBERT GitHub: https://github.com/TongjiZhanglab/ChromBERT
* ChromBERT-tools GitHub: https://github.com/TongjiZhanglab/ChromBERT-tools
* Documentation: https://chrombert-tools.readthedocs.io/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
