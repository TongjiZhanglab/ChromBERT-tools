========
Overview
========

Command Structure
=================

ChromBERT-tools provides a comprehensive suite of command-line tools for regulatory genomics analysis. All commands follow the pattern:

.. code-block:: bash

   chrombert-tools <command> [OPTIONS]

Getting Help
============

To see all available commands:

.. code-block:: bash

   chrombert-tools --help

To get help for a specific command:

.. code-block:: bash

   chrombert-tools <command> --help

Command Categories
==================

ChromBERT-tools commands are organized into five main categories:

1. **Embedding Commands**: Extract embeddings for genes, regions, regulators, and cistromes
2. **Cell-specific Embedding Commands**: Extract cell-type specific embeddings
3. **Imputation Commands**: Impute missing cistrome data
4. **Inference Commands**: Infer transcriptional regulatory networks (TRNs)
5. **Driver Factor Analysis**: Identify driver factors in cell state transitions or differential regions

Common Parameters
=================

Most commands share these common parameters:

* ``--genome``: Genome assembly (``hg38`` or ``mm10``)
* ``--resolution``: Resolution (``200bp``, ``1kb``, ``2kb``, or ``4kb``; mm10 only supports ``1kb``)
* ``--odir``: Output directory (default: ``./output``)
* ``--batch-size``: Batch size for processing (default: 4)
* ``--num-workers``: Number of dataloader workers (default: 8)
* ``--chrombert-cache-dir``: ChromBERT cache directory (default: ``~/.cache/chrombert/data``)

File Format Requirements
========================

Input Data Formats
------------------

**Expression Files (CSV)**

Gene expression files should be in CSV format:

* Rows: genes
* Columns: samples
* Values: TPM/FPKM/counts

**Region Files (BED/CSV)**

Genomic regions can be provided in two formats:

* BED format: Tab-separated (chr, start, end, ...)
* CSV format: Comma-separated with columns chr, start, end

**Peak Files (BED)**

Peak calling results in standard BED format (chr, start, end)

**Signal Files (BigWig)**

Standard BigWig format for continuous signal data

Output Data Formats
-------------------

**Embeddings (PKL)**

Python pickle files containing dictionaries:

.. code-block:: python

   import pickle
   with open('embeddings.pkl', 'rb') as f:
       emb_dict = pickle.load(f)
   # emb_dict is a dictionary: {name: embedding_array}

**Embeddings (HDF5)**

HDF5 files for large-scale embedding data:

.. code-block:: python

   import h5py
   with h5py.File('embeddings.hdf5', 'r') as f:
       emb = f['emb/cistrome_name'][:]
       regions = f['region'][:]

**Network Files (TSV/PKL)**

Regulatory networks in tab-separated or pickle format

