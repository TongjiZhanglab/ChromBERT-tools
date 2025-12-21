================
embed_regulator
================

Extract embeddings for transcription factors and other regulators.

Overview
========

The ``embed_regulator`` command extracts general embeddings for specified regulators across genomic regions using the pre-trained ChromBERT model. 

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed_regulator \
     --regulator "CTCF;MYC;TP53" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--regulator``
   Regulator names separated by semicolons, it will be converted to lowercase for better matching

``--region``
   BED or CSV file specifying regions of interest or csv file with columns: chr, start, end

Optional Parameters
-------------------

``--help``
   Show help message

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Batch size for training (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

Output Files
============

``regulator_emb_on_region.hdf5``
   HDF5 file containing regulator embeddings for each region   
   .. code-block:: python
      import h5py
      with h5py.File('regulator_emb_on_region.hdf5', 'r') as f:
          ctcf_emb = f['/emb/ctcf'][:]
          myc_emb = f['/emb/myc'][:]
          tp53_emb = f['/emb/tp53'][:]

``mean_regulator_emb.pkl``
   Python dictionary containing mean embeddings for each regulator
   .. code-block:: python
      import pickle
      with open('mean_regulator_emb.pkl', 'rb') as f:
          mean_embeddings = pickle.load(f)
      # mean_embeddings = {'ezh2': array([...]), 'myc': array([...]), 'tp53': array([...]), ...}

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found in ChromBERT
   

Troubleshooting
===============

**Regulator not found**

   * Check if the regulator is correct
   * You can find all regulator in your chrombert-cache-dir/anno/*_regulator_list.txt