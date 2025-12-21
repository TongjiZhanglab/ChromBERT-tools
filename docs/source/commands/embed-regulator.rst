================
embed-regulator
================

Extract embeddings for transcription factors and other regulators.

Overview
========

The ``embed-regulator`` command extracts embeddings for specified transcription factors, histone marks, and other regulatory elements across genomic regions. It outputs both per-region embeddings and mean embeddings for each regulator.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed-regulator \
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
   Regulator names separated by semicolons
   
   Examples:
   
   * Transcription factors: ``"CTCF;MYC;TP53;GATA3"``
   * Histone marks: ``"H3K4me3;H3K27ac;H3K9me3"``
   * Mixed: ``"CTCF;H3K4me3;POLR2A;H3K27ac"``

``--region``
   BED or CSV file specifying regions of interest or csv file with columns: chr, start, end

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Optional Parameters
-------------------

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``)

``--batch-size``
   Batch size for processing (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

Output Files
============

``regulator_emb_on_region.hdf5``
   HDF5 file containing regulator embeddings for each region
   
   Structure:
   
   * ``emb/<regulator_name>``: Array of shape [n_regions, 768]
   * ``region``: Array of regions [chr, start, end, index]
   
   .. code-block:: python
   
      import h5py
      
      with h5py.File('regulator_emb_on_region.hdf5', 'r') as f:
          ctcf_emb = f['emb/CTCF'][:]  # Shape: (n_regions, 768)
          myc_emb = f['emb/MYC'][:]
          regions = f['region'][:]

``mean_regulator_emb.pkl``
   Python dictionary containing mean embeddings for each regulator
   
   .. code-block:: python
   
      import pickle
      
      with open('mean_regulator_emb.pkl', 'rb') as f:
          mean_embs = pickle.load(f)
      # mean_embs = {'CTCF': array([...]), 'MYC': array([...]), ...}

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found in ChromBERT

Tips
====

1. **Regulator names**: 
   
   * Check available regulators in ChromBERT database 
   * all regulator name will be converted to lowercase for better matching


2. **Performance**: 
   
   * More regions = more memory and time
   * More regulators = more memory but same time

Troubleshooting
===============

**Regulator not found**

* Check regulator name spelling
* Verify regulator exists in ChromBERT database (~/.cache/chrombert/data/config/*regulators_list.txt)
* Try alternative names (e.g., POLR2A vs Pol2)

**HDF5 file too large**

* Process fewer regulators at once
* Use smaller region sets
* Consider using mean embeddings only

**Slow processing**

* Increase ``--batch-size`` if GPU memory allows
* Increase ``--num-workers``
* Use faster storage (SSD)

See Also
========

* :doc:`embed-cistrome` - Extract cistrome embeddings
* :doc:`embed-cell-regulator` - Cell-specific regulator embeddings
* :doc:`overview` - Common parameters
