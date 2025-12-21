================
impute-cistrome
================

Impute missing cistrome data using ChromBERT.

Overview
========

The ``impute-cistrome`` command uses ChromBERT's learned co-association patterns to impute ChIP-seq or ATAC-seq data for cell types or factors where experimental data is unavailable.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools impute-cistrome \
     --cistrome-query "CTCF:MyCell;H3K27ac:MyCell" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--cistrome-query``
   Cistromes to impute in factor:celltype format
   
   Examples:
   
   * ``"CTCF:Neuron"``
   * ``"H3K27ac:MyCell;H3K4me3:MyCell"``
   * ``"ATAC-seq:RareCell"``

``--region``
   Regions to impute (BED or CSV format)

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Optional Parameters
-------------------

``--top-k``
   Number of similar cistromes to use for imputation
   
   Default: 50

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Batch size (default: 4)

``--num-workers``
   Dataloader workers (default: 8)

Output Files
============

``imputed_signal.hdf5``
   HDF5 file containing imputed signal values
   
   Structure:
   
   * ``signal/<cistrome_query>``: Imputed signal for each query
   * ``region``: Regions

``imputation_report.json``
   Quality metrics and similar cistromes used
   
   Contains:
   
   * Top-k similar cistromes
   * Similarity scores
   * Confidence metrics

Examples
========

Impute for Rare Cell Type
--------------------------

.. code-block:: bash

   chrombert-tools impute-cistrome \
     --cistrome-query "CTCF:RareCell;H3K27ac:RareCell" \
     --region enhancers.bed \
     --top-k 50 \
     --genome hg38 \
     --resolution 1kb \
     --odir rare_cell_imputation

Impute Missing Factor
---------------------

.. code-block:: bash

   chrombert-tools impute-cistrome \
     --cistrome-query "NewTF:K562" \
     --region promoters.bed \
     --top-k 100 \
     --genome hg38 \
     --resolution 1kb \
     --odir new_tf_imputation

Using Imputed Data
==================

.. code-block:: python

   import h5py
   import json
   
   # Load imputed signals
   with h5py.File('output/imputed_signal.hdf5', 'r') as f:
       ctcf_signal = f['signal/CTCF:MyCell'][:]
       h3k27ac_signal = f['signal/H3K27ac:MyCell'][:]
       regions = f['region'][:]
   
   # Load quality report
   with open('output/imputation_report.json', 'r') as f:
       report = json.load(f)
   
   print("Similar cistromes used for CTCF:MyCell:")
   for cis in report['CTCF:MyCell']['top_similar']:
       print(f"  {cis['name']}: {cis['similarity']:.3f}")

Tips
====

1. **Quality assessment**: 
   
   * Check imputation_report.json for confidence
   * Higher similarity scores = better imputation
   * Validate with independent data when possible

2. **Top-k selection**: 
   
   * Default (50) works well for most cases
   * Increase for better quality but slower
   * Decrease for faster results

3. **Use cases**: 
   
   * Rare cell types with no experimental data
   * Missing TF binding data
   * Preliminary analysis before experiments

Limitations
===========

* Quality depends on similar cistromes in database
* Novel cell types may have lower quality
* Should be validated when possible
* Not a replacement for experimental data

See Also
========

* :doc:`embed-cistrome` - Extract cistrome embeddings
* :doc:`overview` - Common parameters

