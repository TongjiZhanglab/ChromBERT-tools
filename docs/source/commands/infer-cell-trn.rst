==============
infer-cell-trn
==============

Infer cell-type specific transcriptional regulatory network (TRN).

Overview
========

The ``infer-cell-trn`` command fine-tunes ChromBERT on cell-specific accessibility data, then infers a cell-type specific regulatory network from expression data.

Basic Usage
===========

Train and infer:

.. code-block:: bash

   chrombert-tools infer-cell-trn \
     --tpm expression.csv \
     --cell-type-bw cell_ATAC.bigwig \
     --cell-type-peak cell_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use existing checkpoint:

.. code-block:: bash

   chrombert-tools infer-cell-trn \
     --tpm expression.csv \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--tpm``
   Gene expression file in CSV format

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Conditional Parameters
----------------------

**If training** (no ``--ft-ckpt``):

``--cell-type-bw``
   Chromatin accessibility BigWig file

``--cell-type-peak``
   Peak calling results in BED format

**If using existing model**:

``--ft-ckpt``
   Path to fine-tuned checkpoint file

Optional Parameters
-------------------

``--top-k``
   Number of top regulators per gene (default: 10)

``--mode``
   Training mode: ``fast`` (default) or ``full``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

Output Files
============

Network Outputs
---------------

``trn_network.tsv``
   Cell-specific regulatory network

``trn_network.pkl``
   Network in pickle format

``network_statistics.json``
   Network statistics

Training Outputs (if trained)
------------------------------

``dataset/``
   Training dataset

``train/``
   Model checkpoint and evaluation metrics

Examples
========

K562 Network
------------

.. code-block:: bash

   chrombert-tools infer-cell-trn \
     --tpm K562_expression.csv \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --top-k 20 \
     --mode fast \
     --genome hg38 \
     --resolution 1kb \
     --odir K562_network

Reuse Checkpoint
----------------

.. code-block:: bash

   # Train once
   chrombert-tools infer-cell-trn \
     --tpm K562_timepoint1.csv \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --genome hg38 \
     --odir K562_tp1_network
   
   # Reuse for another timepoint
   CKPT="K562_tp1_network/train/*/checkpoints/epoch=*.ckpt"
   chrombert-tools infer-cell-trn \
     --tpm K562_timepoint2.csv \
     --ft-ckpt $CKPT \
     --genome hg38 \
     --odir K562_tp2_network

Compare Networks
================

.. code-block:: python

   import pandas as pd
   
   # Load two networks
   net1 = pd.read_csv('condition1/trn_network.tsv', sep='\t')
   net2 = pd.read_csv('condition2/trn_network.tsv', sep='\t')
   
   # Find condition-specific edges
   net1_edges = set(zip(net1['gene'], net1['regulator']))
   net2_edges = set(zip(net2['gene'], net2['regulator']))
   
   unique_to_1 = net1_edges - net2_edges
   unique_to_2 = net2_edges - net1_edges
   shared = net1_edges & net2_edges
   
   print(f"Shared edges: {len(shared)}")
   print(f"Unique to condition 1: {len(unique_to_1)}")
   print(f"Unique to condition 2: {len(unique_to_2)}")

Tips
====

1. **Cell-specific vs. general**: 
   
   * Use cell-specific when accessibility data available
   * More accurate for cell-type specific networks
   * General TRN sufficient for exploration

2. **Checkpoint reuse**: 
   
   * Train once per cell type
   * Reuse for multiple timepoints/conditions
   * Significant time savings

3. **Validation**: 
   
   * Compare with known TF-target relationships
   * Check enrichment of TF motifs
   * Validate key edges experimentally

See Also
========

* :doc:`infer-trn` - General TRN inference
* :doc:`embed-cell-gene` - Cell-specific gene embeddings
* :doc:`overview` - Common parameters

