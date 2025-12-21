====================
embed-cell-cistrome
====================

Extract cell-specific cistrome embeddings on specified regions.

Overview
========

The ``embed-cell-cistrome`` command fine-tunes ChromBERT on cell-type specific data, then extracts cistrome embeddings using the cell-specific model.

Basic Usage
===========

Train new model:

.. code-block:: bash

   chrombert-tools embed-cell-cistrome \
     --cistrome "CTCF:K562;H3K27ac:K562" \
     --region regions.bed \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output \

Use existing checkpoint:

.. code-block:: bash

   chrombert-tools embed-cell-cistrome \
     --cistrome "CTCF:K562;H3K27ac:K562" \
     --region regions.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--cistrome``
   Cistrome identifiers (GSM/ENCODE IDs or factor:cell pairs)

``--region``
   BED or CSV file with regions

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

``--mode``
   Training mode: ``fast`` (default) or ``full``, if ``fast`` mode is used, only the sampled 20000 regions will be used for training

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

Output Files
============

``cell_specific_cistrome_emb_on_region.hdf5``
   HDF5 file with cell-specific cistrome embeddings per region

``cell_specific_mean_cistrome_emb.pkl``
   Mean cell-specific cistrome embeddings

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found

Examples
========

K562 Cistromes
--------------

.. code-block:: bash

   chrombert-tools embed-cell-cistrome \
     --cistrome "CTCF:K562;H3K27ac:K562;ATAC-seq:K562" \
     --region your-regions.bed \
     --cell-type-bw your-cell-type-ATAC.bigwig \
     --cell-type-peak your-cell-type-peaks.bed \
     --mode fast \
     --genome hg38 \
     --resolution 1kb \
     --odir output

See Also
========

* :doc:`embed-cistrome` - General cistrome embeddings
* :doc:`embed-cell-gene` - Cell-specific gene embeddings
* :doc:`overview` - Common parameters

