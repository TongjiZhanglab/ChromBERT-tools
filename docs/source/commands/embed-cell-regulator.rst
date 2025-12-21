=====================
embed-cell-regulator
=====================

Extract cell-specific regulator embeddings.

Overview
========

The ``embed-cell-regulator`` command fine-tunes ChromBERT on cell-type specific data, then extracts regulator embeddings using the cell-specific model. This produces regulat or embeddings that reflect cell-type specific regulatory patterns.

Basic Usage
===========

Train new model:

.. code-block:: bash

   chrombert-tools embed-cell-regulator \
     --regulator "CTCF;MYC;TP53" \
     --region regions.bed \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use existing checkpoint:

.. code-block:: bash

   chrombert-tools embed-cell-regulator \
     --regulator "CTCF;MYC;TP53" \
     --region regions.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--regulator``
   Regulator names separated by semicolons

``--region``
   BED or CSV file with regions

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

``--mode``
   Training mode: ``fast`` (default) or ``full``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

Output Files
============

``cell_specific_regulator_emb_on_region.hdf5``
   HDF5 file with cell-specific regulator embeddings per region

``mean_cell_specific_regulator_emb.pkl``
   Mean cell-specific regulator embeddings

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found

Examples
========

Cell-specific TF Analysis
--------------------------

.. code-block:: bash

   chrombert-tools embed-cell-regulator \
     --regulator "CTCF;MYC;JUN;FOS;STAT3" \
     --region promoters.bed \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --mode fast \
     --genome hg38 \
     --resolution 1kb \
     --odir K562_tf_embeddings

See Also
========

* :doc:`embed-regulator` - General regulator embeddings
* :doc:`embed-cell-gene` - Cell-specific gene embeddings
* :doc:`overview` - Common parameters

