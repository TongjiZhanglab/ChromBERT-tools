==================
embed_cell_region
==================

Extract cell-specific region embeddings.

Overview
========

The ``embed_cell_region`` command fine-tunes ChromBERT on cell-type specific accessibility data (if you don't provide finetuned checkpoint, else use the finetuned checkpoint), then extracts gene embeddings using the cell-specific model. This produces embeddings that reflect cell-type specific regulatory patterns.

Basic Usage
===========

Train new model:

.. code-block:: bash

   chrombert-tools embed_cell_region \
     --region regions.bed \
     --cell-type-bw cell-type.bigwig \
     --cell-type-peak cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output
If you are use the ChromBERT Singularity image, you can run the command as follows:
.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_region \
     --region regions.bed \
     --cell-type-bw cell-type.bigwig \
     --cell-type-peak cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use existing checkpoint:

.. code-block:: bash

   chrombert-tools embed_cell_region \
     --region regions.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output
If you are use the ChromBERT Singularity image, you can run the command as follows:
.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_region \
     --region regions.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--region``
   regions of interest:BED or CSV or TSV file (CSV/TSV need with columns: chrom, start, end)

``--cell-type-bw``
   Chromatin accessibility BigWig file, if you not provide finetuned checkpoint, this file must be provided

``--cell-type-peak``
   Peak calling results in BED format, if you not provide finetuned checkpoint, this file must be provided

Optional Parameters:
-------------------

``--help``
   Show help message and exit

``--ft-ckpt``
   Path to fine-tuned checkpoint file

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``, mouse only supports 1kb resolution

``--mode``
   Training mode: ``fast`` (default) or ``full``, if ``fast`` mode is used, only the sampled 20000 regions will be used for training 

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Batch size for processing (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

Output Files
============

Training Outputs (if trained)
------------------------------

``dataset/``
   Training dataset directory
   
   * ``up_region.csv``: Regions more accessible in this cell type
   * ``nochange_region.csv``: Regions with no accessibility change

``train/try_XX_seed_YY/``
   Training outputs for attempt XX with seed YY
   
   * ``lightning_logs/*/checkpoints/*.ckpt``: Model checkpoint
   * ``eval_performance.json``: Evaluation metrics (pearsonr, spearmanr, etc.)

Embedding Outputs
-----------------

``cell_specific_overlap_region_emb.npy``
   NumPy array of cell-specific region embeddings (shape: [n_regions, 768])

``overlap_region.bed``
   Regions successfully embedded

``no_overlap_region.bed``
   Regions not found in ChromBERT
   

Tips
====

1. **Data quality**: 
   
   * Use high-quality ATAC-seq or DNase-seq data

2. **Training mode**: 
   
   * Start with ``--mode fast`` for exploration
   * Use ``--mode full`` for final results
   * Fast mode is usually sufficient for most analyses

3. **Checkpoint reuse**: 
   
   * Save checkpoints for reuse across analyses

4. **Memory errors during training**

   * Reduce ``--batch-size``