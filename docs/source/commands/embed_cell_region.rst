==================
embed_cell_region
==================

Extract cell-type-specific region embeddings.

Overview
========

The ``embed_cell_region`` command fine-tunes ChromBERT on cell-type-specific accessibility data (BigWig + peaks) and then extracts region embeddings from the fine-tuned model. If a fine-tuned checkpoint is provided, fine-tuning is skipped and embeddings are generated directly from the checkpoint. The resulting embeddings reflect cell-type-specific regulatory patterns.

Basic Usage
===========

Fine-tune a new model:

.. code-block:: bash

   chrombert-tools embed_cell_region \
     --region regions.bed \
     --cell-type-bw /path/to/cell-type.bigwig \
     --cell-type-peak /path/to/cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_region \
     --region regions.bed \
     --cell-type-bw /path/to/cell-type.bigwig \
     --cell-type-peak /path/to/cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use an existing checkpoint:

.. code-block:: bash

   chrombert-tools embed_cell_region \
     --region regions.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

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
   Regions of interest in BED/CSV/TSV format. For CSV/TSV, the file must contain columns: ``chrom``, ``start``, ``end``.

``--cell-type-bw``
   Chromatin-accessibility BigWig file (``.bw``/``.bigWig``). Required if ``--ft-ckpt`` is not provided.

``--cell-type-peak``
   Peak calls in BED or narrowPeak format. Required if ``--ft-ckpt`` is not provided.

Optional Parameters
-------------------

``--help``
   Show help message.

``--ft-ckpt``
   Path to a fine-tuned checkpoint file. If provided, the tool will skip fine-tuning and use this checkpoint directly to generate region embeddings. In this case, ``--cell-type-bw`` and ``--cell-type-peak`` are not required.

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``.

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``. For ``mm10``, only ``1kb`` is supported.

``--mode``
   Training mode: ``fast`` (default) or ``full``. In ``fast`` mode, only 20,000 sampled regions are used for training.

``--odir``
   Output directory (default: ``./output``).

``--batch-size``
   Batch size for processing (default: 4).

``--num-workers``
   Number of dataloader workers (default: 8).

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``). If your cache is located elsewhere, set this path accordingly.

Output Files
============

Training Outputs (if trained)
-----------------------------

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
   NumPy array of cell-type-specific region embeddings (shape: ``[n_regions, 768]``).

``overlap_region.bed``
   Regions successfully embedded.

``no_overlap_region.bed``
   Regions not found in ChromBERT.

Tips
====

1. **Data quality**

   * Use high-quality ATAC-seq or DNase-seq data.

2. **Training mode**

   * Start with ``--mode fast`` for exploration.
   * Use ``--mode full`` for final results.
   * Fast mode is usually sufficient for most analyses.

3. **Checkpoint reuse**

   * Save checkpoints for reuse across analyses.

4. **Memory errors during training**

   * Reduce ``--batch-size``.
