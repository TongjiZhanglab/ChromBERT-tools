============================
infer_cell_key_regulator
============================

Infer cell-type-specific regulator-regulator networks.

Overview
========

The ``infer_cell_key_regulator`` command fine-tunes ChromBERT on cell-type-specific accessibility data (BigWig + peaks) and then infers a cell-type-specific key regulators. If a fine-tuned checkpoint is provided, fine-tuning is skipped and the regulator-regulator network is inferred directly from the checkpoint.

Basic Usage
===========

Fine-tune and infer:

.. code-block:: bash

   chrombert-tools infer_cell_key_regulator \
     --cell-type-bw /path/to/cell_type.bigwig \
     --cell-type-peak /path/to/cell_type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools infer_cell_key_regulator \
     --cell-type-bw /path/to/cell_type.bigwig \
     --cell-type-peak /path/to/cell_type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use an existing checkpoint:

.. code-block:: bash

   chrombert-tools infer_cell_key_regulator \
     --cell-type-bw /path/to/cell_type.bigwig \
     --cell-type-peak /path/to/cell_type.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools infer_cell_key_regulator \
     --cell-type-bw /path/to/cell_type.bigwig \
     --cell-type-peak /path/to/cell_type.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--cell-type-bw``
   Chromatin-accessibility BigWig file (``.bw``/``.bigWig``). Required in all cases.

``--cell-type-peak``
   Peak calls in BED or narrowPeak format. Required in all cases.

Optional Parameters
-------------------

``--help``
   Show help message.

``--ft-ckpt``
   Path to a fine-tuned checkpoint file. If provided, the tool will skip fine-tuning and infer the key regulators directly from this checkpoint.

``--mode``
   Training mode: ``fast`` (default) or ``full``. In ``fast`` mode, only 20,000 sampled regions are used for training.

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``.

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``. For ``mm10``, only ``1kb`` is supported.

``--odir``
   Output directory (default: ``./output``).

``--batch-size``
   Region batch size (default: 4).

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

   * ``highly_accessible_region.csv``: Regions more accessible in this cell type
   * ``background_region.csv``: Regions with no accessibility change

``train/try_XX_seed_YY/``
   Training outputs for attempt XX with seed YY

   * ``lightning_logs/*/checkpoints/*.ckpt``: Model checkpoint
   * ``eval_performance.json``: Evaluation metrics (pearsonr, spearmanr, etc.)

``emb/``
   Mean regulator embeddings

   * ``up_region_mean_regulator_embs_dict.pkl``: Regulator embeddings on hign accessible regions
   * ``background_region_mean_regulator_embs_dict.pkl``: Regulator embeddings on background region

``results/``
   Result files

   * ``factor_importance_rank.csv``: Ranked key regulators for this cell type

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
