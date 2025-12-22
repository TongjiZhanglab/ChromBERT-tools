==============
infer_cell_trn
==============

Infer cell-type specific transcriptional regulatory network (TRN).

Overview
========

The ``infer_cell_trn`` command fine-tunes ChromBERT on cell-specific accessibility data (if you don't provide finetuned checkpoint, else use the finetuned checkpoint), then infers a cell-type specific regulatory network and cell-specific regulators.

Basic Usage
===========

Train and infer:

.. code-block:: bash

   chrombert-tools infer_cell_trn \
     --cell-type-bw cell_type.bigwig \
     --cell-type-peak cell_type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:
.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools infer_cell_trn \
     --cell-type-bw cell_type.bigwig \
     --cell-type-peak cell_type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use existing checkpoint:

.. code-block:: bash

   chrombert-tools infer_cell_trn \
      --cell-type-bw cell_type.bigwig \
      --cell-type-peak cell_type.bed \
      --ft-ckpt /path/to/checkpoint.ckpt \
      --genome hg38 \
      --resolution 1kb \
      --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:
.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools infer_cell_trn \
     --cell-type-bw cell_type.bigwig \
     --cell-type-peak cell_type.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--cell-type-bw``
   Chromatin accessibility BigWig file, required in all cases

``--cell-type-peak``
   Peak calling results in BED format, required in all cases


Optional Parameters
-------------------

``--help``
   Show help message and exit

``--ft-ckpt``
   Path to fine-tuned checkpoint file, whether you provide this file, you need to provide ``--cell-type-bw`` and ``--cell-type-peak``

``--mode``
   Training mode: ``fast`` (default) or ``full``, if ``fast`` mode is used, only the sampled 20000 regions will be used for training 

``--regulator``
   You want to plot the subnetwork for this regulator

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``, mouse only supports 1kb resolution

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Region batch size (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

``--quantile``
   Quantile threshold for cosine similarity edges (default: 0.99)

``--k-hot``
   k-hop radius for subnetwork plotting (default: 1)

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

``emb/``
   Mean regulator embeddings
   * ``up_region_mean_regulator_embs_dict.pkl``: Regulator embeddings on up regions
   * ``nochange_region_mean_regulator_embs_dict.pkl``: Regulator embeddings on nochange regions

``results``
   * ``factor_importance_rank.csv``: Key regulators in this specific cell-type
   * ``regulator_cosine_similarity.tsv``: regulator-regulator cosine similarity on up region, more higher value, more similarity function for this pair regulators
   * ``subnetwork_regulator_k*.pdf``
   regulator subnetwork on this regions

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