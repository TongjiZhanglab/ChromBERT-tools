=====================
embed_cell_regulator
=====================

Generate cell-type-specific regulator embeddings.

Overview
========

The ``embed_cell_regulator`` command fine-tunes ChromBERT on cell-type-specific accessibility data and then generates regulator embeddings from the fine-tuned model. If a fine-tuned checkpoint is provided, fine-tuning is skipped and embeddings are generated directly from the checkpoint.

Basic Usage
===========

Fine-tune a new model:

.. code-block:: bash

   chrombert-tools embed_cell_regulator \
     --regulator "regulator1;regulator2;regulator3" \
     --region regions.bed \
     --cell-type-bw /path/to/cell-type.bigwig \
     --cell-type-peak /path/to/cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_regulator \
     --regulator "regulator1;regulator2;regulator3" \
     --region regions.bed \
     --cell-type-bw /path/to/cell-type.bigwig \
     --cell-type-peak /path/to/cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use existing checkpoint:

.. code-block:: bash

   chrombert-tools embed_cell_regulator \
     --regulator "regulator1;regulator2;regulator3" \
     --region regions.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_regulator \
     --regulator "regulator1;regulator2;regulator3" \
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
   Regulator names separated by semicolons (e.g., CTCF;MYC;TP53). Names will be converted to lowercase for matching.

``--region``
   Regions of interest in BED/CSV/TSV format. For CSV/TSV, the file must contain columns: ``chrom``, ``start``, ``end``.


``--cell-type-bw``
   Chromatin-accessibility BigWig file (.bw/.bigWig). Required if ``--ft-ckpt`` is not provided.

``--cell-type-peak``
   Peak calls in BED or narrowPeak format. Required if ``--ft-ckpt`` is not provided.


Optional Parameters
-------------------

``--help``
   Show help message

``--ft-ckpt``
   Path to a fine-tuned checkpoint file. If provided, the tool will skip fine-tuning and use this checkpoint directly to generate regulator embeddings. In this case, ``--cell-type-bw`` and ``--cell-type-peak`` are not required.

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``. For mm10, only 1kb is supported.

``--mode``
   Training mode: ``fast`` (default) or ``full``. In ``fast`` mode, only 20,000 sampled regions are used for training.

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Batch size for processing (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``). If your cache is located elsewhere, set this path accordingly.


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

``cell_specific_regulator_emb_on_region.hdf5``
   HDF5 file with cell-type-specific regulator embeddings per region

   .. code-block:: python

      import h5py
      with h5py.File('cell_specific_regulator_emb_on_region.hdf5', 'r') as f:
          ctcf_emb = f['/emb/ctcf'][:]
          myc_emb = f['/emb/myc'][:]
          tp53_emb = f['/emb/tp53'][:]

``cell_specific_mean_regulator_emb.pkl``
   Mean cell-type-specific regulator embeddings

   .. code-block:: python
      
      import pickle
      with open('cell_specific_mean_regulator_emb.pkl', 'rb') as f:
          mean_embeddings = pickle.load(f)
      # mean_embeddings = {'ezh2': array([...]), 'myc': array([...]), 'tp53': array([...]), ...}

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found

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

5. **Regulator not found**

   * Check if the regulator is correct
   * You can find all regulators in your <chrombert-cache-dir>/config/*_regulator_list.txt

