====================
embed_cell_cistrome
====================

Extract cell-specific cistrome embeddings on specified regions.

Overview
========

The ``embed_cell_cistrome`` command fine-tunes ChromBERT on cell-type specific accessibility data (if you don't provide finetuned checkpoint, else use the finetuned checkpoint), then extracts cistrome embeddings using the cell-specific model.

Basic Usage
===========

Train new model:

.. code-block:: bash
   chrombert-tools embed_cell_cistrome \
     --cistrome "cistrome1;cistrome2" \
     --region regions.bed \
     --cell-type-bw cell-type.bigwig \
     --cell-type-peak cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output \

If you are use the ChromBERT Singularity image, you can run the command as follows:
.. code-block:: bash
   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_cistrome \
     --cistrome "cistrome1;cistrome2" \
     --region region.bed \
     --cell-type-bw cell-type.bigwig \
     --cell-type-peak cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use existing checkpoint:

.. code-block:: bash
   chrombert-tools embed_cell_cistrome \
     --cistrome "cistrome1;cistrome2" \
     --region regions.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:
.. code-block:: bash
   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_cistrome \
     --cistrome "cistrome1;cistrome2" \
     --region region.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--cistrome``
   Cistrome identifiers: GSM/ENCODE IDs or factor:cell pairs, use ; to separate multiple cistromes. It will be converted to lowercase for better matching, such as "CTCF:K562;H3K27ac:K562;GSM1208591"

``--region``
   regions of interest: BED or CSV or TSV file (CSV/TSV need with columns: chrom, start, end)

``--cell-type-bw``
   Chromatin accessibility BigWig file, if you not provide finetuned checkpoint, this file must be provided for training new model

``--cell-type-peak``
   Peak calling results in BED format, if you not provide finetuned checkpoint, this file must be provided for training new model


Optional Parameters
-------------------
``--help``
   Show help message

``--ft-ckpt``
   Path to fine-tuned checkpoint file, if you provide this file, you don't need to provide ``--cell-type-bw`` and ``--cell-type-peak``

``--mode``
   Training mode: ``fast`` (default) or ``full``, if ``fast`` mode is used, only the sampled 20000 regions will be used for training 

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

``cell_specific_cistrome_emb_on_region.hdf5``
   HDF5 file with cell-specific cistrome embeddings per region
   .. code-block:: python
      import python
      # if you specify cistrome: "CTCF:K562;H3K27ac:K562;GSM1208591", you can get the embeddings by:
      with h5py.File('cell_specific_cistrome_emb_on_region.hdf5', 'r') as f:
          emb1 = f['/emb/ctcf:k562'][:]
          emb2 = f['/emb/h3k27ac:k562'][:]
          emb3 = f['/emb/gsm1208591'][:]

``cell_specific_mean_cistrome_emb.pkl``
   Mean cell-specific cistrome embeddings
   .. code-block:: python
   
      import pickle
      # if you specify cistrome: "CTCF:K562;H3K27ac:K562;GSM1208591", you can get the embeddings by:
      with open('cell_specific_mean_cistrome_emb.pkl', 'rb') as f:
          mean_embeddings = pickle.load(f)
      # mean_embeddings = {'ctcf:k562': array([...]), 'h3k27ac:k562': array([...]), 'gsm1208591': array([...]), ...}

``overlap_region.bed``
   Regions overlap with chrombert regions (your chrombert-cache-dir/config/*region.bed)

``no_overlap_region.bed``
   Regions not overlap with chrombert regions (your chrombert-cache-dir/config/*region.bed)

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

5. **Cistrome not found**

   * Check if the cistrome identifier is correct
   * Check if the cistrome identifier is in the correct format
   * Replace cistrome with GSM IDs or ENCODE accessions (used in ChromBERT).
   * The cistrome must be listed in your ``chrombert-cache-dir/config/*_meta.tsv``
