================
embed_cell_gene
================

Extract cell-type-specific gene embeddings.

Overview
========

The ``embed_cell_gene`` command fine-tunes ChromBERT on cell-type-specific accessibility data (bigwig + peaks) and then generates gene embeddings from the fine-tuned model. If a fine-tuned checkpoint is provided, fine-tuning is skipped and embeddings are generated directly from the checkpoint. The resulting embeddings capture cell-type-specific regulatory context.

Basic Usage
===========

Fine-tune a new model:

.. code-block:: bash

   chrombert-tools embed_cell_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --cell-type-bw /path/to/cell-type.bigwig \
     --cell-type-peak /path/to/cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --cell-type-bw /path/to/cell-type.bigwig \
     --cell-type-peak /path/to/cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use an existing checkpoint:

.. code-block:: bash

   chrombert-tools embed_cell_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output


Parameters
==========

Required Parameters
-------------------

``--gene``
   Gene symbols or Ensembl IDs separated by semicolons (e.g., ``BRCA1;TP53;MYC;ENSG00000170921``). Identifiers will be converted to lowercase for matching.

``--cell-type-bw``
   Chromatin-accessibility BigWig file (``.bw``/``.bigWig``). Required if ``--ft-ckpt`` is not provided.

``--cell-type-peak``
   Peak calls in BED or narrowPeak format. Required if ``--ft-ckpt`` is not provided.

Optional Parameters
-------------------

``--help``
   Show help message.

``--ft-ckpt``
   Path to a fine-tuned checkpoint file. If provided, the tool will skip fine-tuning and use this checkpoint directly to generate gene embeddings. In this case, ``--cell-type-bw`` and ``--cell-type-peak`` are not required.

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

``cell_specific_gene_embs_dict.pkl``
   Dictionary mapping genes to cell-type-specific embeddings.

   .. code-block:: python

      import pickle

      # Example: if you specify --gene "BRCA1;TP53;MYC;ENSG00000170921"
      with open("cell_specific_gene_embs_dict.pkl", "rb") as f:
          embeddings = pickle.load(f)

      # embeddings = {"brca1": array([...]), "tp53": array([...]), ...}

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

5. **Gene not found**

   * Check whether the gene identifier is correct.
   * The gene must be listed in ``<chrombert-cache-dir>/anno/*_gene_meta.tsv``.
