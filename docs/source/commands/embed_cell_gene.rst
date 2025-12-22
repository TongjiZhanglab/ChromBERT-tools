================
embed_cell_gene
================

Extract cell-specific gene embeddings.

Overview
========

The ``embed_cell_gene`` command fine-tunes ChromBERT on cell-type specific accessibility data (if you don't provide finetuned checkpoint, else use the finetuned checkpoint), then extracts gene embeddings using the cell-specific model. This produces embeddings that capture cell-type specific regulatory context.

Basic Usage
===========

Train new model:

.. code-block:: bash

   chrombert-tools embed_cell_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --cell-type-bw cell-type.bigwig \
     --cell-type-peak cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cell_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --cell-type-bw cell-type.bigwig \
     --cell-type-peak cell-type.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use existing checkpoint:

.. code-block:: bash

   chrombert-tools embed_cell_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:

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
   Gene names or Ensembl IDs separated by semicolons. It will be converted to lowercase for better matching, such as "BRCA1;TP53;MYC;ENSG00000170921"

``--cell-type-bw``
   Chromatin accessibility BigWig file, if you not provide finetuned checkpoint, this file must be provided

``--cell-type-peak``
   Peak calling results in BED format, if you not provide finetuned checkpoint, this file must be provided

Optional Parameters:
-------------------

``--help``
   Show help message

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
   Gene Batch size for processing (default: 4)

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

``cell_specific_gene_embs_dict.pkl``
   Dictionary mapping gene names to cell-specific embeddings
   
   .. code-block:: python
   
      import pickle
      # if you specify gene: "BRCA1;TP53;MYC;ENSG00000170921", you can get the embeddings by:
      with open('cell_specific_gene_embs_dict.pkl', 'rb') as f:
          embeddings = pickle.load(f)
      # embeddings = {'brca1': array([...]), 'tp53': array([...]), 'myc': array([...]), 'ensg00000170921': array([...]), ...}

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

5. **Gene not found**

   * Check if the gene identifier is correct
   * The gene must be listed in your ``chrombert-cache-dir/anno/*_gene_meta.tsv``
