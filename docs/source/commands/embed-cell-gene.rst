================
embed-cell-gene
================

Extract cell-specific gene embeddings.

Overview
========

The ``embed-cell-gene`` command first fine-tunes ChromBERT on cell-type specific accessibility data, then extracts gene embeddings using the cell-specific model. This produces embeddings that capture cell-type specific regulatory context.

Basic Usage
===========

Train new model:

.. code-block:: bash

   chrombert-tools embed-cell-gene \
     --gene "BRCA1;TP53;MYC" \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use existing checkpoint:

.. code-block:: bash

   chrombert-tools embed-cell-gene \
     --gene "BRCA1;TP53;MYC" \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--gene``
   Gene names separated by semicolons

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Conditional Parameters
----------------------

**If training a new model** (no ``--ft-ckpt``):

``--cell-type-bw``
   Chromatin accessibility BigWig file
   
   Should be normalized (e.g., CPM, RPKM)

``--cell-type-peak``
   Peak calling results in BED format
   
   Standard 3-column or MACS2 format

**If using existing model**:

``--ft-ckpt``
   Path to fine-tuned checkpoint file
   
   Example: ``output/train/try_00_seed_55/lightning_logs/*/checkpoints/epoch=*.ckpt``

Optional Parameters
-------------------

``--mode``
   Training mode (default: ``fast``)
   
   * ``fast``: Downsample to 20k regions for training
   * ``full``: Use all regions (slower but more accurate)

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Batch size for processing (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``)

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
      
      with open('cell_specific_gene_embs_dict.pkl', 'rb') as f:
          embeddings = pickle.load(f)
      # embeddings = {'BRCA1': array([...]), 'TP53': array([...]), ...}

Examples
========

K562 Cell Line
--------------

.. code-block:: bash

   chrombert-tools embed-cell-gene \
     --gene "BRCA1;TP53;MYC;EGFR;KRAS" \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --mode fast \
     --genome hg38 \
     --resolution 1kb \
     --odir K562_gene_embeddings

Primary Cells
-------------

.. code-block:: bash

   chrombert-tools embed-cell-gene \
     --gene "CD4;CD8A;IL2;IFNG" \
     --cell-type-bw Tcell_ATAC.bigwig \
     --cell-type-peak Tcell_peaks.bed \
     --mode full \
     --genome hg38 \
     --resolution 1kb \
     --odir Tcell_gene_embeddings

Reuse Checkpoint
----------------

Train once:

.. code-block:: bash

   chrombert-tools embed-cell-gene \
     --gene "BRCA1;TP53" \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir K562_model

Reuse for more genes:

.. code-block:: bash

   CKPT="K562_model/train/try_00_seed_55/lightning_logs/version_0/checkpoints/epoch=8-step=917.ckpt"
   
   chrombert-tools embed-cell-gene \
     --gene "MYC;EGFR;KRAS;PTEN;AKT1" \
     --ft-ckpt $CKPT \
     --genome hg38 \
     --resolution 1kb \
     --odir K562_more_genes

Using Embeddings
================

Compare Cell Types
------------------

.. code-block:: python

   import pickle
   import numpy as np
   from scipy.spatial.distance import cosine
   
   # Load embeddings from two cell types
   with open('K562/cell_specific_gene_embs_dict.pkl', 'rb') as f:
       k562_embs = pickle.load(f)
   
   with open('GM12878/cell_specific_gene_embs_dict.pkl', 'rb') as f:
       gm12878_embs = pickle.load(f)
   
   # Compare same gene across cell types
   gene = 'BRCA1'
   similarity = 1 - cosine(k562_embs[gene], gm12878_embs[gene])
   print(f"{gene} similarity between K562 and GM12878: {similarity:.3f}")
   
   # Find genes with different embeddings
   common_genes = set(k562_embs.keys()) & set(gm12878_embs.keys())
   
   differences = {}
   for gene in common_genes:
       dist = cosine(k562_embs[gene], gm12878_embs[gene])
       differences[gene] = dist
   
   # Top genes with different regulatory context
   import pandas as pd
   diff_df = pd.DataFrame(list(differences.items()),
                          columns=['gene', 'distance'])
   diff_df = diff_df.sort_values('distance', ascending=False)
   print("Genes with most different regulatory context:")
   print(diff_df.head(10))

Tips
====

1. **Data quality**: 
   
   * Use high-quality ATAC-seq or DNase-seq data
   * Ensure proper peak calling (MACS2 recommended)
   * Normalize BigWig files (CPM or RPKM)
   * Remove duplicates and blacklisted regions

2. **Training mode**: 
   
   * Start with ``--mode fast`` for exploration
   * Use ``--mode full`` for final publication results
   * Fast mode is usually sufficient for most analyses

3. **Checkpoint reuse**: 
   
   * Save checkpoints for reuse across analyses
   * One checkpoint per cell type can be reused multiple times
   * Significant time savings for large projects

4. **Gene selection**: 
   
   * Process all genes of interest in one run when possible
   * Embeddings for different genes are independent
   * Can process genes in batches if needed

Troubleshooting
===============

**Training fails or unstable**

* Check data quality (peaks, BigWig)
* Ensure BigWig has sufficient coverage
* Try different seed (automatic retry mechanism)
* Use ``--mode fast`` for testing

**Low evaluation performance**

* Check eval_performance.json for metrics
* pearsonr < 0.2 indicates poor model quality
* May need better quality accessibility data
* Consider using general embeddings if cell-specific fails

**Memory errors during training**

* Reduce ``--batch-size``
* Use ``--mode fast`` (uses less data)
* Close other applications
* Use machine with more RAM/GPU memory

**Checkpoint file not found**

* Check exact path to checkpoint file
* Look in ``train/try_*/lightning_logs/*/checkpoints/``
* Use tab completion or find command
* Checkpoint filename includes epoch and step numbers

See Also
========

* :doc:`embed-gene` - General gene embeddings
* :doc:`embed-cell-region` - Cell-specific region embeddings
* :doc:`embed-cell-regulator` - Cell-specific regulator embeddings
* :doc:`overview` - Common parameters

