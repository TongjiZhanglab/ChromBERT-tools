==========
embed-gene
==========

Extract general gene embeddings from ChromBERT.

Overview
========

The ``embed-gene`` command extracts 768-dimensional embeddings for specified genes using the pre-trained ChromBERT model. These embeddings capture regulatory context and can be used for downstream analysis.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed-gene \
     --gene "BRCA1;TP53;MYC" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--gene``
   Gene names separated by semicolons
   
   Example: ``"BRCA1;TP53;MYC;EGFR;KRAS"``

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Optional Parameters
-------------------

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``
   
   Note: mm10 only supports ``1kb``

``--odir``
   Output directory (default: ``./output``)

``--chrombert-cache-dir``
   ChromBERT cache directory
   
   Default: ``~/.cache/chrombert/data``

``--batch-size``
   Batch size for processing (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

Output Files
============

``embs_dict.pkl``
   Python dictionary mapping gene names to 768-dimensional embeddings
   
   .. code-block:: python
   
      import pickle
      with open('embs_dict.pkl', 'rb') as f:
          embeddings = pickle.load(f)
      # embeddings = {'BRCA1': array([...]), 'TP53': array([...]), ...}

``overlap_gene_meta.tsv``
   Tab-separated file containing metadata for genes found in ChromBERT
   
   Columns: gene_name, chromosome, start, end, strand, etc.

Examples
========

Basic Example
-------------

Extract embeddings for a few genes:

.. code-block:: bash

   chrombert-tools embed-gene \
     --gene "BRCA1;TP53;MYC" \
     --genome hg38 \
     --resolution 1kb \
     --odir gene_embeddings

Large Gene List
---------------

Extract embeddings for many genes:

.. code-block:: bash

   chrombert-tools embed-gene \
     --gene "BRCA1;TP53;MYC;EGFR;KRAS;PTEN;AKT1;BRAF;PIK3CA;RB1" \
     --genome hg38 \
     --resolution 1kb \
     --odir cancer_gene_embeddings

Mouse Genes
-----------

.. code-block:: bash

   chrombert-tools embed-gene \
     --gene "Myc;Trp53;Brca1" \
     --genome mm10 \
     --resolution 1kb \
     --odir mouse_gene_embeddings

Tips
====

1. **Gene naming**: 
   
   * Use official gene symbols or ensembl id (all gene name will be converted to lowercase for better matching)
   * Case-sensitive: use exact gene names
   * Check ``overlap_gene_meta.tsv`` for genes not found


Troubleshooting
===============

**Gene not found**

* Check gene name spelling and case
* Verify gene exists in the reference genome
* Check ``overlap_gene_meta.tsv`` for available genes 
* Verify gene exists in the reference genome (~/.cache/chrombert/data/anno/*_gene_meta.tsv)
* Try alternative gene names/aliases

**Empty output**

* Verify input gene names are correct
* Check that ChromBERT data is properly installed
* Ensure genome and resolution are supported

**Memory errors**

* Reduce ``--batch-size``
* Process genes in smaller groups
* Close other applications


See Also
========

* :doc:`overview` - Common parameters and file formats
* :doc:`../installation` - Installation guide

