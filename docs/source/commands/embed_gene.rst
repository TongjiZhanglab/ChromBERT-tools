==========
embed_gene
==========

Extract general gene embeddings from ChromBERT.

Overview
========

The ``embed_gene`` command extracts general (pre-trained) embeddings for user-specified genes using the pre-trained ChromBERT model. These embeddings capture gene-level regulatory context.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--gene``
   Gene symbols or Ensembl IDs separated by semicolons (e.g., ``BRCA1;TP53;MYC;ENSG00000170921``). Identifiers will be converted to lowercase for matching.

Optional Parameters
-------------------

``--help``
   Show help message.

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``. For ``mm10``, only ``1kb`` is supported.

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``.

``--odir``
   Output directory (default: ``./output``).

``--batch-size``
   Gene batch size (default: 4).

``--num-workers``
   Number of dataloader workers (default: 8).

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``). If your cache is located elsewhere, set this path accordingly.

Output Files
============

``embs_dict.pkl``
   Python dictionary mapping genes to 768-dimensional embeddings.

   .. code-block:: python

      import pickle

      # Example: if you specify --gene "BRCA1;TP53;MYC;ENSG00000170921"
      with open("embs_dict.pkl", "rb") as f:
          embeddings = pickle.load(f)

      # embeddings = {"brca1": array([...]), "tp53": array([...]), ...}

``overlap_gene_meta.tsv``
   Tab-separated file containing metadata for genes found in ChromBERT.

   Columns include: gene_name, chromosome, start, end, strand, etc.

Tips
====

1. **Gene not found**

   * Check whether the gene identifier is correct.
   * The gene must be listed in ``<chrombert-cache-dir>/anno/*_gene_meta.tsv``.
