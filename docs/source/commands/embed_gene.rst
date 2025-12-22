==========
embed_gene
==========

Extract general gene embeddings from ChromBERT.

Overview
========

The ``embed_gene`` command extracts general embeddings for specified genes using the pre-trained ChromBERT model. These embeddings capture regulatory context.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed_gene \
     --gene "gene1;gene2;gene3;gene4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:

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
   Gene names or Ensembl IDs separated by semicolons. It will be converted to lowercase for better matching, such as "BRCA1;TP53;MYC;ENSG00000170921"

Optional Parameters
-------------------

``--help``
   Show help message

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Gene batch size (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

Output Files
============

``embs_dict.pkl``
   Python dictionary mapping gene names to 768-dimensional embeddings
   
   .. code-block:: python
   
      import pickle
      # if you specify gene: "BRCA1;TP53;MYC;ENSG00000170921", you can get the embeddings by:
      with open('embs_dict.pkl', 'rb') as f:
          embeddings = pickle.load(f)
      # embeddings = {'brca1': array([...]), 'tp53': array([...]), 'myc': array([...]), 'ensg00000170921': array([...]), ...}

``overlap_gene_meta.tsv``
   Tab-separated file containing metadata for genes found in ChromBERT
   
   Columns: gene_name, chromosome, start, end, strand, etc.


Tips
===============

1. **Gene not found**

   * Check if the gene identifier is correct
   * The gene must be listed in your ``chrombert-cache-dir/anno/*_gene_meta.tsv``

