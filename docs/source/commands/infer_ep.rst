========
infer_ep
========

Infer enhancer–promoter loop scores on specified regions.

Overview
========

The ``infer_ep`` command uses the pre-trained ChromBERT model to compute cosine similarity between region embeddings and gene TSS embeddings within a window.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools infer_ep \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools infer_ep \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--region``
   Regions of interest in BED/CSV/TSV format. For CSV/TSV, the file must contain columns: ``chrom``, ``start``, ``end``.

Optional Parameters
-------------------

``--help``
   Show help message.

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``.

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``. For ``mm10``, only ``1kb`` is supported.

``--odir``
   Output directory (default: ``./output``).

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``). If your cache is located elsewhere, set this path accordingly.

``--chrombert-region-file``
   ChromBERT region BED file. If not provided, use the default region BED in the cache directory.

``--chrombert-region-emb-file``
   ChromBERT region embedding file (.npy). If not provided, use the default region embedding in the cache directory.

Output Files
============

``model_input.tsv``
   Combined and de-duplicated regions used for embedding (focus regions + gene TSS regions).

``use_region_emb.npy``
   Embeddings of the combined regions used in this run.

``tss_region_pairs_cos.tsv``
   Cosine similarity between TSS and region pairs within the window.
