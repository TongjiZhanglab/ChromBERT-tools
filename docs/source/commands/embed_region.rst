============
embed_region
============

Extract general region embeddings from ChromBERT.

Overview
========

The ``embed_region`` command extracts 768-dimensional embeddings for user-specified genomic regions using the pre-trained ChromBERT model. These embeddings capture general regulatory patterns.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_region \
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

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``. For ``mm10``, only ``1kb`` is supported.

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``.

``--odir``
   Output directory (default: ``./output``).

``--batch-size``
   Region batch size (default: 4).

``--num-workers``
   Number of dataloader workers (default: 8).

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``). If your cache is located elsewhere, set this path accordingly.

Output Files
============

``overlap_region_emb.npy``
   NumPy array containing region embeddings (shape: ``[n_regions, 768]``).

   .. code-block:: python

      import numpy as np

      embeddings = np.load("overlap_region_emb.npy")
      print(embeddings.shape)  # (n_regions, 768)

``overlap_region.bed``
   Regions that overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).

``no_overlap_region.bed``
   Regions that do not overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).
