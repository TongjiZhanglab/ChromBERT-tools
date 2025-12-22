============
embed_region
============

Extract general region embeddings from ChromBERT.

Overview
========

The ``embed_region`` command extracts 768-dimensional embeddings for specified genomic regions using the pre-trained ChromBERT model. These embeddings capture regulatory patterns.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:
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
   regions of interest: BED or CSV or TSV file (CSV/TSV need with columns: chrom, start, end)

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
   Region batch size (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

Output Files
============

``overlap_region_emb.npy``
   NumPy array containing region embeddings (shape: [n_regions, 768])
   
   .. code-block:: python
   
      import numpy as np
      
      # Load embeddings
      embeddings = np.load('overlap_region_emb.npy')
      print(f"Shape: {embeddings.shape}")  # (n_regions, 768)

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found in ChromBERT

