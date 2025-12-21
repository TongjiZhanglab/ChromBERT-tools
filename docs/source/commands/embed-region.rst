============
embed-region
============

Extract general region embeddings from ChromBERT.

Overview
========

The ``embed-region`` command extracts 768-dimensional embeddings for specified genomic regions using the pre-trained ChromBERT model. These embeddings capture regulatory patterns and can be used for region classification, clustering, or other downstream analyses.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed-region \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--region``
   BED or CSV file specifying regions of interest
   
   * BED format: tab-separated (chr, start, end, ...)
   * CSV format: comma-separated with columns chr, start, end
   
   Example BED:
   
   .. code-block:: text
   
      chr1    1000    2000
      chr2    5000    6000
   
   Example CSV:
   
   .. code-block:: text
   
      chr,start,end
      chr1,1000,2000
      chr2,5000,6000

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

``overlap_region_emb.npy``
   NumPy array containing region embeddings (shape: [n_regions, 768])
   
   .. code-block:: python
   
      import numpy as np
      
      # Load embeddings
      embeddings = np.load('overlap_region_emb.npy')
      print(f"Shape: {embeddings.shape}")  # (n_regions, 768)

``overlap_region.bed``
   BED file containing regions successfully embedded
   
   These regions were found in the ChromBERT database

``no_overlap_region.bed``
   BED file containing regions not found in ChromBERT
   
   These regions were outside the ChromBERT coverage

Examples
========

Basic Example
-------------

Extract embeddings for enhancers:

.. code-block:: bash

   chrombert-tools embed-region \
     --region enhancers.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir enhancer_embeddings

High Resolution
---------------

Use 200bp resolution for finer detail:

.. code-block:: bash

   chrombert-tools embed-region \
     --region promoters.bed \
     --genome hg38 \
     --resolution 200bp \
     --odir promoter_embeddings_200bp

Large Region Set
----------------

Process many regions with increased batch size:

.. code-block:: bash

   chrombert-tools embed-region \
     --region large_regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --batch-size 16 \
     --num-workers 16 \
     --odir large_region_embeddings

Using Embeddings
================

Python Example
--------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.cluster import KMeans
   
   # Load embeddings and regions
   embeddings = np.load('output/overlap_region_emb.npy')
   regions = pd.read_csv('output/overlap_region.bed', sep='\t', header=None)
   regions.columns = ['chr', 'start', 'end']
   
   # Cluster regions
   kmeans = KMeans(n_clusters=5)
   clusters = kmeans.fit_predict(embeddings)
   regions['cluster'] = clusters
   
   # Save clustered regions
   regions.to_csv('clustered_regions.bed', sep='\t', header=False, index=False)

Tips
====

1. **Region format**: 
   
   * Ensure proper chromosome naming (chr1 vs 1)
   * Start coordinates should be less than end coordinates
   * Remove header lines from BED files
   * Use 0-based coordinates (BED standard)

2. **Performance**: 
   
   * Increase ``--batch-size`` if you have GPU memory
   * Increase ``--num-workers`` for faster data loading
   * Process regions in chunks if dataset is very large

3. **Resolution selection**: 
   
   * 1kb: Best balance of resolution and data availability
   * 200bp: Higher resolution but slower
   * 2kb/4kb: Faster but less detailed

Troubleshooting
===============

**Many regions in no_overlap_region.bed**

* Check chromosome naming (chr1 vs 1)
* Verify genome build matches your regions (hg19 vs hg38)
* Some regions may be outside ChromBERT coverage
* Try different resolution

**Memory errors**

* Reduce ``--batch-size`` (try 2 or 1)
* Process regions in smaller chunks
* Use lower resolution (2kb instead of 200bp)

**Empty output**

* Verify BED/CSV file format
* Check file has no header (or use CSV with header)
* Ensure coordinates are valid (start < end)

See Also
========

* :doc:`embed-gene` - Extract gene embeddings
* :doc:`embed-regulator` - Extract regulator embeddings
* :doc:`embed-cell-region` - Extract cell-specific region embeddings
* :doc:`overview` - Common parameters and file formats
