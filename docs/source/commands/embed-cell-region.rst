==================
embed-cell-region
==================

Extract cell-specific region embeddings.

Overview
========

The ``embed-cell-region`` command fine-tunes ChromBERT on cell-type specific accessibility data, then extracts region embeddings using the cell-specific model. This produces embeddings that reflect cell-type specific regulatory patterns.

Basic Usage
===========

Train new model:

.. code-block:: bash

   chrombert-tools embed-cell-region \
     --region regions.bed \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use existing checkpoint:

.. code-block:: bash

   chrombert-tools embed-cell-region \
     --region regions.bed \
     --ft-ckpt /path/to/checkpoint.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--region``
   BED or CSV file specifying regions of interest

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Conditional Parameters
----------------------

**If training** (no ``--ft-ckpt``):

``--cell-type-bw``
   Chromatin accessibility BigWig file

``--cell-type-peak``
   Peak calling results in BED format

**If using existing model**:

``--ft-ckpt``
   Path to fine-tuned checkpoint file

Optional Parameters
-------------------

``--mode``
   Training mode: ``fast`` (default) or ``full``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Batch size (default: 4)

``--num-workers``
   Dataloader workers (default: 8)

Output Files
============

``cell_specific_overlap_region_emb.npy``
   NumPy array of cell-specific region embeddings (shape: [n_regions, 768])

``overlap_region.bed``
   Regions successfully embedded

``no_overlap_region.bed``
   Regions not found in ChromBERT

``dataset/`` (if trained)
   Training dataset

``train/`` (if trained)
   Model checkpoint and evaluation metrics

Examples
========

Enhancer Analysis
-----------------

.. code-block:: bash

   chrombert-tools embed-cell-region \
     --region enhancers.bed \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --mode fast \
     --genome hg38 \
     --resolution 1kb \
     --odir K562_enhancer_embeddings

Promoter Analysis
-----------------

.. code-block:: bash

   chrombert-tools embed-cell-region \
     --region promoters.bed \
     --cell-type-bw Neuron_ATAC.bigwig \
     --cell-type-peak Neuron_peaks.bed \
     --mode full \
     --genome hg38 \
     --resolution 1kb \
     --odir Neuron_promoter_embeddings

Using Embeddings
================

Cluster Regions
---------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.cluster import KMeans
   import matplotlib.pyplot as plt
   
   # Load embeddings
   embeddings = np.load('output/cell_specific_overlap_region_emb.npy')
   regions = pd.read_csv('output/overlap_region.bed', sep='\t', header=None)
   regions.columns = ['chr', 'start', 'end']
   
   # Cluster regions
   n_clusters = 5
   kmeans = KMeans(n_clusters=n_clusters, random_state=42)
   clusters = kmeans.fit_predict(embeddings)
   regions['cluster'] = clusters
   
   # Save
   for i in range(n_clusters):
       cluster_regions = regions[regions['cluster'] == i]
       cluster_regions[['chr', 'start', 'end']].to_csv(
           f'cluster_{i}_regions.bed', 
           sep='\t', header=False, index=False
       )

See Also
========

* :doc:`embed-region` - General region embeddings
* :doc:`embed-cell-gene` - Cell-specific gene embeddings
* :doc:`overview` - Common parameters

