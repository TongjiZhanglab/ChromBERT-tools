===============
embed-cistrome
===============

Extract embeddings for cistromes (ChIP-seq/ATAC-seq datasets).

Overview
========

The ``embed-cistrome`` command extracts embeddings for specified cistrome datasets (ChIP-seq, ATAC-seq, etc.) across genomic regions. Cistromes can be specified using GSM IDs, ENCODE IDs, or factor:celltype pairs.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed-cistrome \
     --cistrome "CTCF:K562;H3K27ac:K562" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--cistrome``
   Cistrome identifiers separated by semicolons
   
   Supported formats:
   
   * **GSM IDs**: ``"GSM1208591;GSM1234567"``
   * **ENCODE IDs**: ``"ENCSR440VKE_2;ENCSR123ABC"``
   * **Factor:Cell pairs**: ``"CTCF:K562;H3K27ac:K562;ATAC-seq:HEK293T"``
   * **Mixed**: ``"GSM1208591;CTCF:K562;ENCSR440VKE_2"``

``--region``
   BED or CSV file specifying regions of interest

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Optional Parameters
-------------------

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``)

``--batch-size``
   Batch size for processing (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

Output Files
============

``cistrome_emb_on_region.hdf5``
   HDF5 file containing cistrome embeddings for each region
   
   .. code-block:: python
   
      import h5py
      
      with h5py.File('cistrome_emb_on_region.hdf5', 'r') as f:
          # List available cistromes
          print(list(f['emb'].keys()))
          
          # Load specific cistrome
          ctcf_emb = f['emb/CTCF:K562'][:]  # Shape: (n_regions, 768)
          regions = f['region'][:]

``mean_cistrome_emb.pkl``
   Python dictionary containing mean embeddings for each cistrome
   
   .. code-block:: python
   
      import pickle
      
      with open('mean_cistrome_emb.pkl', 'rb') as f:
          mean_embs = pickle.load(f)
      # mean_embs = {'CTCF:K562': array([...]), ...}

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found in ChromBERT

Examples
========

Using Factor:Cell Pairs
------------------------

.. code-block:: bash

   chrombert-tools embed-cistrome \
     --cistrome "CTCF:K562;H3K27ac:K562;ATAC-seq:K562" \
     --region promoters.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir k562_cistromes

Using GSM IDs
-------------

.. code-block:: bash

   chrombert-tools embed-cistrome \
     --cistrome "GSM1208591;GSM1234567;GSM9876543" \
     --region enhancers.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir gsm_cistromes

Using ENCODE IDs
----------------

.. code-block:: bash

   chrombert-tools embed-cistrome \
     --cistrome "ENCSR440VKE_2;ENCSR123ABC;ENCSR456DEF" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir encode_cistromes

Mixed Identifiers
-----------------

.. code-block:: bash

   chrombert-tools embed-cistrome \
     --cistrome "CTCF:K562;GSM1208591;ENCSR440VKE_2;H3K27ac:HepG2" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir mixed_cistromes

Using Embeddings
================

Cistrome Similarity Analysis
-----------------------------

.. code-block:: python

   import pickle
   import numpy as np
   from scipy.spatial.distance import pdist, squareform
   import seaborn as sns
   import matplotlib.pyplot as plt
   
   # Load mean embeddings
   with open('output/mean_cistrome_emb.pkl', 'rb') as f:
       mean_embs = pickle.load(f)
   
   # Create similarity matrix
   names = list(mean_embs.keys())
   embs = np.array([mean_embs[n] for n in names])
   
   # Calculate pairwise cosine similarity
   distances = pdist(embs, metric='cosine')
   similarity = 1 - squareform(distances)
   
   # Plot heatmap
   plt.figure(figsize=(10, 8))
   sns.heatmap(similarity, xticklabels=names, yticklabels=names,
               cmap='RdYlBu_r', center=0.5)
   plt.title('Cistrome Similarity')
   plt.tight_layout()
   plt.savefig('cistrome_similarity.png')

Region-specific Patterns
-------------------------

.. code-block:: python

   import h5py
   import pandas as pd
   import numpy as np
   
   # Load per-region embeddings
   with h5py.File('output/cistrome_emb_on_region.hdf5', 'r') as f:
       ctcf = f['emb/CTCF:K562'][:]
       h3k27ac = f['emb/H3K27ac:K562'][:]
       regions = f['region'][:]
   
   # Find regions with high CTCF signal
   ctcf_signal = ctcf.mean(axis=1)
   high_ctcf = np.percentile(ctcf_signal, 90)
   high_ctcf_regions = regions[ctcf_signal > high_ctcf]
   
   # Save high CTCF regions
   df = pd.DataFrame(high_ctcf_regions[:, :3], 
                     columns=['chr', 'start', 'end'])
   df.to_csv('high_ctcf_regions.bed', sep='\t', 
             header=False, index=False)

Tips
====

1. **Cistrome identifiers**: 
   
   * Check available cistromes in ChromBERT database
   * Factor:Cell format is most flexible
   * Use exact cell type names (case-sensitive)
   * GSM/ENCODE IDs must exist in database

2. **Choosing cistromes**: 
   
   * Select biologically relevant cistromes for your question
   * Include both TFs and histone marks for comprehensive analysis
   * Consider cell-type specificity
   * Mix experimental (GSM/ENCODE) and predicted (factor:cell) data

3. **Performance**: 
   
   * Each cistrome adds to processing time and memory
   * Process in batches if analyzing many cistromes
   * HDF5 format handles large datasets efficiently

Troubleshooting
===============

**ValueError: No requested cistromes matched ChromBERT meta**

* Check cistrome identifier spelling
* Verify factor:cell pairs exist in database
* Try alternative cell type names
* Check that genome matches (hg38 vs mm10)

**Some cistromes missing in output**

* Some identifiers may not be in the database
* Check output messages for warnings
* Verify factor and cell type names
* Try factor:cell format instead of IDs

**HDF5 file very large**

* This is normal for many cistromes/regions
* HDF5 is compressed and efficient
* Access specific cistromes as needed
* Consider processing fewer cistromes at once

See Also
========

* :doc:`embed-regulator` - Extract regulator embeddings
* :doc:`embed-cell-cistrome` - Cell-specific cistrome embeddings
* :doc:`impute-cistrome` - Impute missing cistrome data
* :doc:`overview` - Common parameters

