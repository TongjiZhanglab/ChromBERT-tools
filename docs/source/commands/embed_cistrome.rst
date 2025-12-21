===============
embed_cistrome
===============

Extract embeddings for cistromes.

Overview
========

The ``embed_cistrome`` command extracts general embeddings for specified cistrome datasets (ChIP-seq, ATAC-seq, etc.) across genomic regions using the pre-trained ChromBERT model. Cistromes can be specified using GSM IDs, ENCODE IDs, or factor:celltype pairs.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed_cistrome \
     --cistrome "CTCF:K562;H3K27ac:K562" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--help``
   Show help message

``--cistrome``
   Cistrome identifiers: GSM/ENCODE IDs or factor:cell pairs, use ; to separate multiple cistromes. It will be converted to lowercase for better matching


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
   Batch size for training (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

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
          ctcf_emb = f['emb/ctcf:k562'][:]  # Shape: (n_regions, 768)
          regions = f['region'][:]

``mean_cistrome_emb.pkl``
   Python dictionary containing mean embeddings for each cistrome
   
   .. code-block:: python
   
      import pickle
      
      with open('mean_cistrome_emb.pkl', 'rb') as f:
          mean_embs = pickle.load(f)
      # mean_embs = {'ctcf:k562': array([...]), ...}

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found in ChromBERT

Examples
========

Using Factor:Cell Pairs
------------------------

.. code-block:: bash

   chrombert-tools embed_cistrome \
     --cistrome "CTCF:K562;H3K27ac:K562;ATAC-seq:K562" \
     --region promoters.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir k562_cistromes

Using GSM IDs
-------------

.. code-block:: bash

   chrombert-tools embed_cistrome \
     --cistrome "GSM1208591;GSM1234567;GSM9876543" \
     --region enhancers.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir gsm_cistromes

Using ENCODE IDs
----------------

.. code-block:: bash

   chrombert-tools embed_cistrome \
     --cistrome "ENCSR440VKE_2;ENCSR123ABC;ENCSR456DEF" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir encode_cistromes

Mixed Identifiers
-----------------

.. code-block:: bash

   chrombert-tools embed_cistrome \
     --cistrome "CTCF:K562;GSM1208591;ENCSR440VKE_2;H3K27ac:HepG2" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir mixed_cistromes

Troubleshooting
===============

**Cistrome not found**

   * Check if the cistrome identifier is correct
   * Check if the cistrome identifier is in the correct format
   * You can find all cistrome in your chrombert-cache-dir/config/*_meta.tsv
   * Replace cistrome with GSM IDs or ENCODE accessions (used in ChromBERT).

