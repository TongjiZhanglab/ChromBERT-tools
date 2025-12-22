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
     --cistrome "cistrome1;cistrome2" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:
.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_cistrome \
     --cistrome "cistrome1;cistrome2" \
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
   Cistrome identifiers: GSM/ENCODE IDs or factor:cell pairs, use ; to separate multiple cistromes. It will be converted to lowercase for better matching, such as "CTCF:K562;H3K27ac:K562;GSM1208591"


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

``cistrome_emb_on_region.hdf5``
   HDF5 file containing cistrome embeddings for each region
   
   .. code-block:: python
   
      import h5py
      
      with h5py.File('cistrome_emb_on_region.hdf5', 'r') as f:
          # if you specify cistrome: "CTCF:K562;H3K27ac:K562;GSM1208591", you can get the embeddings by:
          emb1 = f['/emb/ctcf:k562'][:]
          emb2 = f['/emb/h3k27ac:k562'][:]
          emb3 = f['/emb/gsm1208591'][:]

``mean_cistrome_emb.pkl``
   Python dictionary containing mean embeddings for each cistrome
   
   .. code-block:: python
   
      import pickle
      
      # if you specify cistrome: "CTCF:K562;H3K27ac:K562;GSM1208591", you can get the embeddings by:
      with open('mean_cistrome_emb.pkl', 'rb') as f:
          mean_embs = pickle.load(f)
      # mean_embs = {'ctcf:k562': array([...]), 'h3k27ac:k562': array([...]), 'gsm1208591': array([...]), ...}

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found in ChromBERT


Tips
====

1. **Cistrome not found**

   * Check if the cistrome identifier is correct
   * Replace cistrome with GSM IDs or ENCODE accessions (used in ChromBERT).
   * The cistrome must be listed in your ``chrombert-cache-dir/config/*_meta.tsv``
