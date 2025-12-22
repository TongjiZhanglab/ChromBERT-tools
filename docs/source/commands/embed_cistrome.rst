===============
embed_cistrome
===============

Extract embeddings for cistromes.

Overview
========

The ``embed_cistrome`` command extracts general (pre-trained) embeddings for user-specified cistrome datasets (e.g., ChIP-seq, ATAC-seq) across genomic regions using the pre-trained ChromBERT model. Cistromes can be specified using GSM IDs, ENCODE accessions, or factor:cell pairs.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed_cistrome \
     --cistrome "cistrome1;cistrome2" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

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

``--cistrome``
   Cistrome identifiers separated by semicolons (e.g., ``CTCF:K562;H3K27ac:K562;GSM1208591``). Supported formats include GSM IDs, ENCODE accessions, and factor:cell pairs. Identifiers will be converted to lowercase for matching.

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

``cistrome_emb_on_region.hdf5``
   HDF5 file containing cistrome embeddings for each region.

   .. code-block:: python

      import h5py

      # Example: if you specify --cistrome "CTCF:K562;H3K27ac:K562;GSM1208591"
      with h5py.File("cistrome_emb_on_region.hdf5", "r") as f:
          emb1 = f["/emb/ctcf:k562"][:]
          emb2 = f["/emb/h3k27ac:k562"][:]
          emb3 = f["/emb/gsm1208591"][:]

``mean_cistrome_emb.pkl``
   Python dictionary containing mean embeddings for each cistrome.

   .. code-block:: python

      import pickle

      # Example: if you specify --cistrome "CTCF:K562;H3K27ac:K562;GSM1208591"
      with open("mean_cistrome_emb.pkl", "rb") as f:
          mean_embs = pickle.load(f)

      # mean_embs = {"ctcf:k562": array([...]), "h3k27ac:k562": array([...]), ...}

``overlap_region.bed``
   Regions that overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).

``no_overlap_region.bed``
   Regions that do not overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).

Tips
====

1. **Cistrome not found**

   * Check whether the cistrome identifier is correct and in the expected format.
   * Prefer GSM IDs or ENCODE accessions (as used in ChromBERT).
   * The cistrome must be listed in ``<chrombert-cache-dir>/config/*_meta.tsv``.
