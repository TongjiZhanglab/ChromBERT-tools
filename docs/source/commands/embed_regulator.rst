================
embed_regulator
================

Extract embeddings for transcription factors and other regulators.

Overview
========

The ``embed_regulator`` command extracts general (pre-trained) embeddings for user-specified regulators across genomic regions using the pre-trained ChromBERT model.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools embed_regulator \
     --regulator "regulator1;regulator2;regulator3" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_regulator \
     --regulator "regulator1;regulator2;regulator3" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--regulator``
   Regulator names separated by semicolons (e.g., ``CTCF;MYC;TP53``). Names will be converted to lowercase for matching.

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

``regulator_emb_on_region.hdf5``
   HDF5 file containing regulator embeddings for each region.

   .. code-block:: python

      import h5py

      # Example: if you specify --regulator "CTCF;MYC;TP53"
      with h5py.File("regulator_emb_on_region.hdf5", "r") as f:
          emb1 = f["/emb/ctcf"][:]
          emb2 = f["/emb/myc"][:]
          emb3 = f["/emb/tp53"][:]

``mean_regulator_emb.pkl``
   Python dictionary containing mean embeddings for each regulator.

   .. code-block:: python

      import pickle

      with open("mean_regulator_emb.pkl", "rb") as f:
          mean_embeddings = pickle.load(f)

      # mean_embeddings = {"ctcf": array([...]), "myc": array([...]), ...}

``overlap_region.bed``
   Regions that overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).

``no_overlap_region.bed``
   Regions that do not overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).

Tips
====

1. **Regulator not found**

   * Check whether the regulator identifier is correct.
   * The regulator must be listed in ``<chrombert-cache-dir>/config/*_regulator_list.txt``.
