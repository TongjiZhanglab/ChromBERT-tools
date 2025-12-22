================
impute_cistrome
================

Impute missing cistrome data using ChromBERT.

Overview
========

The ``impute_cistrome`` command uses ChromBERT's learned co-association patterns to impute cistrome signals (e.g., ChIP-seq) for factor–cell pairs where experimental data is unavailable.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools impute_cistrome \
     --cistrome "cistrome1;cistrome2;cistrome3" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools impute_cistrome \
     --cistrome "cistrome1;cistrome2;cistrome3" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--cistrome``
   Cistromes to impute in factor:cell format, separated by semicolons (e.g., ``CTCF:K562;H3K27ac:K562;BCL11A:GM12878``). Identifiers will be converted to lowercase for matching.

``--region``
   Regions to impute in BED/CSV/TSV format. For CSV/TSV, the file must contain columns: ``chrom``, ``start``, ``end``.

Optional Parameters
-------------------

``--help``
   Show help message.

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``.

``--resolution``
   Resolution: only ``1kb`` (default).

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

``results_prob_df.csv``
   Imputed peak probabilities.

``overlap_region.bed``
   Regions that overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).

``no_overlap_region.bed``
   Regions that do not overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).

Tips
====

ChromBERT cistrome imputation relies on two types of information: (1) regulator embeddings and (2) cell-type accessibility embeddings.

1. **Cell type not found**

   * The DNase-seq (or accessibility) data for the target cell type must be listed in ``<chrombert-cache-dir>/config/*_meta.tsv``.

2. **Regulator not found**

   * The regulator must be listed in ``<chrombert-cache-dir>/config/*_regulator_list.txt``.
