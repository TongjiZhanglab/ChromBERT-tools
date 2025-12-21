================
impute_cistrome
================

Impute missing cistrome data using ChromBERT.

Overview
========

The ``impute_cistrome`` command uses ChromBERT's learned co-association patterns to impute ChIP-seq for cell types or factors where experimental data is unavailable.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools impute_cistrome \
     --cistrome "BCL11A:GM12878;BRD4:MCF7;CTCF:HepG2" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--cistrome``
   Cistromes to impute in factor:celltype format, use ; to separate multiple cistromes. It will be converted to lowercase for better matching

``--region``
   Regions to impute (BED or CSV format)

Optional Parameters
-------------------

``--help``
   Show help message

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``

``--resolution``
   Resolution: only ``1kb`` (default)

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Batch size (default: 4)

``--num-workers``
   Dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

Output Files
============

``results_prob_df.csv``
   imputed peak probability
   

``overlap_region.bed``
   Regions overlap with chrombert regions (your chrombert-cache-dir/config/*region.bed)

``no_overlap_region.bed``
   Regions not overlap with chrombert regions (your chrombert-cache-dir/config/*region.bed)


Troubleshooting
===============

**Regulator not found**
   * Check if the regulator is correct
   * You can find all regulator in your chrombert-cache-dir/anno/*_regulator_list.txt

**Celltype not found**
   * Check if the celltype is correct
   * You can find all celltype with wide type dnase-seq in your chrombert-cache-dir/config/*_meta.tsv
   * Replace celltype names with GSM IDs or ENCODE accessions (used DNase data in ChromBERT).

