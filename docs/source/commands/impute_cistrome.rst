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
     --cistrome "cistrome1;cistrome2;cistrome3" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:
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
   Cistromes to impute in factor:celltype format, use ; to separate multiple cistromes. It will be converted to lowercase for better matching, such as "CTCF:K562;H3K27ac:K562;GSM1208591"

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
   Region batch size (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

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


Tips
===============
ChromBERT requires two types of embeddings for cistrome imputation:
1. **Cell-type embedding**: Wild-type DNase-seq embedding of the target cell type (stored in ChromBERT's HDF5 data) as the cell-type prompt
2. **Regulator embedding**: Embedding of the target regulator as the regulator prompt
* The cell type's DNase-seq data must be listed in ``chrombert-cache-dir/config/*_meta.tsv``
* The regulator must be listed in ``chrombert-cache-dir/config/*_regulator_list.txt``
