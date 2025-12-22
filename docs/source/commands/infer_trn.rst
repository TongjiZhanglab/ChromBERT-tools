=========
infer_trn
=========

Infer general transcriptional regulatory network (TRN) on specific rgeions.

Overview
========

The ``infer_trn`` command uses pre-trained ChromBERT's learned regulatory patterns to infer regulator-regulator relationships on specific regions. 

Basic Usage
===========

.. code-block:: bash

   chrombert-tools infer_trn \
     --region regions.bed \
     --regulator "regulator1;regulator2;regulator3" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are use the ChromBERT Singularity image, you can run the command as follows:
.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools infer_trn \
     --region regions.bed \
     --regulator "regulator1;regulator2;regulator3" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--region``
   regions of interest: BED or CSV or TSV file (CSV/TSV need with columns: chrom, start, end)

Optional Parameters
-------------------

``--help``
   Show help message and exit

``--regulator``
   You want to infer the TRN for these regulators, regulator names separated by semicolons, it will be converted to lowercase for better matching, such as "CTCF;MYC;TP53"

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``, mouse only supports 1kb resolution

``--odir``
   Output directory (default: ``./output``)

``--batch-size``
   Region batch size (default: 4)

``--num-workers``
   Number of dataloader workers (default: 8)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

``--quantile``
   Quantile threshold for cosine similarity edges (default: 0.99)

``--k-hot``
   k-hop radius for subnetwork plotting (default: 1)

Output Files
============

``regulator_cosine_similarity.tsv``
   regulator-regulator cosine similarity. Higher values indicate greater similarity between the two regulators.

``total_graph_edgh_threshold*_quantile*.tsv ``
   regulator-regulator edges on this regions (filtered by the specified threshold/quantile)

``subnetwork_regulator_k*.pdf``
   regulator subnetwork on this regions, if you specify ``--regulator "regulator1;regulator2;regulator3"``, you will get the subnetwork for each regulator

``overlap_region.bed``
   Regions successfully processed

``no_overlap_region.bed``
   Regions not found in ChromBERT


Tips
===============

**Regulator not found**

   * Check if the regulator is correct
   * The regulator must be listed in ``chrombert-cache-dir/config/*_regulator_list.txt``
