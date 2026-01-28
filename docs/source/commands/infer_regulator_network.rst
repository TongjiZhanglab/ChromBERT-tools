=======================
infer_regulator_network
=======================

Infer a general regulator-regulator network on specified regions.

Overview
========

The ``infer_regulator_network`` command uses the pre-trained ChromBERT model to infer regulator–regulator co-association relationships on user-specified genomic regions.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools infer_regulator_network \
     --region regions.bed \
     --regulator "regulator1;regulator2;regulator3" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools infer_regulator_network \
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
   Regions of interest in BED/CSV/TSV format. For CSV/TSV, the file must contain columns: ``chrom``, ``start``, ``end``.

Optional Parameters
-------------------

``--help``
   Show help message.

``--regulator``
   Regulators for subnetwork plotting, separated by semicolons (e.g., ``CTCF;MYC;TP53``). Names will be converted to lowercase for matching. If not provided, the full network will still be computed, but subnetworks will not be plotted.

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``.

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``. For ``mm10``, only ``1kb`` is supported.

``--odir``
   Output directory (default: ``./output``).

``--batch-size``
   Region batch size (default: 4).

``--num-workers``
   Number of dataloader workers (default: 8).

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``). If your cache is located elsewhere, set this path accordingly.

``--quantile``
   Quantile threshold for cosine-similarity edges (default: 0.99).

``--k-hop``
   k-hop radius for subnetwork plotting (default: 1).

Output Files
============

``regulator_cosine_similarity.tsv``
   Regulator–regulator cosine similarity matrix. Higher values indicate greater similarity between two regulators.

``total_graph_edge_threshold*_quantile*.tsv``
   Regulator–regulator edges on the specified regions, filtered by the chosen threshold/quantile.

``subnetwork_regulator_k*.pdf``
   Regulator subnetworks on the specified regions. If you provide ``--regulator "A;B;C"``, a subnetwork will be generated for each requested regulator.

``overlap_region.bed``
   Regions that overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).

``no_overlap_region.bed``
   Regions that do not overlap with ChromBERT regions (see ``<chrombert-cache-dir>/config/*region.bed``).

Tips
====

1. **Regulator not found**

   * Check whether the regulator identifier is correct.
   * The regulator must be listed in ``<chrombert-cache-dir>/config/*_regulator_list.txt``.
