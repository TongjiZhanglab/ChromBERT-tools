===========================
find_driver_in_dual_region
===========================

Identify driver factors distinguishing two sets of genomic regions.

Overview
========

The ``find_driver_in_dual_region`` command trains a classifier to distinguish two region sets and identifies which regulatory factors contribute most to the classification. This helps identify factors that define different types of regulatory elements.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools find_driver_in_dual_region \
     --function1-bed "regions1.bed;regions2.bed" \
     --function2-bed "regions1.bed" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Singularity Usage
=================

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools find_driver_in_dual_region \
     --function1-bed "regions1.bed;regions2.bed" \
     --function2-bed "regions1.bed" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Command Parameters
==================

Required Parameters
-------------------

``--function1-bed``
   Different genomic regions for function1. Use ';' to separate multiple BED files"

``--function2-bed``
   Different genomic regions for function2. Use ';' to separate multiple BED files"

Optional Parameters
-------------------
``--odir``
   Output directory (default: ``./output``)

``--function1-mode``
   Logic mode for function1 regions: ``and`` requires all conditions; ``or`` requires any condition. Default: ``and``

``--function2-mode``
   Logic mode for function2 regions: ``and`` requires all conditions; ``or`` requires any condition. Default: ``and``

``--dual-regulator``
   Dual-functional regulator(s) for extracting dual subnetworks. Use ';' to separate multiple regulators. Default: None. Specify regulators that can bind to both function 1 and function 2 regions.

``--ignore-regulator``
   Regulators to ignore. Use ';' to separate multiple regulators. Default: None. Specify regulators that you want to exclude (e.g., known distinguishing factors between function 1 and function 2). If not specified, all regulators will be analyzed.

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``, mouse only supports 1kb resolution

``--mode``
   Training mode: ``fast`` (default) or ``full``, if ``fast`` mode is used, only the sampled 20000 regions will be used for training 

``--ft-ckpt``
   Fine-tuned ChromBERT checkpoint file. Default: None. If you have a fine-tuned ChromBERT checkpoint file, you can specify it here.

``--batch-size``
   Batch size for processing (default: 4)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

Output Files
============

``dataset/``
   Training dataset

``train/try_XX_seed_YY/``
   Training outputs for attempt XX with seed YY
   * ``lightning_logs/*/checkpoints/*.ckpt``: Model checkpoint
   * ``eval_performance.json``: Evaluation metrics (pearsonr, spearmanr, etc.)

``emb/``
   Mean regulator embeddings
   * ``func1_regulator_embs_dict.pkl``: Regulator embeddings on function1 regions
   * ``func2_regulator_embs_dict.pkl``: Regulator embeddings on function2 regions
   
``results/``
   * ``factor_importance_rank.csv``: Driver factors for dual-functional regions (Columns: factors, similarity, rank)
   * ``dual_regulator_subnetwork.pdf``: Dual-functional regulator subnetworks, Generated only if ``--dual-regulator`` is specified
   * ``regulator_cosine_similarity_on_function1_region.csv``: regulator-regulator cosine similarity on function1 regions
   * ``regulator_cosine_similarity_on_function2_region.csv``: regulator-regulator cosine similarity on function2 regions

