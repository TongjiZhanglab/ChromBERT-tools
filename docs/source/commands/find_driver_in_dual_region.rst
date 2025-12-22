===========================
find_driver_in_dual_region
===========================

Identify driver factors distinguishing two sets of genomic regions.

Overview
========

The ``find_driver_in_dual_region`` command trains a classifier to distinguish two sets of genomic regions and identifies regulatory factors that contribute most to the classification.

A common use case is to compare:
1) regions that satisfy multiple conditions (e.g., ``region1 AND region2``), and
2) regions that satisfy only one condition (e.g., ``region1 ONLY``, i.e., ``region1 \ region2``).

Basic Usage
===========

Compare ``region1 AND region2`` vs ``region1 ONLY``:

.. code-block:: bash

   chrombert-tools find_driver_in_dual_region \
     --function1-bed "region1.bed;region2.bed" \
     --function2-bed "region1.bed" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

.. note::

   ``function1`` is the intersection of ``region1`` and ``region2`` (``region1 AND region2``).
   ``function2`` should represent ``region1 ONLY`` (``region1 \ region2``).


If you are using the ChromBERT Singularity image, you can run:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools find_driver_in_dual_region \
     --function1-bed "regions1.bed;regions2.bed" \
     --function2-bed "regions1.bed" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--function1-bed``
   BED file(s) for function 1 regions. Use semicolons to separate multiple BED files (e.g., ``a.bed;b.bed``).

``--function2-bed``
   BED file(s) for function 2 regions. Use semicolons to separate multiple BED files (e.g., ``c.bed;d.bed``).

Optional Parameters
-------------------

``--odir``
   Output directory (default: ``./output``).

``--function1-mode``
   Logic mode for combining function 1 region sets: ``and`` requires all inputs; ``or`` requires any input (default: ``and``).

``--function2-mode``
   Logic mode for combining function 2 region sets: ``and`` requires all inputs; ``or`` requires any input (default: ``and``).

``--dual-regulator``
   Dual-functional regulator(s) for extracting dual subnetworks. Use semicolons to separate multiple regulators (default: None). Specify regulators that can bind to both function 1 and function 2 regions.

``--ignore-regulator``
   Regulators to ignore. Use semicolons to separate multiple regulators (default: None). Use this to exclude regulators you do not want to analyze (e.g., known distinguishing factors).

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``.

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``. For ``mm10``, only ``1kb`` is supported.

``--mode``
   Training mode: ``fast`` (default) or ``full``. In ``fast`` mode, only 20,000 sampled regions are used for training.

``--ft-ckpt``
   Path to a fine-tuned ChromBERT checkpoint file (default: None). If provided, the tool will use this checkpoint instead of fine-tuning a new model.

``--batch-size``
   Batch size for processing (default: 4).

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``). If your cache is located elsewhere, set this path accordingly.

Output Files
============

``dataset/``
   Training dataset.

``train/try_XX_seed_YY/``
   Training outputs for attempt XX with seed YY.

   * ``lightning_logs/*/checkpoints/*.ckpt``: Model checkpoint
   * ``eval_performance.json``: Evaluation metrics (auprc, auroc, etc.)

``emb/``
   Mean regulator embeddings.

   * ``func1_regulator_embs_dict.pkl``: Regulator embeddings on function 1 regions
   * ``func2_regulator_embs_dict.pkl``: Regulator embeddings on function 2 regions

``results/``
   Result files.

   * ``factor_importance_rank.csv``: Ranked driver factors (columns: factors, similarity, rank)
   * ``dual_regulator_subnetwork.pdf``: Dual-functional regulator subnetwork (generated only if ``--dual-regulator`` is specified)
   * ``regulator_cosine_similarity_on_function1_region.csv``: Regulator–regulator cosine similarity on function 1 regions
   * ``regulator_cosine_similarity_on_function2_region.csv``: Regulator–regulator cosine similarity on function 2 regions
