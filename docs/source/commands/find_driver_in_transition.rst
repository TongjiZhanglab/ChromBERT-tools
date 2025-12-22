==========================
find_driver_in_transition
==========================

Identify driver factors in cell state transitions.

Overview
========

The ``find_driver_in_transition`` command identifies key transcription factors that drive changes in gene expression and/or chromatin accessibility during cell state transitions (e.g., differentiation or reprogramming).

Basic Usage
===========

Provide both expression and chromatin accessibility data for the transition:

.. code-block:: bash

   chrombert-tools find_driver_in_transition \
     --exp-tpm1 /path/to/state1_expression.csv \
     --exp-tpm2 /path/to/state2_expression.csv \
     --acc-peak1 /path/to/state1_peaks.bed \
     --acc-peak2 /path/to/state2_peaks.bed \
     --acc-signal1 /path/to/state1_signal.bigwig \
     --acc-signal2 /path/to/state2_signal.bigwig \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output

Provide only chromatin accessibility data for the transition:

.. code-block:: bash

   chrombert-tools find_driver_in_transition \
     --acc-peak1 /path/to/state1_peaks.bed \
     --acc-peak2 /path/to/state2_peaks.bed \
     --acc-signal1 /path/to/state1_signal.bigwig \
     --acc-signal2 /path/to/state2_signal.bigwig \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output

Provide only expression data for the transition:

.. code-block:: bash

   chrombert-tools find_driver_in_transition \
     --exp-tpm1 /path/to/state1_expression.csv \
     --exp-tpm2 /path/to/state2_expression.csv \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output

If you are using the ChromBERT Singularity image, you can run the command as follows:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools find_driver_in_transition \
     --exp-tpm1 /path/to/state1_expression.csv \
     --exp-tpm2 /path/to/state2_expression.csv \
     --acc-peak1 /path/to/state1_peaks.bed \
     --acc-peak2 /path/to/state2_peaks.bed \
     --acc-signal1 /path/to/state1_signal.bigwig \
     --acc-signal2 /path/to/state2_signal.bigwig \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output

Parameters
==========

Input Data
----------

``--exp-tpm1``, ``--exp-tpm2``
   Expression data (CSV) for two cell states. Each file must contain columns: ``gene_id`` and ``tpm``.

``--acc-peak1``, ``--acc-peak2``
   Accessibility peaks (BED or narrowPeak) for two states.

``--acc-signal1``, ``--acc-signal2``
   Accessibility signal tracks (BigWig: ``.bw``/``.bigWig``) for two states.

.. note::

   You can run this command with:
   (1) expression only, (2) accessibility only, or (3) both expression and accessibility.
   Provide the corresponding input files for the analyses you want to perform.

Optional Parameters
-------------------

``--direction``
   Direction of transition:

   * ``"2-1"``: from state 1 to state 2
   * ``"1-2"``: from state 2 to state 1

``--genome``
   Genome assembly: ``hg38`` (default) or ``mm10``.

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``. For ``mm10``, only ``1kb`` is supported.

``--ft-ckpt-exp``
   Fine-tuned expression-model checkpoint. If provided, the tool will skip fine-tuning and use this checkpoint directly to identify key transcription factors that drive gene-expression changes during this transition.

``--ft-ckpt-acc``
   Fine-tuned accessibility-model checkpoint. If provided, the tool will skip fine-tuning and use this checkpoint directly to identify key transcription factors that drive chromatin-accessibility changes during this transition.

``--mode``
   Training mode: ``fast`` (default) or ``full``. In ``fast`` mode, only 20,000 sampled regions are used for training.

``--odir``
   Output directory (default: ``./output``).

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``). If your cache is located elsewhere, set this path accordingly.

Output Files
============

Expression Analysis
-------------------

``exp/results/``
   * ``factor_importance_rank.csv``: Driver factors for expression changes (columns: factors, similarity, rank)

``exp/train/try_XX_seed_YY/``
   Training outputs for attempt XX with seed YY

   * ``lightning_logs/*/checkpoints/*.ckpt``: Model checkpoint
   * ``eval_performance.json``: Evaluation metrics (pearsonr, spearmanr, etc.)

``exp/dataset/``
   Training dataset

   * ``up.csv``: genes with higher expression in state 2 if direction is ``"2-1"``, or in state 1 if direction is ``"1-2"``
   * ``nochange.csv``: genes with no expression change

``exp/emb/``
   Mean regulator embeddings

   * ``up_regulator_embs_dict.pkl``: Regulator embeddings on up genes
   * ``nochange_regulator_embs_dict.pkl``: Regulator embeddings on no-change genes

Accessibility Analysis
----------------------

``acc/results/``
   * ``factor_importance_rank.csv``: Driver factors for accessibility changes (columns: factors, similarity, rank)

``acc/train/try_XX_seed_YY/``
   Training outputs for attempt XX with seed YY

   * ``lightning_logs/*/checkpoints/*.ckpt``: Model checkpoint
   * ``eval_performance.json``: Evaluation metrics (pearsonr, spearmanr, etc.)

``acc/dataset/``
   Training dataset

   * ``up.csv``: regions with higher accessibility in state 2 if direction is ``"2-1"``, or in state 1 if direction is ``"1-2"``
   * ``nochange.csv``: regions with no accessibility change

``acc/emb/``
   Mean regulator embeddings

   * ``up_regulator_embs_dict.pkl``: Regulator embeddings on up regions
   * ``nochange_regulator_embs_dict.pkl``: Regulator embeddings on no-change regions

Tips
====

1. **Data quality**

   * Use high-quality expression and accessibility data.

2. **Training mode**

   * Start with ``--mode fast`` for exploration.
   * Use ``--mode full`` for final results.
   * Fast mode is usually sufficient for most analyses.

3. **Checkpoint reuse**

   * Save checkpoints for reuse across analyses.

4. **Interpretation**

   * Lower similarity indicates a more important driver factor.
   * Check drivers from both expression and accessibility analyses.
   * Shared drivers across both analyses are more likely to play key roles.
   * Validate candidates with literature and follow-up experiments.
