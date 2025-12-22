==========================
find_driver_in_transition
==========================

Identify driver factors in cell state transitions.

Overview
========

The ``find_driver_in_transition`` command identifies key transcription factors drive changes in gene expression and chromatin accessibility during cell state transitions such as differentiation or reprogramming.

Basic Usage
===========
If you provide the expression data and chromatin accessibility data for this tranistion: 

.. code-block:: bash

   chrombert-tools find_driver_in_transition \
     --exp-tpm1 state1_expression.csv \
     --exp-tpm2 state2_expression.csv \
     --acc-peak1 state1_peaks.bed \
     --acc-peak2 state2_peaks.bed \
     --acc-signal1 state1_signal.bigwig \
     --acc-signal2 state2_signal.bigwig \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output


If you provide only the chromatin accessibility data for this tranistion:

.. code-block:: bash

   chrombert-tools find_driver_in_transition \
     --acc-peak1 state1_peaks.bed \
     --acc-peak2 state2_peaks.bed \
     --acc-signal1 state1_signal.bigwig \
     --acc-signal2 state2_signal.bigwig \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output

If you provide only the expression data for this tranistion: 

.. code-block:: bash

   chrombert-tools find_driver_in_transition \
     --exp-tpm1 state1_expression.csv \
     --exp-tpm2 state2_expression.csv \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output

If you are using the ChromBERT Singularity image, you can run the command as follows:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools find_driver_in_transition \
     --exp-tpm1 state1_expression.csv \
     --exp-tpm2 state2_expression.csv \
     --acc-peak1 state1_peaks.bed \
     --acc-peak2 state2_peaks.bed \
     --acc-signal1 state1_signal.bigwig \
     --acc-signal2 state2_signal.bigwig \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output

   singularity exec --nv /path/to/chrombert.sif chrombert-tools find_driver_in_transition \
     --acc-peak1 state1_peaks.bed \
     --acc-peak2 state2_peaks.bed \
     --acc-signal1 state1_signal.bigwig \
     --acc-signal2 state2_signal.bigwig \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output

   singularity exec --nv /path/to/chrombert.sif chrombert-tools find_driver_in_transition \
     --exp-tpm1 state1_expression.csv \
     --exp-tpm2 state2_expression.csv \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output


Parameters
==========

Required Parameters
-------------------

``--exp-tpm1``, ``--exp-tpm2``
   Expression data (CSV) for two cell states, must contain columns: gene_id, tpm

``--acc-peak1``, ``--acc-peak2``
   Accessibility peaks (BED) for two states

``--acc-signal1``, ``--acc-signal2``
   Accessibility signal (BigWig) for two states


Optional Parameters
-------------------
``--direction``
   Direction of transition
   
   * ``"2-1"``: From state 1 to state 2
   * ``"1-2"``: From state 2 to state 1

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--ft-ckpt-exp``
   fine-tuned expression model checkpoint

``--ft-ckpt-acc``
   fine-tuned accessibility model checkpoint

``--mode``
   Training mode: ``fast`` (default) or ``full``, if ``fast`` mode is used, only the sampled 20000 regions will be used for training 

``--odir``
   Output directory (default: ``./output``)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``), If your cache file in different directory, you can specify the path here

Output Files
============

Expression Analysis
-------------------

``exp/results/``
   * ``factor_importance_rank.csv``: Driver factors for expression changes (Columns: factors, similarity, rank)

``exp/train/try_XX_seed_YY/``
   Training outputs for attempt XX with seed YY
   * ``lightning_logs/*/checkpoints/*.ckpt``: Model checkpoint
   * ``eval_performance.json``: Evaluation metrics (pearsonr, spearmanr, etc.)

``exp/dataset/``
   Training dataset
   * ``up.csv``: genes with more expression in state 2 if direction is "2-1", or genes with more expression in state 1 if direction is "1-2"
   * ``nochange.csv``: genes with no expression change

``exp/emb/``
   Mean regulator embeddings
   * ``up_regulator_embs_dict.pkl``: Regulator embeddings on up genes
   * ``nochange_regulator_embs_dict.pkl``: Regulator embeddings on nochange genes

Accessibility Analysis
----------------------

``acc/results/``
   * ``factor_importance_rank.csv``: Driver factors for accessibility changes (Columns: factors, similarity, rank)

``acc/train/``
   Training outputs for attempt XX with seed YY
   * ``lightning_logs/*/checkpoints/*.ckpt``: Model checkpoint
   * ``eval_performance.json``: Evaluation metrics (pearsonr, spearmanr, etc.)

``acc/dataset/``
   Training dataset
   * ``up.csv``: regions with more accessibility in state 2 if direction is "2-1", or regions with more accessibility in state 1 if direction is "1-2"
   * ``nochange.csv``: regions with no accessibility change

``acc/emb/``
   Mean regulator embeddings
   * ``up_regulator_embs_dict.pkl``: Regulator embeddings on up regions
   * ``nochange_regulator_embs_dict.pkl``: Regulator embeddings on nochange regions

Tips
====

1. **Data quality**: 
   
   * Use high-quality expression and accessibility data

2. **Training mode**: 
   
   * Start with ``--mode fast`` for exploration
   * Use ``--mode full`` for final results
   * Fast mode is usually sufficient for most analyses

3. **Checkpoint reuse**: 
   
   * Save checkpoints for reuse across analyses

4. **Memory errors during training**

   * Reduce ``--batch-size``

5. **Interpretation**: 
   
   * Lower similarity = more important driver
   * Check both expression and accessibility drivers
   * Shared drivers likely play key roles
   * Validate with literature and experiments

