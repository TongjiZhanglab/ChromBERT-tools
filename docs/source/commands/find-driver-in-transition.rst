==========================
find-driver-in-transition
==========================

Identify driver factors in cell state transitions.

Overview
========

The ``find-driver-in-transition`` command identifies key transcription factors and epigenetic marks that drive changes in gene expression and chromatin accessibility during cell state transitions such as differentiation or reprogramming.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools find-driver-in-transition \
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

Parameters
==========

Required Parameters
-------------------

``--exp-tpm1``, ``--exp-tpm2``
   Expression data (CSV) for two cell states

``--acc-peak1``, ``--acc-peak2``
   Accessibility peaks (BED) for two states

``--acc-signal1``, ``--acc-signal2``
   Accessibility signal (BigWig) for two states

``--direction``
   Direction of transition
   
   * ``"2-1"``: From state 2 to state 1
   * ``"1-2"``: From state 1 to state 2

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Optional Parameters
-------------------

``--ft-ckpt-exp``
   Pre-trained expression model checkpoint

``--ft-ckpt-acc``
   Pre-trained accessibility model checkpoint

``--mode``
   Training mode: ``fast`` (default) or ``full``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

Output Files
============

Expression Analysis
-------------------

``exp/driver_factors_exp.csv``
   Driver factors for expression changes
   
   Columns: factors, similarity, rank

``exp/train/``
   Fine-tuned model for expression prediction

``exp/dataset/``
   Training dataset

Accessibility Analysis
----------------------

``acc/driver_factors_acc.csv``
   Driver factors for accessibility changes

``acc/train/``
   Fine-tuned model for accessibility prediction

``acc/dataset/``
   Training dataset

Examples
========

Differentiation Analysis
------------------------

Fibroblast to myoblast:

.. code-block:: bash

   chrombert-tools find-driver-in-transition \
     --exp-tpm1 fibroblast_expression.csv \
     --exp-tpm2 myoblast_expression.csv \
     --acc-peak1 fibroblast_peaks.bed \
     --acc-peak2 myoblast_peaks.bed \
     --acc-signal1 fibroblast_ATAC.bigwig \
     --acc-signal2 myoblast_ATAC.bigwig \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --mode fast \
     --odir fib_to_myo

Reprogramming Analysis
----------------------

.. code-block:: bash

   chrombert-tools find-driver-in-transition \
     --exp-tpm1 fibroblast_expression.csv \
     --exp-tpm2 ipsc_expression.csv \
     --acc-peak1 fibroblast_peaks.bed \
     --acc-peak2 ipsc_peaks.bed \
     --acc-signal1 fibroblast_ATAC.bigwig \
     --acc-signal2 ipsc_ATAC.bigwig \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --mode full \
     --odir reprogramming

Using Pre-trained Models
-------------------------

.. code-block:: bash

   chrombert-tools find-driver-in-transition \
     --exp-tpm1 state1_exp.csv \
     --exp-tpm2 state2_exp.csv \
     --acc-peak1 state1_peaks.bed \
     --acc-peak2 state2_peaks.bed \
     --acc-signal1 state1_signal.bigwig \
     --acc-signal2 state2_signal.bigwig \
     --ft-ckpt-exp /path/to/exp_ckpt.ckpt \
     --ft-ckpt-acc /path/to/acc_ckpt.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --direction "2-1" \
     --odir output

Interpreting Results
====================

Analyze Driver Factors
-----------------------

.. code-block:: python

   import pandas as pd
   
   # Load driver factors
   exp_drivers = pd.read_csv('output/exp/driver_factors_exp.csv')
   acc_drivers = pd.read_csv('output/acc/driver_factors_acc.csv')
   
   # Top 10 expression drivers
   print("Top 10 expression drivers:")
   print(exp_drivers.head(10))
   
   # Top 10 accessibility drivers
   print("\nTop 10 accessibility drivers:")
   print(acc_drivers.head(10))
   
   # Shared drivers (appear in both analyses)
   exp_set = set(exp_drivers.head(25)['factors'])
   acc_set = set(acc_drivers.head(25)['factors'])
   shared = exp_set & acc_set
   
   print(f"\nShared top drivers: {shared}")

Visualize Results
-----------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Plot top drivers
   top_n = 20
   top_drivers = exp_drivers.head(top_n)
   
   plt.figure(figsize=(10, 8))
   plt.barh(range(top_n), top_drivers['similarity'])
   plt.yticks(range(top_n), top_drivers['factors'])
   plt.xlabel('Similarity Score')
   plt.title('Top 20 Expression Driver Factors')
   plt.gca().invert_yaxis()
   plt.tight_layout()
   plt.savefig('top_drivers.png')

Tips
====

1. **Data quality**: 
   
   * Use high-quality expression and accessibility data
   * Ensure proper normalization
   * Match sequencing depth between states

2. **Direction parameter**: 
   
   * ``"2-1"``: Factors driving from state 2 to state 1
   * ``"1-2"``: Factors driving from state 1 to state 2
   * Choose based on biological question

3. **Mode selection**: 
   
   * ``fast``: Quick exploration (20k regions)
   * ``full``: Publication quality (all regions)

4. **Interpretation**: 
   
   * Lower similarity = more important driver
   * Check both expression and accessibility drivers
   * Shared drivers likely play key roles
   * Validate with literature and experiments

Use Cases
=========

* Identify master regulators of differentiation
* Find factors for cellular reprogramming
* Track regulatory changes in development
* Identify drivers of disease progression

See Also
========

* :doc:`find-driver-in-dual-region` - Driver factors for region differences
* :doc:`infer-cell-trn` - Cell-specific TRN inference
* :doc:`overview` - Common parameters

