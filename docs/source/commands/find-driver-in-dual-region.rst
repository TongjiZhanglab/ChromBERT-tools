===========================
find-driver-in-dual-region
===========================

Identify driver factors distinguishing two sets of genomic regions.

Overview
========

The ``find-driver-in-dual-region`` command trains a classifier to distinguish two region sets and identifies which regulatory factors contribute most to the classification. This helps identify factors that define different types of regulatory elements.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools find-driver-in-dual-region \
     --region1 active_regions.bed \
     --region2 inactive_regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--region1``
   First set of regions (BED format)

``--region2``
   Second set of regions (BED format)

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Optional Parameters
-------------------

``--label1``
   Label for first set
   
   Default: ``"positive"``

``--label2``
   Label for second set
   
   Default: ``"negative"``

``--mode``
   Training mode: ``fast`` (default) or ``full``

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

Output Files
============

``driver_factors.csv``
   Ranked list of driver factors
   
   Columns: factors, importance_score, rank

``train/``
   Fine-tuned classifier model

``classification_metrics.json``
   Model performance metrics
   
   Includes:
   
   * Accuracy, precision, recall, F1 score
   * AUC-ROC, AUC-PR
   * Confusion matrix

``dataset/``
   Training dataset

Examples
========

Active vs. Inactive Enhancers
------------------------------

.. code-block:: bash

   chrombert-tools find-driver-in-dual-region \
     --region1 active_enhancers.bed \
     --region2 inactive_enhancers.bed \
     --label1 "active" \
     --label2 "inactive" \
     --genome hg38 \
     --resolution 1kb \
     --mode fast \
     --odir enhancer_drivers

Promoter vs. Enhancer
---------------------

.. code-block:: bash

   chrombert-tools find-driver-in-dual-region \
     --region1 promoters.bed \
     --region2 enhancers.bed \
     --label1 "promoter" \
     --label2 "enhancer" \
     --genome hg38 \
     --resolution 1kb \
     --odir promoter_vs_enhancer

Disease vs. Healthy Regions
----------------------------

.. code-block:: bash

   chrombert-tools find-driver-in-dual-region \
     --region1 disease_regions.bed \
     --region2 healthy_regions.bed \
     --label1 "disease" \
     --label2 "healthy" \
     --genome hg38 \
     --resolution 1kb \
     --mode full \
     --odir disease_drivers

Interpreting Results
====================

Analyze Driver Factors
-----------------------

.. code-block:: python

   import pandas as pd
   import json
   
   # Load driver factors
   drivers = pd.read_csv('output/driver_factors.csv')
   
   # Load classification metrics
   with open('output/classification_metrics.json', 'r') as f:
       metrics = json.load(f)
   
   print(f"Classification accuracy: {metrics['accuracy']:.3f}")
   print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
   
   print("\nTop 10 driver factors:")
   print(drivers.head(10))

Visualize Drivers
-----------------

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot top 15 drivers
   top_drivers = drivers.head(15)
   
   plt.figure(figsize=(10, 6))
   plt.barh(range(15), top_drivers['importance_score'])
   plt.yticks(range(15), top_drivers['factors'])
   plt.xlabel('Importance Score')
   plt.title('Top 15 Driver Factors')
   plt.gca().invert_yaxis()
   plt.tight_layout()
   plt.savefig('driver_factors.png')

Tips
====

1. **Region selection**: 
   
   * Use comparable region sets (similar size, number)
   * Avoid overlapping regions between sets
   * Balance region counts if possible

2. **Interpretation**: 
   
   * Higher importance = stronger discriminatory power
   * Top drivers define region identity
   * Validate with known biology

3. **Model performance**: 
   
   * Check classification_metrics.json
   * AUC-ROC > 0.7 indicates good separation
   * Low performance may indicate similar regions

Use Cases
=========

* Distinguish active vs. inactive regulatory elements
* Compare element types (promoters, enhancers, silencers)
* Identify disease-associated regulatory changes
* Find conserved vs. divergent regulatory patterns
* Compare cell-type specific regulatory regions

Troubleshooting
===============

**Low classification accuracy**

* Regions may be too similar
* Try different resolution
* Ensure regions are properly defined
* Check region quality and annotation

**No clear driver factors**

* Regions may not have strong regulatory differences
* Try larger/more specific region sets
* Check that regions are biologically distinct

See Also
========

* :doc:`find-driver-in-transition` - Driver factors in transitions
* :doc:`embed-region` - Extract region embeddings
* :doc:`overview` - Common parameters

