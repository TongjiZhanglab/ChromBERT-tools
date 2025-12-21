=========
infer-trn
=========

Infer transcriptional regulatory network (TRN) from expression data.

Overview
========

The ``infer-trn`` command uses ChromBERT's learned regulatory patterns and gene expression data to infer gene-regulator relationships, producing a transcriptional regulatory network.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools infer-trn \
     --tpm expression.csv \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required Parameters
-------------------

``--tpm``
   Gene expression file in CSV format
   
   * Rows: genes
   * Columns: samples
   * Values: TPM/FPKM/normalized counts

``--genome``
   Genome assembly (``hg38`` or ``mm10``)

Optional Parameters
-------------------

``--top-k``
   Number of top regulators to identify per gene
   
   Default: 10

``--resolution``
   Resolution: ``200bp``, ``1kb`` (default), ``2kb``, or ``4kb``

``--odir``
   Output directory (default: ``./output``)

``--chrombert-cache-dir``
   ChromBERT cache directory (default: ``~/.cache/chrombert/data``)

Output Files
============

``trn_network.tsv``
   Regulatory network in tab-separated format
   
   Columns:
   
   * ``gene``: Target gene
   * ``regulator``: Regulator name
   * ``score``: Regulatory score
   * ``rank``: Rank for this gene (1 = top regulator)

``trn_network.pkl``
   Network in pickle format for Python

``network_statistics.json``
   Network statistics and metrics
   
   Contains:
   
   * Number of genes
   * Number of regulators
   * Number of edges
   * Network density
   * Top hub regulators

Examples
========

Basic Network Inference
-----------------------

.. code-block:: bash

   chrombert-tools infer-trn \
     --tpm K562_expression.csv \
     --top-k 20 \
     --genome hg38 \
     --resolution 1kb \
     --odir K562_network

Comprehensive Network
---------------------

.. code-block:: bash

   chrombert-tools infer-trn \
     --tpm expression.csv \
     --top-k 50 \
     --genome hg38 \
     --resolution 1kb \
     --odir comprehensive_network

Using Network Output
====================

Load and Analyze
----------------

.. code-block:: python

   import pandas as pd
   import json
   
   # Load network
   network = pd.read_csv('output/trn_network.tsv', sep='\t')
   
   # Load statistics
   with open('output/network_statistics.json', 'r') as f:
       stats = json.load(f)
   
   print(f"Network contains:")
   print(f"  {stats['n_genes']} genes")
   print(f"  {stats['n_regulators']} regulators")
   print(f"  {stats['n_edges']} edges")
   
   # Get regulators for a specific gene
   gene_regs = network[network['gene'] == 'BRCA1']
   print(f"\nTop regulators of BRCA1:")
   print(gene_regs.head(10))

Network Visualization
---------------------

.. code-block:: python

   import networkx as nx
   import matplotlib.pyplot as plt
   
   # Create graph
   G = nx.from_pandas_edgelist(
       network,
       source='regulator',
       target='gene',
       edge_attr='score',
       create_using=nx.DiGraph()
   )
   
   # Get top hub regulators
   in_degree = dict(G.in_degree())
   top_hubs = sorted(in_degree.items(), 
                     key=lambda x: x[1], 
                     reverse=True)[:10]
   
   print("Top hub regulators:")
   for node, degree in top_hubs:
       print(f"  {node}: regulates {degree} genes")

Tips
====

1. **Expression data**: 
   
   * Use normalized data (TPM, FPKM)
   * Remove lowly expressed genes
   * Filter outlier samples
   * Ensure proper gene naming

2. **Top-k selection**: 
   
   * 10-20: Focus on key regulators
   * 20-50: Comprehensive network
   * 5-10: High-confidence edges only

3. **Interpretation**: 
   
   * Lower rank = stronger regulatory relationship
   * Validate key edges with literature
   * Check enrichment of known TF targets

Troubleshooting
===============

**Empty network output**

* Check expression file format
* Verify gene names match reference genome
* Ensure sufficient gene expression

**Network too large**

* Reduce ``--top-k``
* Filter low-confidence edges
* Focus on specific gene sets

**No regulators for some genes**

* Some genes may not be in ChromBERT
* Check gene naming conventions
* Try alternative gene names

See Also
========

* :doc:`infer-cell-trn` - Cell-specific TRN inference
* :doc:`embed-gene` - Extract gene embeddings
* :doc:`overview` - Common parameters

