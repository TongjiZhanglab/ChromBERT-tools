=====
Usage
=====

Welcome to the ChromBERT-tools usage guide. This section provides comprehensive documentation for all command-line tools.

Quick Start
===========

To see all available commands:

.. code-block:: bash

   chrombert-tools --help

To get help for a specific command:

.. code-block:: bash

   chrombert-tools <command> --help

Command Reference
=================

Overview
--------

.. toctree::
   :maxdepth: 1

   commands/overview

Embedding Commands
------------------

Extract general embeddings from pre-trained ChromBERT:

.. toctree::
   :maxdepth: 1

   commands/embed-gene
   commands/embed-region
   commands/embed-regulator
   commands/embed-cistrome

Cell-specific Embedding Commands
---------------------------------

Fine-tune ChromBERT and extract cell-type specific embeddings:

.. toctree::
   :maxdepth: 1

   commands/embed-cell-gene
   commands/embed-cell-region
   commands/embed-cell-regulator
   commands/embed-cell-cistrome

Imputation Commands
-------------------

Impute missing cistrome data:

.. toctree::
   :maxdepth: 1

   commands/impute-cistrome

Inference Commands
------------------

Infer transcriptional regulatory networks:

.. toctree::
   :maxdepth: 1

   commands/infer-trn
   commands/infer-cell-trn

Driver Factor Analysis
----------------------

Identify key regulatory factors:

.. toctree::
   :maxdepth: 1

   commands/find-driver-in-transition
   commands/find-driver-in-dual-region

Common Workflows
================

Workflow 1: Basic Gene Embedding
---------------------------------

Extract embeddings for a list of genes:

.. code-block:: bash

   chrombert-tools embed-gene \
     --gene "BRCA1;TP53;MYC" \
     --genome hg38 \
     --resolution 1kb \
     --odir gene_embeddings

See :doc:`commands/embed-gene` for details.

Workflow 2: Cell-specific Analysis
-----------------------------------

Get cell-type specific gene embeddings:

.. code-block:: bash

   chrombert-tools embed-cell-gene \
     --gene "BRCA1;TP53;MYC" \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --mode fast \
     --odir K562_embeddings

See :doc:`commands/embed-cell-gene` for details.

Workflow 3: Network Inference
------------------------------

Infer a cell-specific regulatory network:

.. code-block:: bash

   chrombert-tools infer-cell-trn \
     --tpm K562_expression.csv \
     --cell-type-bw K562_ATAC.bigwig \
     --cell-type-peak K562_peaks.bed \
     --top-k 20 \
     --genome hg38 \
     --resolution 1kb \
     --odir K562_network

See :doc:`commands/infer-cell-trn` for details.

Workflow 4: Driver Factor Analysis
-----------------------------------

Identify factors driving cell differentiation:

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
     --odir differentiation_drivers

See :doc:`commands/find-driver-in-transition` for details.

Getting Help
============

If you encounter issues:

1. Check the :doc:`installation` troubleshooting section
2. Review individual command documentation
3. Visit the GitHub issues page
4. Join community discussions

Next Steps
==========

* Start with :doc:`commands/overview` for common parameters and file formats
* Explore specific command documentation for detailed usage
* Check the :doc:`api` for programmatic access
