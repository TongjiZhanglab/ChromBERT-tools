========
Overview
========

Command Structure
=================

ChromBERT-tools provides a comprehensive suite of command-line tools for regulatory genomics analysis. All commands follow the pattern:

.. code-block:: bash

   chrombert-tools <command> [OPTIONS]

Getting Help
============

To see all available commands:

.. code-block:: bash

   chrombert-tools --help

To get help for a specific command:

.. code-block:: bash

   chrombert-tools <command> --help

Command Categories
==================

ChromBERT-tools commands are organized into five main categories:

1. **Embedding Commands**: Extract embeddings for genes, regions, regulators, and cistromes
2. **Cell-specific Embedding Commands**: Extract cell-type specific embeddings for genes, regions, regulators, and cistromes
3. **Imputation Commands**: Impute missing cistrome data
4. **Inference Commands**: Infer transcriptional regulatory networks (TRNs) and cell-type specific regulatory networks
5. **Driver Factor Analysis**: Identify driver factors in cell state transitions or differential regions


