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

Embedding Commands
------------------

Extract general embeddings from pre-trained ChromBERT:

.. toctree::
   :maxdepth: 1

   commands/embed_gene
   commands/embed_region
   commands/embed_regulator
   commands/embed_cistrome

Cell-specific Embedding Commands
---------------------------------

Fine-tune ChromBERT and extract cell-type specific embeddings:

.. toctree::
   :maxdepth: 1

   commands/embed_cell_gene
   commands/embed_cell_region
   commands/embed_cell_regulator
   commands/embed_cell_cistrome

Imputation Commands
-------------------

Impute missing cistrome data:

.. toctree::
   :maxdepth: 1

   commands/impute_cistrome

Inference Commands
------------------

Infer transcriptional regulatory networks:

.. toctree::
   :maxdepth: 1

   commands/infer_trn
   commands/infer_cell_trn

Driver Factor Analysis
----------------------

Identify key regulatory factors:

.. toctree::
   :maxdepth: 1

   commands/find_driver_in_transition
   commands/find_driver_in_dual_region


Next Steps
==========

* Explore specific command documentation for detailed usage
