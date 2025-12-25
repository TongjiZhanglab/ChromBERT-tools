=====
Usage
=====

Welcome to the ChromBERT-tools usage guide. This section provides documentation for all command-line tools.

Quick Start
===========

To see all available commands:

.. code-block:: bash

   chrombert-tools --help

To get help for a specific command:

.. code-block:: bash

   chrombert-tools <command> --help

CLI Reference
=============

Embedding CLI
-------------

Extract general embeddings from the pre-trained ChromBERT model:

.. toctree::
   :maxdepth: 1

   commands/embed_gene
   commands/embed_region
   commands/embed_regulator
   commands/embed_cistrome

Example Notebooks
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/cli/embed
   examples/cli/singularity_use


Cell-type-specific Embedding CLI
--------------------------------

Fine-tune ChromBERT and extract cell-type-specific embeddings:

.. toctree::
   :maxdepth: 1

   commands/embed_cell_gene
   commands/embed_cell_region
   commands/embed_cell_regulator
   commands/embed_cell_cistrome

Example Notebooks
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/cli/embed_cell_specific

TRN Inference CLI
-----------------

Infer transcriptional regulatory networks:

.. toctree::
   :maxdepth: 1

   commands/infer_trn
   commands/infer_cell_trn

Example Notebooks
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/cli/infer_trn
   examples/cli/infer_cell_trn

Imputation CLI
--------------

Impute missing cistrome data:

.. toctree::
   :maxdepth: 1

   commands/impute_cistrome

Example Notebooks
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/cli/impute_cistrome

Driver Factor CLI
-----------------

Identify key regulatory factors:

.. toctree::
   :maxdepth: 1

   commands/find_driver_in_transition
   commands/find_driver_in_dual_region

Example Notebooks
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/cli/find_driver_in_transition
   examples/cli/find_driver_in_dual_region

API Reference
=============

In addition to CLI commands, you can now call ChromBERT-tools directly in Python. It currently supports tasks that do not require fine-tuning:

Embedding API
-------------

Extract general embeddings from the pre-trained ChromBERT model:

Example Notebooks
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/api/embed


TRN Inference API
-----------------

Infer transcriptional regulatory networks:

Example Notebooks
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/api/infer_trn

Imputation API
--------------

Impute missing cistrome data:

Example Notebooks
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/api/impute_cistrome

Next Steps
==========

* Explore specific command documentation for detailed usage.
