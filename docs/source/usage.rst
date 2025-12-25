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

Notebooks
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   Extract general embeddings <examples/cli/embed>


Cell-type-specific Embedding CLI
--------------------------------

Fine-tune ChromBERT and extract cell-type-specific embeddings:

.. toctree::
   :maxdepth: 1

   commands/embed_cell_gene
   commands/embed_cell_region
   commands/embed_cell_regulator
   commands/embed_cell_cistrome

Notebooks
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   Cell-type-specific Embedding <examples/cli/embed_cell_specific>

TRN Inference CLI
-----------------

Infer transcriptional regulatory networks:

.. toctree::
   :maxdepth: 1

   commands/infer_trn
   commands/infer_cell_trn

Notebooks
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   Infer transcriptional regulatory networks (TRNs) <examples/cli/infer_trn>
   Infer cell-type-specific transcriptional regulatory networks (TRNs) <examples/cli/infer_cell_trn>

Imputation CLI
--------------

Impute missing cistrome data:

.. toctree::
   :maxdepth: 1

   commands/impute_cistrome

Notebooks
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   Impute cistromes <examples/cli/impute_cistrome>

Driver Factor CLI
-----------------

Identify key regulatory factors:

.. toctree::
   :maxdepth: 1

   commands/find_driver_in_transition
   commands/find_driver_in_dual_region

Notebooks
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   Find driver factors in cell-state transitions <examples/cli/find_driver_in_transition>
   Find driver factors in dual-functional regions <examples/cli/find_driver_in_dual_region>

General Notebooks
-----------------

Workflows and examples for running ChromBERT-tools in a Singularity container, including embedding, TRN inference etc.

.. toctree::
   :maxdepth: 1

   Run ChromBERT-tools with Singularity <examples/cli/singularity_use>


API Reference
=============

In addition to CLI commands, you can now call ChromBERT-tools directly in Python. It currently supports tasks that do not require fine-tuning:

Embedding API
-------------

.. toctree::
   :maxdepth: 1

   Extract general embeddings from the pre-trained ChromBERT model <examples/api/embed>


TRN Inference API
-----------------

.. toctree::
   :maxdepth: 1

   Infer transcriptional regulatory networks (TRNs) <examples/api/infer_trn>

Imputation API
--------------

.. toctree::
   :maxdepth: 1

   Impute cistromes <examples/api/impute_cistrome>



Next Steps
==========

* Explore specific command documentation for detailed usage.
