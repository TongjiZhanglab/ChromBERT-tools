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

Generation of regulation-informed embeddings
--------------------------------------------

Generate regulation-informed embeddings from the pre-trained ChromBERT model or fine-tuned ChromBERT model:

Commands
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   commands/embed_gene
   commands/embed_region
   commands/embed_regulator
   commands/embed_cistrome
   commands/embed_cell_gene
   commands/embed_cell_region
   commands/embed_cell_regulator
   commands/embed_cell_cistrome

Notebooks
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   Extract general embeddings <examples/cli/embed>
   Cell-type-specific Embedding <examples/cli/embed_cell_specific>


Interpretation of regulation-informed embeddings
------------------------------------------------

Interpret regulation-informed embeddings:

Commands
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   commands/infer_ep
   commands/infer_regulator_network
   commands/impute_cistrome
   commands/infer_cell_key_regulator
   commands/find_driver_in_transition
   commands/find_context_specific_cofactor

Notebooks
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   Infer regulator-regulator networks <examples/cli/infer_regulator_network>
   Infer cell-type-specific regulator-regulator networks <examples/cli/infer_cell_regulator_network>
   Infer enhancer-promoter loops <examples/cli/infer_ep>
   Impute cistromes <examples/cli/impute_cistrome>
   Find driver factors in cell-state transitions <examples/cli/find_driver_in_transition>
   Find context-specific cofactors in different regions <examples/cli/find_context_specific_cofactor>

General Notebooks
-----------------

Workflows and examples for running ChromBERT-tools in a Singularity container, including embedding, regulator-regulator network inference etc.

.. toctree::
   :maxdepth: 1

   Run ChromBERT-tools with Singularity <examples/cli/singularity_use>


API Reference
=============

In addition to CLI commands, you can now call ChromBERT-tools directly in Python. It currently supports tasks that do not require fine-tuning:

Generation of regulation-informed embeddings API
-----------------------------------------------

.. toctree::
   :maxdepth: 1

   Extract general embeddings from the pre-trained ChromBERT model <examples/api/embed>


Interpretation of regulation-informed embeddings API
-----------------------------------------

.. toctree::
   :maxdepth: 1

   Infer regulator-regulator networks <examples/api/infer_regulator_network>
   Infer enhancer-promoter loops <examples/api/infer_ep>
   Impute cistromes <examples/api/impute_cistrome>

Next Steps
==========

* Explore specific command documentation for detailed usage.
