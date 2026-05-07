=====
Usage
=====

ChromBERT-tools can be used in two ways:

* **Command-line interface (CLI)** — run commands from a terminal using
  ``chrombert-tools <command>``.
* **Python API** — call the same core functions directly from Python using
  ``import chrombert_tools``.

Most users can start with the CLI. The Python API is useful for customized
workflows, notebooks, or integration into existing pipelines.

For runnable examples, see the example notebooks:

* CLI examples: https://github.com/TongjiZhanglab/ChromBERT-tools/tree/main/examples/cli
* Python API examples: https://github.com/TongjiZhanglab/ChromBERT-tools/tree/main/examples/api


Quick start
===========

List all available commands:

.. code-block:: bash

   chrombert-tools --help

Get help for a specific command:

.. code-block:: bash

   chrombert-tools <command> --help

Run the example notebooks with GPU support inside the Apptainer image:

.. code-block:: bash

   cd ChromBERT-tools/examples/
   apptainer exec --nv /path/to/chrombert-tools.sif jupyter-notebook


.. _cli-api-reference:

Command reference
=================

ChromBERT-tools commands are organized into three functional layers and a set of
end-to-end application workflows. Each command can be used from the CLI as
``chrombert-tools <command>`` or called through the Python API.


Representation generation
-------------------------

Generate context-specific regulatory representations for genomic regions or
regulators.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - :doc:`commands/embed_region`
     - Generate embeddings for specified genomic regions or promoter-centered gene regions.
   * - :doc:`commands/embed_regulator`
     - Generate regulator embeddings for specified regulators across specified genomic regions.

.. toctree::
   :maxdepth: 1
   :hidden:

   commands/embed_region
   commands/embed_regulator


Predictive modeling
-------------------

Build supervised models on top of ChromBERT-derived representations for region-
or gene-level prediction tasks.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - :doc:`commands/region_function_classification`
     - Classify genomic regions into functional classes.
   * - :doc:`commands/region_activity_regression`
     - Predict quantitative region activity, such as chromatin accessibility or activity fold change.
   * - :doc:`commands/gene_activity_regression`
     - Predict gene expression or expression fold change from TSS-centered regulatory context.

.. toctree::
   :maxdepth: 1
   :hidden:

   commands/region_function_classification
   commands/region_activity_regression
   commands/gene_activity_regression


Regulatory interpretation
-------------------------

Interpret context-specific regulatory representations to infer relationships
among regions, regulators, or region groups.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Command
     - Description
   * - :doc:`commands/interpret_region_region_interactions`
     - Identify functionally similar genomic regions.
   * - :doc:`commands/interpret_regulator_regulator_interactions`
     - Identify potentially cooperative regulators.
   * - :doc:`commands/interpret_regulator_effects_between_region_groups`
     - Compare regulator effects between two groups of genomic regions.

.. toctree::
   :maxdepth: 1
   :hidden:

   commands/interpret_region_region_interactions
   commands/interpret_regulator_regulator_interactions
   commands/interpret_regulator_effects_between_region_groups

Integrated workflow
-------------------

ChromBERT-tools also provides an integrated workflow for inferring
cell-type-specific enhancer–promoter interactions. This workflow combines
region representation generation, cell-type-specific predictive modeling, and
region–region interpretation into a single analysis pipeline.

Tutorial notebook:

.. toctree::
   :maxdepth: 1
   :hidden:

   examples/api/infer_cell_type_specific_ep_interactions


End-to-end application commands
-------------------------------

Run integrated workflows for common biological analysis tasks.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Command
     - Description
   * - :doc:`commands/predict_cell_type_master_regulators`
     - Infer cell-type-specific key regulators.
   * - :doc:`commands/predict_transition_driver_regulators`
     - Identify driver regulators during cell-state transitions.
   * - :doc:`commands/predict_regulator_context_cofactors`
     - Identify context-specific cofactors for a target regulator.
   * - :doc:`commands/predict_tf_binding_regions`
     - Predict TF-binding regions in a target cellular context.

.. toctree::
   :maxdepth: 1
   :hidden:

   commands/predict_cell_type_master_regulators
   commands/predict_transition_driver_regulators
   commands/predict_regulator_context_cofactors
   commands/predict_tf_binding_regions


Tutorials
=========

Each command page includes a **Tutorials** section with links to the matching CLI
and / or Python API notebook in the ``examples/`` directory.

You can also browse the example notebooks directly on GitHub:

* CLI examples: https://github.com/TongjiZhanglab/ChromBERT-tools/tree/main/examples/cli
* Python API examples: https://github.com/TongjiZhanglab/ChromBERT-tools/tree/main/examples/api

The following tutorial is not tied to a single command:

.. toctree::
   :maxdepth: 1

   Run with Apptainer / Singularity <examples/cli/singularity_use>


Next steps
==========

* Open a command page to view its full options, required inputs, outputs, tips,
  and tutorials.
* Combine commands to build customized workflows, for example:
  ``embed_regulator`` → ``interpret_regulator_regulator_interactions`` →
  ``predict_cell_type_master_regulators``.