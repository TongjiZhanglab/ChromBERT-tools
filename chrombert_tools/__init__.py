# -*- coding: utf-8 -*-
"""ChromBERT-tools: Command-line tools for ChromBERT"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='nxviz')

__version__ = "1.1.0"


from .cli import check_region_file, umap_plot, resolve_paths
from .api.embed_region import embed_region
from .api.embed_regulator import embed_regulator
from .api.predict_tf_binding_regions import predict_tf_binding_regions
from .api.interpret_regulator_regulator_interactions import interpret_regulator_regulator_interactions
from .api.interpret_region_region_interactions import interpret_region_region_interactions
from .api.interpret_regulator_effects_between_region_groups import (
    interpret_regulator_effects_between_region_groups,
)
from .api.region_function_classification import region_function_classification
from .api.region_activity_regression import region_activity_regression
from .api.gene_activity_regression import gene_activity_regression
from .cli.embed_run_result import ChrombertEmbedRegulatorRunResult, ChrombertEmbedRunResult
from .cli.prediction_run_result import ChrombertPredictionRunResult

__all__ = [
    "__version__",
    "embed_region",
    "embed_regulator",
    "predict_tf_binding_regions",
    "interpret_regulator_regulator_interactions",
    "interpret_region_region_interactions",
    "interpret_regulator_effects_between_region_groups",
    "region_function_classification",
    "region_activity_regression",
    "gene_activity_regression",
]