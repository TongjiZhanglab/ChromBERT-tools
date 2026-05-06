"""
Python API for ChromBERT-tools
Provides programmatic access to ChromBERT functionality
"""

# from .embed_gene import embed_gene
# from .embed_cistrome import embed_cistrome
from .embed_region import embed_region
from .embed_regulator import embed_regulator
from .predict_tf_binding_regions import predict_tf_binding_regions
from .interpret_region_region_interactions import interpret_region_region_interactions
from .interpret_regulator_regulator_interactions import interpret_regulator_regulator_interactions
from .interpret_regulator_effects_between_region_groups import (
    interpret_regulator_effects_between_region_groups,
)
from .region_function_classification import region_function_classification
from .region_activity_regression import region_activity_regression
from .gene_activity_regression import gene_activity_regression

__all__ = [
    # "embed_gene",
    # "embed_cistrome",
    # "impute_cistrome",
    "embed_region",
    "embed_regulator",
    "predict_tf_binding_regions",
    "interpret_region_region_interactions",
    "interpret_regulator_regulator_interactions",
    "interpret_regulator_effects_between_region_groups",
    "region_function_classification",
    "region_activity_regression",
    "gene_activity_regression",
]

