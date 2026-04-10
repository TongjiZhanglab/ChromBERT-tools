# -*- coding: utf-8 -*-
"""ChromBERT-tools: Command-line tools for ChromBERT"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='nxviz')

__version__ = "1.0.0"

# from .cli import make_dataset
from .cli import check_region_file
# from .api.embed_gene import embed_gene
# from .api.embed_cistrome import embed_cistrome
from .api.embed_region import embed_region
from .api.embed_regulator import embed_regulator
from .api.predict_tf_binding_regions import predict_tf_binding_regions
from .api.interpret_regulator_regulator_interactions import interpret_regulator_regulator_interactions
from .api.interpret_region_region_interactions import interpret_region_region_interactions
from .api.interpret_regulator_effects_between_region_groups import (
    interpret_regulator_effects_between_region_groups,
)

__all__ = [
    "__version__",
    # "embed_gene",
    # "embed_cistrome",
    "embed_region",
    "embed_regulator",
    "predict_tf_binding_regions",
    "interpret_regulator_regulator_interactions",
    "interpret_region_region_interactions",
    "interpret_regulator_effects_between_region_groups",
]