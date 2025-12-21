# -*- coding: utf-8 -*-
"""ChromBERT-tools: Command-line tools for ChromBERT"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='nxviz')

__version__ = "1.0.0"

__all__ = ["__version__"]

from chrombert_tools import cli
from chrombert_tools.cli.embed_cistrome import embed_cistrome
from chrombert_tools.cli.embed_cell_cistrome import embed_cell_cistrome
from chrombert_tools.cli.embed_gene import embed_gene
from chrombert_tools.cli.embed_cell_gene import embed_cell_gene
from chrombert_tools.cli.embed_region import embed_region
from chrombert_tools.cli.embed_cell_region import embed_cell_region
from chrombert_tools.cli.embed_regulator import embed_regulator
from chrombert_tools.cli.embed_cell_regulator import embed_cell_regulator
from chrombert_tools.cli.impute_cistrome import impute_cistrome
from chrombert_tools.cli.infer_trn import infer_trn
from chrombert_tools.cli.infer_cell_trn import infer_cell_trn
from chrombert_tools.cli.find_dirver_in_transition import find_driver_in_transition
from chrombert_tools.cli.find_driver_in_dual_region import find_driver_in_dual_region

__all__ = [
    "__version__",
    "embed_cistrome",
    "embed_cell_cistrome",
    "embed_gene",
    "embed_cell_gene",
    "embed_region",
    "embed_cell_region",
    "embed_regulator",
    "embed_cell_regulator",
    "impute_cistrome",
    "infer_trn",
    "infer_cell_trn",
    "find_driver_in_transition",
    "find_driver_in_dual_region",
]