"""
API interface for gene embedding
Provides a Python-friendly interface reusing the CLI implementation.
"""
from types import SimpleNamespace
from typing import Dict, List, Optional, Union
import numpy as np
import os
from ..cli.embed_gene import run as _cli_run


def embed_gene(
    gene: Union[str, List[str]],
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    chrombert_cache_dir: Optional[str] = None,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    chrombert_gene_meta: Optional[str] = None,
):
    
    # Convert list to semicolon-separated string if needed
    if isinstance(gene, list):
        gene_str = ";".join(gene)
    else:
        gene_str = gene
    
    # Set default cache dir if not provided
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")
    
    # Create args namespace (same as CLI)
    args = SimpleNamespace(
        gene=gene_str,
        odir=odir,
        genome=genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
        chrombert_gene_meta=chrombert_gene_meta,
    )
    
    # Run the core logic (reuse CLI implementation)
    gene_emb_dict = _cli_run(args, return_data=True)
    
    return gene_emb_dict