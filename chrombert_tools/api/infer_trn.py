"""
API interface for TRN inference
Provides a Python-friendly interface reusing the CLI implementation.
"""
from types import SimpleNamespace
from typing import List, Optional, Union
from ..cli.infer_trn import run as _cli_run
import os

def infer_trn(
    region: str,
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    regulators: Optional[Union[str, List[str]]] = None,
    batch_size: int = 64,
    num_workers: int = 8,
    quantile: float = 0.99,
    k_hop: int = 1,
    chrombert_cache_dir: Optional[str] = None,
):
    
    # Convert regulator list to semicolon-separated string if needed
    if regulators is not None:
        if isinstance(regulators, list):
            regulator_str = ";".join(regulators)
        else:
            regulator_str = regulators
    else:
        regulator_str = None
    
    # Set default cache dir if not provided
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")
    
    # Create args namespace (same as CLI)
    args = SimpleNamespace(
        region=region,
        regulator=regulator_str,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        quantile=quantile,
        k_hop=k_hop,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    
    # Run the core logic (reuse CLI implementation)
    df_edges = _cli_run(args, return_data=True)
    
    return df_edges


