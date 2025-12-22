"""
API interface for cistrome imputation
Provides a Python-friendly interface reusing the CLI implementation.
"""
from types import SimpleNamespace
from typing import List, Union, Tuple
from ..cli.impute_cistrome import run as _cli_run
import os

def impute_cistrome(
    region: str,
    cistromes: Union[str, List[str]],
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    batch_size: int = 4,
    chrombert_cache_dir: str = None,
):
    
    # Convert list to semicolon-separated string if needed
    if isinstance(cistromes, list):
        cistrome_str = ";".join(cistromes)
    else:
        cistrome_str = cistromes
    
    # Set default cache dir if not provided
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")
    
    # Create args namespace (same as CLI)
    args = SimpleNamespace(
        region=region,
        cistrome=cistrome_str,
        odir=odir,
        genome=genome,
        resolution=resolution,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    
    # Run the core logic (reuse CLI implementation)
    results_pro_df = _cli_run(args, return_data=True)
    
    return results_pro_df


