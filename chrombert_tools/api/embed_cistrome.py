"""
API interface for cistrome embedding
Provides a Python-friendly interface reusing the CLI implementation.
"""
from types import SimpleNamespace
from typing import Dict, List, Optional, Union, Tuple
from ..cli.embed_cistrome import run as _cli_run
import os

def embed_cistrome(
    region: str,
    cistrome: Union[str, List[str]],
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    batch_size: int = 64,
    num_workers: int = 8,
    chrombert_cache_dir: Optional[str] = None,
):

    # Convert list to semicolon-separated string if needed
    if isinstance(cistrome, list):
        cistrome_str = ";".join(cistrome)
    else:
        cistrome_str = cistrome
    
    # Set default cache dir if not provided
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")
    
    # Create args namespace (same as CLI)
    args = SimpleNamespace(
        region=region,
        cistrome=cistrome_str,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    
    # Run the core logic (reuse CLI implementation)
    cistrome_means, cistrome_emb_dict, regions = _cli_run(args, return_data=True)
    
    return cistrome_means, cistrome_emb_dict, regions

