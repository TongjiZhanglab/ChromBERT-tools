"""
Regulator-regulator interaction interpretation API.

Thin wrapper around ``chrombert_tools.cli.interpret_regulator_regulator_interactions.run``,
matching the ``chrombert-tools interpret_regulator_regulator_interactions`` CLI.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Optional, Union

import pandas as pd

from ..cli.interpret_regulator_regulator_interactions import run as _cli_run


def interpret_regulator_regulator_interactions(
    region: str,
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    regulator: Optional[Union[str, List[str]]] = None,
    batch_size: int = 64,
    num_workers: int = 8,
    quantile: float = 0.98,
    k_hop: int = 1,
    chrombert_cache_dir: Optional[str] = None,
    ft_ckpt: Optional[str] = None,
    ignore_regulator: Optional[str] = None,
    gep: bool = False,
    flank_window: int = 4,
    chrombert_region_file: Optional[str] = None,
    chrombert_regulator_file: Optional[str] = None,
    return_data: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Pool regulator embeddings over user regions, threshold pairwise cosines by a
    quantile to build a co-association graph, and optionally plot k-hop ego
    subgraphs for requested regulators.

    Args:
        region:
            BED path for focus regions (intersected with ChromBERT bins).
        odir:
            Output directory (embeddings subdirectory, TSV/PDF products).
        genome:
            ``hg38`` or ``mm10`` (``mm10`` is restricted to ``1kb`` in the CLI run).
        resolution:
            ChromBERT resolution (``1kb``, ``200bp``, ``2kb``, ``4kb``).
        regulator:
            Optional regulators to plot subnetworks for; string with ``;`` separators
            or a list of names (matched to ChromBERT's regulator list).
        batch_size:
            DataLoader batch size for region embedding passes.
        num_workers:
            DataLoader workers (passed through to ``build_interpret_config``).
        quantile:
            Quantile on upper-triangle cosine similarities used as edge threshold.
        k_hop:
            Ego-network radius when plotting subnetworks for requested regulators.
        chrombert_cache_dir:
            ChromBERT cache root; ``None`` → ``~/.cache/chrombert/data``.
        ft_ckpt:
            Optional fine-tuned ChromBERT checkpoint for embeddings.
        ignore_regulator:
            Regulator names to ignore (``;``-separated string), resolved against the
            ChromBERT regulator list.
        gep:
            Use the GEP multi-flank-window dataset/model path when ``True``.
        flank_window:
            Flank window size for the GEP path.
        chrombert_region_file:
            Optional override for ChromBERT reference region BED.
        chrombert_regulator_file:
            Optional override for ChromBERT regulator list file.
        return_data:
            If ``True``, return the edge-list DataFrame; if ``False``, write files only
            and return ``None``. Also forwarded as ``return_fig`` when plotting PDFs.

    Returns:
        DataFrame of all regulator cosine similarities and graph edges above the cosine threshold, or ``None`` if
        ``return_data`` is ``False``.
    """
    if regulator is not None:
        regulator_str = ";".join(regulator) if isinstance(regulator, list) else regulator
    else:
        regulator_str = None

    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")

    args = SimpleNamespace(
        region=region,
        regulator=regulator_str,
        odir=odir,
        genome=genome.lower() if isinstance(genome, str) else genome,
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        quantile=quantile,
        k_hop=k_hop,
        chrombert_cache_dir=chrombert_cache_dir,
        ft_ckpt=ft_ckpt,
        ignore_regulator=ignore_regulator,
        gep=gep,
        flank_window=flank_window,
    )
    if chrombert_region_file is not None:
        args.chrombert_region_file = chrombert_region_file
    if chrombert_regulator_file is not None:
        args.chrombert_regulator_file = chrombert_regulator_file

    cos_sim_df, df_edges = _cli_run(args, return_data=return_data)
    if return_data:
        return cos_sim_df, df_edges
    return None, None
