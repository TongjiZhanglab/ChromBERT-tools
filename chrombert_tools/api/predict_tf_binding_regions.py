"""
Predict TF binding on specified regions across cell types API.

Thin wrapper around ``chrombert_tools.cli.predict_tf_binding_regions.run`` so Python callers
match the ``chrombert-tools predict_tf_binding_regions`` CLI (path resolution and file checks).
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Optional, Union

import pandas as pd

from ..cli.predict_tf_binding_regions import run as _cli_run


def predict_tf_binding_regions(
    region: str,
    cistrome: Union[str, List[str]],
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    batch_size: int = 4,
    chrombert_cache_dir: Optional[str] = None,
    oname: str = "cistrome_impute",
    num_workers: int = 8,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    return_results: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Predict TF binding probabilities on user regions via the ChromBERT
    prompt/cistrome head. Writes ``results_prob_df.csv`` and per-track BigWig files
    under ``odir``.

    Args:
        region:
            Path to a BED file of regions to score. Overlap with ChromBERT bins is
            handled like the CLI; non-overlapping rows are recorded under ``odir``.
        cistrome:
            Target cistromes as ``"factor:cell"`` strings. Multiple targets use
            ``;`` in a single string (e.g. ``"BCL11A:GM12878;BRD4:MCF7"``), or pass
            a list of such strings which will be joined the same way.
        odir:
            Output directory for ``model_input.tsv``, overlap BEDs,
            ``results_prob_df.csv``, and ``{factor}_{cell}.bw`` tracks.
        genome:
            Reference assembly: ``hg38`` or ``mm10`` (case-insensitive).
        resolution:
            ChromBERT resolution for this task; imputation currently supports ``1kb``
            (same restriction as the CLI).
        batch_size:
            DataLoader batch size during inference.
        chrombert_cache_dir:
            ChromBERT data root. If ``None``, uses ``~/.cache/chrombert/data``.
        oname:
            Output name prefix on ``args`` for CLI parity (reserved for future
            naming; main outputs still use fixed names under ``odir``).
        num_workers:
            DataLoader worker processes.
        chrombert_region_file:
            Optional override for the ChromBERT reference region BED.
        chrombert_region_emb_file:
            Optional override for precomputed region embeddings ``.npy`` if used by
            the resolved pipeline.
        return_results:
            If ``True``, return the merged probability DataFrame in memory. If
            ``False``, only write files and return ``None``.

    Returns:
        DataFrame with input coordinates, ChromBERT bin coordinates, and one column
        per requested cistrome (probabilities in ``[0, 1]``). ``None`` if
        ``return_results`` is ``False``.
    """
    if isinstance(cistrome, list):
        cistrome_str = ";".join(cistrome)
    else:
        cistrome_str = cistrome

    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")

    args = SimpleNamespace(
        region=region,
        cistrome=cistrome_str,
        odir=odir,
        oname=oname,
        genome=genome.lower() if isinstance(genome, str) else genome,
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    if chrombert_region_file is not None:
        args.chrombert_region_file = chrombert_region_file
    if chrombert_region_emb_file is not None:
        args.chrombert_region_emb_file = chrombert_region_emb_file

    out = _cli_run(args, return_data=return_results)
    if return_results:
        return out
    return None
