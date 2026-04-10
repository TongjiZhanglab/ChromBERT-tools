"""
Regulator embedding API.

Thin wrapper around ``chrombert_tools.cli.embed_regulator.run`` so Python callers
match the ``chrombert-tools embed_regulator`` CLI (general or cell-specific
mode, path resolution, and file checks).
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..cli.embed_regulator import run as _cli_run


def embed_regulator(
    region: str,
    regulator: Union[str, List[str]],
    odir: str = "./output",
    oname: str = "regulator_emb",
    genome: str = "hg38",
    resolution: str = "1kb",
    mode: str = "fast",
    batch_size: int = 4,
    num_workers: int = 8,
    chrombert_cache_dir: Optional[str] = None,
    ft_ckpt: Optional[str] = None,
    cell_type_bw: Optional[str] = None,
    cell_type_peak: Optional[str] = None,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    chrombert_regulator_file: Optional[str] = None,
    return_embeddings: bool = True,
    # ignore_regulator: Optional[str] = None,
    # gep: bool = False,
    # flank_window: int = 4,
) -> Tuple[
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Optional[pd.DataFrame],
]:
    """
    Extract regulator embeddings for ChromBERT regulators on user regions that
    overlap the ChromBERT reference tiling (general pretrained or cell-specific model).

    Args:
        region:
            Path to a BED file of genomic regions. Only intervals overlapping
            ChromBERT bins are embedded; non-overlapping rows are reported under
            ``odir`` (see CLI reports).
        regulator:
            Regulator name(s) as a string or list. Multiple names use ``;`` when
            passed as a string (e.g. ``"EZH2;BRD4"``), matching the CLI.
        odir:
            Output directory. Writes ``model_input.tsv``, ``overlap_region.bed``,
            ``mean_{oname}.pkl``, ``region_aware_{oname}.hdf5``, and related logs.
        oname:
            Basename tag for outputs (e.g. ``mean_{oname}.pkl``,
            ``region_aware_{oname}.hdf5``).
        genome:
            Reference assembly: ``hg38`` or ``mm10`` (case-insensitive; lowercased
            internally).
        resolution:
            ChromBERT resolution: ``1kb``, ``200bp``, ``2kb``, or ``4kb``. Must match
            cached data for the genome.
        mode:
            When fine-tuning a cell-specific model from BigWig + peaks: ``fast`` or
            ``full``. Ignored in practice if ``ft_ckpt`` is provided (no training).
        batch_size:
            DataLoader batch size for forward passes through the model.
        num_workers:
            Number of worker processes for the embedding DataLoader.
        chrombert_cache_dir:
            ChromBERT data root (config, checkpoint, HDF5, etc.). If ``None``,
            uses ``~/.cache/chrombert/data``.
        ft_ckpt:
            Fine-tuned checkpoint path. If set, uses the cell-specific model without
            training from accessibility tracks.
        cell_type_bw:
            Cell-type BigWig for accessibility. Use together with ``cell_type_peak``
            to enable cell-specific mode when ``ft_ckpt`` is not set.
        cell_type_peak:
            Peak BED matching ``cell_type_bw``; required as a pair for that path.
        chrombert_region_file:
            Optional override for the ChromBERT reference region BED resolved from
            the cache.
        chrombert_region_emb_file:
            Optional override for precomputed region embedding ``.npy`` if used by
            the pipeline.
        chrombert_regulator_file:
            Optional override for the ChromBERT regulator list file resolved from
            the cache.
        return_embeddings:
            If ``True``, return mean and per-region regulator arrays plus the overlap
            table. If ``False``, only write files and return ``(None, None, None)``.
        ignore_regulator:
            Ignore regulator. Use ';' to separate multiple regulators.
        gep:
            Use GEP model (multi-flank-window).
        flank_window:
            Flank window size for gep model.

    Returns:
        Tuple ``(regulator_means, regulator_emb_dict, overlap_bed)`` when
        ``return_embeddings`` is ``True``:

        - **regulator_means**: Per-regulator mean embedding across all overlapping
          regions (~768-d vectors), as a ``dict`` keyed by regulator name.
        - **regulator_emb_dict**: Per-regulator stacked embeddings for each region
          (concatenated along the region axis), as a ``dict`` keyed by name.
        - **overlap_bed**: DataFrame of overlapping regions and indices used for the
          forward pass (same rows as the dataloader order).

        If ``return_embeddings`` is ``False``, returns ``(None, None, None)``.
    """
    if isinstance(regulator, list):
        regulator_str = ";".join(regulator)
    else:
        regulator_str = regulator

    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")

    args = SimpleNamespace(
        region=region,
        regulator=regulator_str,
        odir=odir,
        oname=oname,
        genome=genome.lower() if isinstance(genome, str) else genome,
        resolution=resolution,
        mode=mode,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
        ft_ckpt=ft_ckpt,
        cell_type_bw=cell_type_bw,
        cell_type_peak=cell_type_peak,
        # ignore_regulator=ignore_regulator,
        # gep=gep,
        # flank_window=flank_window,
    )
    if chrombert_region_file is not None:
        args.chrombert_region_file = chrombert_region_file
    if chrombert_region_emb_file is not None:
        args.chrombert_region_emb_file = chrombert_region_emb_file
    if chrombert_regulator_file is not None:
        args.chrombert_regulator_file = chrombert_regulator_file

    out = _cli_run(args, return_data=return_embeddings)
    if return_embeddings:
        regulator_means, regulator_emb_dict, regions = out
        return regulator_means, regulator_emb_dict, regions
    return None, None, None
