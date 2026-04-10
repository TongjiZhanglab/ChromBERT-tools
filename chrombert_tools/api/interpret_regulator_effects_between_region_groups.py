"""
Regulator effects between two region groups API.

Thin wrapper around ``chrombert_tools.cli.interpret_regulator_effects_between_regions_groups.run``,
matching the ``chrombert-tools interpret_regulator_effects_between_regions_groups`` CLI.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from ..cli.interpret_regulator_effects_between_region_groups import run as _cli_run


def interpret_regulator_effects_between_region_groups(
    region1_file: str,
    region2_file: str,
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    batch_size: int = 4,
    chrombert_cache_dir: Optional[str] = None,
    ft_ckpt: Optional[str] = None,
    ignore_regulator: Optional[str] = None,
    gep: bool = False,
    flank_window: int = 4,
    num_workers: int = 8,
    chrombert_region_file: Optional[str] = None,
    chrombert_regulator_file: Optional[str] = None,
    return_results: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Rank regulators whose pooled embeddings differ most between two region BEDs
    (embedding-shift style analysis). Writes ``factor_importance_rank.csv`` under
    ``{odir}/results/`` and embedding caches under ``{odir}/emb/``.

    Args:
        region1_file:
            Path to BED file for region set 1 (intersected with ChromBERT bins).
        region2_file:
            Path to BED file for region set 2.
        odir:
            Output root; creates ``dataset/``, ``emb/``, and ``results/`` subdirs.
        genome:
            Reference assembly ``hg38`` or ``mm10`` (case-insensitive).
        resolution:
            ChromBERT bin resolution: ``200bp``, ``1kb``, ``2kb``, or ``4kb``.
        batch_size:
            DataLoader batch size for embedding passes.
        chrombert_cache_dir:
            ChromBERT cache root; ``None`` → ``~/.cache/chrombert/data``.
        ft_ckpt:
            Optional fine-tuned checkpoint; if ``None``, uses pretrained weights from
            the cache layout.
        ignore_regulator:
            Regulators to drop (``;``-separated), resolved via ChromBERT regulator list.
        gep:
            If ``True``, use the GEP multi-flank-window data path.
        flank_window:
            Flank bins for GEP when ``gep`` is ``True``.
        num_workers:
            DataLoader workers (see ``build_interpret_config``).
        chrombert_region_file:
            Optional override for ChromBERT reference region BED.
        chrombert_regulator_file:
            Optional override for ChromBERT regulator list file.
        return_results:
            If ``True``, return the ranking DataFrame; if ``False``, only write files
            and return ``None``.

    Returns:
        DataFrame of regulator importance / ranking, or ``None`` when
        ``return_results`` is ``False``.
    """
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")

    args = SimpleNamespace(
        region1_file=region1_file,
        region2_file=region2_file,
        ft_ckpt=ft_ckpt,
        ignore_regulator=ignore_regulator,
        gep=gep,
        flank_window=flank_window,
        genome=genome.lower() if isinstance(genome, str) else genome,
        resolution=resolution,
        odir=odir,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
        num_workers=num_workers,
    )
    if chrombert_region_file is not None:
        args.chrombert_region_file = chrombert_region_file
    if chrombert_regulator_file is not None:
        args.chrombert_regulator_file = chrombert_regulator_file

    regulator_efftec_rank_df = _cli_run(args, return_data=return_results)
    return regulator_efftec_rank_df
