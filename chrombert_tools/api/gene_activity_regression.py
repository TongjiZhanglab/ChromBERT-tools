"""
Python API for gene expression (GEP) regression (``gene_activity_repression`` CLI).
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

from ..cli.gene_activity_regression import run as _cli_run
from ..cli.prediction_run_result import ChrombertPredictionRunResult


def gene_activity_regression(
    exp_tpm1: Optional[str] = None,
    exp_tpm2: Optional[str] = None,
    direction: str = "2-1",
    predict_file: Optional[str] = None,
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    mode: str = "fast",
    ft_ckpt: Optional[str] = None,
    chrombert_cache_dir: Optional[str] = None,
    batch_size: int = 4,
    flank_window: int = 4,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    chrombert_gene_meta: Optional[str] = None,
    hdf5_file: Optional[str] = None,
    pretrain_ckpt: Optional[str] = None,
    mtx_mask: Optional[str] = None,
    meta_file: Optional[str] = None,
) -> ChrombertPredictionRunResult:
    """
    Same pipeline as ``chrombert-tools gene_activity_regression``.

    Returns:
        :class:`~chrombert_tools.prediction_run_result.ChrombertPredictionRunResult`
        with ``model``, ``data_config``, ``predictions_df``, ``ft_ckpt``, ``train_output_dir``.
    """
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")

    args = SimpleNamespace(
        exp_tpm1=exp_tpm1,
        exp_tpm2=exp_tpm2,
        direction=direction,
        predict_file=predict_file,
        odir=odir,
        genome=genome.lower() if isinstance(genome, str) else genome,
        resolution=resolution,
        mode=str(mode).lower(),
        ft_ckpt=ft_ckpt,
        chrombert_cache_dir=chrombert_cache_dir,
        batch_size=batch_size,
        flank_window=flank_window,
    )
    if chrombert_region_file is not None:
        args.chrombert_region_file = chrombert_region_file
    if chrombert_region_emb_file is not None:
        args.chrombert_region_emb_file = chrombert_region_emb_file
    if chrombert_gene_meta is not None:
        args.chrombert_gene_meta = chrombert_gene_meta
    if hdf5_file is not None:
        args.hdf5_file = hdf5_file
    if pretrain_ckpt is not None:
        args.pretrain_ckpt = pretrain_ckpt
    if mtx_mask is not None:
        args.mtx_mask = mtx_mask
    if meta_file is not None:
        args.meta_file = meta_file

    return _cli_run(args)
