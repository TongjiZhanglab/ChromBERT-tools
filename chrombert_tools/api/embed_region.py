"""
API interface for region embedding
Provides a Python-friendly interface reusing the CLI implementation.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..cli.utils import check_files, resolve_paths
from ..cli.utils_embed import build_cell_model_emb, is_cell_specific
from ..cli.embed_region import (
    get_required_keys,
    run_gene_cell,
    run_gene_general,
    run_region_cell,
    run_region_general,
    validate_args,
)


def embed_region(
    region: Optional[str] = None,
    gene: Optional[str] = None,
    odir: str = "./output",
    oname: str = "emb",
    genome: str = "hg38",
    resolution: str = "1kb",
    mode: str = "fast",
    batch_size: int = 4,
    chrombert_cache_dir: Optional[str] = None,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    chrombert_gene_meta: Optional[str] = None,
    ft_ckpt: Optional[str] = None,
    cell_type_bw: Optional[str] = None,
    cell_type_peak: Optional[str] = None,
    return_embeddings: bool = True,
    # ignore_regulator: Optional[str] = None,
    # gep: bool = False,
    # flank_window: int = 4,
) -> Tuple[
    Optional[np.ndarray],
    Optional[pd.DataFrame],
    Optional[Dict[str, np.ndarray]],
]:
    """
    Compute ChromBERT region embeddings for user regions and/or genes (), using either
    the general pretrained model or a cell-specific model.

    Args:
        region:
            Path to a BED file of regions of interest. Embeddings are produced for
            ChromBERT bins overlapping these intervals. Use ``None`` if you only need
            gene embeddings; at least one of ``region`` or ``gene`` must be set.
        gene:
            Gene symbols or IDs, multiple entries separated by ``;`` (same as the CLI).
            Promoter-neighbourhood bins are mapped and pooled to one vector per gene.
            Use ``None`` if you only need region embeddings.
        odir:
            Output directory. Writes intermediates and results such as
            ``model_input.tsv``, ``overlap_region.bed``, ``region_emb_{oname}.npy``,
            and ``gene_emb_{oname}.pkl``.
        oname:
            Suffix tag for output filenames (e.g. ``region_emb_{oname}.npy``), useful
            for multiple runs in the same ``odir``.
        genome:
            Reference assembly matching pretrained data: ``hg38`` or ``mm10``
            (case-insensitive; lowercased internally).
        resolution:
            ChromBERT bin resolution: ``1kb``, ``200bp``, ``2kb``, or ``4kb``. Must
            match the cached HDF5 and config for the chosen genome.
        mode:
            Only used when fine-tuning a cell-specific model from BigWig + peaks:
            ``fast`` or ``full``, same semantics as the CLI. If ``ft_ckpt`` is
            provided, training is skipped and this argument has little effect.
        batch_size:
            DataLoader batch size for forward passes; reduce if GPU memory is tight.
        chrombert_cache_dir:
            Root directory for ChromBERT assets (``config/``, ``checkpoint/``, HDF5,
            etc.). If ``None``, uses ``~/.cache/chrombert/data``. Missing trees can
            trigger download logic inside ``resolve_paths``.
        chrombert_region_file:
            Override path to the ChromBERT reference region BED. ``None`` uses the
            file implied by genome and resolution under the cache.
        chrombert_region_emb_file:
            Override path to the precomputed global region embedding ``.npy``. If
            present on disk, embeddings can be sliced by index instead of running
            the model for regions.
        chrombert_gene_meta:
            Override path to the gene metadata TSV used for gene-to-region mapping.
            ``None`` uses the cache file for the genome/resolution pair.
        ft_ckpt:
            Path to a fine-tuned cell-specific checkpoint. When set, runs
            cell-specific embedding without training from BigWig/peaks. Cell-specific
            mode is also enabled by supplying both ``cell_type_bw`` and
            ``cell_type_peak`` (see ``validate_args``).
        cell_type_bw:
            Cell-type chromatin accessibility BigWig. Together with
            ``cell_type_peak``, enables cell-specific mode when ``ft_ckpt`` is absent.
        cell_type_peak:
            Peak BED for the same cell type as ``cell_type_bw``; must be used as a
            pair when not using ``ft_ckpt``.
        return_embeddings:
            If ``True``, return in-memory arrays/dicts in addition to writing files.
            If ``False``, only write outputs; all three return values are ``None``.

    Returns:
        Tuple ``(region_emb, overlap_region_bed, gene_emb_dict)``:

        - **region_emb**: Matrix of embeddings for overlapping ChromBERT bins, shape
          about ``(n_overlap, 768)``. ``None`` if ``region`` was not requested or
          ``return_embeddings`` is ``False``.
        - **overlap_region_bed**: DataFrame of overlapping intervals with
          ``build_region_index`` and related columns. ``None`` if ``region`` was not
          requested or ``return_embeddings`` is ``False``.
        - **gene_emb_dict**: Mapping from lowercased gene keys to 1D vectors (~768),
          mean-pooled over bins. ``None`` if ``gene`` was not requested or
          ``return_embeddings`` is ``False``.
    """
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")

    args = SimpleNamespace(
        region=region,
        gene=gene,
        cell_type_bw=cell_type_bw,
        cell_type_peak=cell_type_peak,
        ft_ckpt=ft_ckpt,
        odir=odir,
        oname=oname,
        genome=genome.lower() if isinstance(genome, str) else genome,
        resolution=resolution,
        mode=mode,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
        chrombert_gene_meta=chrombert_gene_meta,
        # ignore_regulator=ignore_regulator,
        # gep=gep,
        # flank_window=flank_window,
    )

    validate_args(args)
    os.makedirs(odir, exist_ok=True)
    files_dict = resolve_paths(args)
    check_files(files_dict, required_keys=get_required_keys(args))

    region_emb: Optional[np.ndarray] = None
    region_bed: Optional[pd.DataFrame] = None
    gene_emb_dict: Optional[Dict[str, np.ndarray]] = None

    cell_mode = is_cell_specific(args)
    if cell_mode:
        model_emb, _ = build_cell_model_emb(args, files_dict, odir)
        if region is not None:
            pair = run_region_cell(
                args,
                files_dict,
                odir,
                return_data=return_embeddings,
                model_emb=model_emb,
            )
            if return_embeddings and pair is not None:
                region_emb, region_bed = pair
        if gene is not None:
            gout = run_gene_cell(
                args,
                files_dict,
                odir,
                return_data=return_embeddings,
                model_emb=model_emb,
            )
            if return_embeddings and gout is not None:
                gene_emb_dict = gout
    else:
        if region is not None:
            pair = run_region_general(
                args,
                files_dict,
                odir,
                return_data=return_embeddings,
            )
            if return_embeddings and pair is not None:
                region_emb, region_bed = pair
        if gene is not None:
            gout = run_gene_general(
                args,
                files_dict,
                odir,
                return_data=return_embeddings,
            )
            if return_embeddings and gout is not None:
                gene_emb_dict = gout

    return region_emb, region_bed, gene_emb_dict
