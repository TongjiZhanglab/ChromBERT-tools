import os
import click
from types import SimpleNamespace

import pandas as pd
import numpy as np
from .embed_region import embed_region_processed
from .utils import resolve_paths, check_files, check_region_file, chrom_to_int_series
from .utils_interpret import (
    build_interpret_config,
    load_interpret_model,
    cal_sim_tss_region_pairs,
    cross_region_set_cosine_pairs,
    
)




def _normalize_chrom_int(df: pd.DataFrame, genome: str, col: str = "chrom") -> pd.DataFrame:
    """Map chrom names (chr1 / 1 / int) to ChromBERT integer codes so BED and gene meta align."""
    out = df.copy()
    if len(out) == 0:
        return out
    out[col] = chrom_to_int_series(out[col].astype(str), genome)
    out = out.dropna(subset=[col]).copy()
    out[col] = out[col].astype(int)
    return out

def get_union_embeddings(args, files_dict, sup_file, union_idx):
    odir = args.odir
    oname = getattr(args, "oname", "region_emb")
    chrombert_region_emb_file = files_dict["region_emb_npy"]
    if args.ft_ckpt is None:
        if chrombert_region_emb_file is not None:
            region_embs, _ = embed_region_processed(
                emb_npy_file=chrombert_region_emb_file,
                overlap_idx=union_idx,
                odir=odir,
                oname=oname,
            )
        else:
            data_config, model_config = build_interpret_config(args, files_dict, sup_file)
            dl = data_config.init_dataloader(supervised_file=sup_file)
            _, model_emb = load_interpret_model(model_config)
            region_embs, _ = embed_region_processed(
                dl=dl, model_emb=model_emb, odir=odir, oname=oname
            )
    else:
        data_config, model_config = build_interpret_config(args, files_dict, sup_file)
        dl = data_config.init_dataloader(supervised_file=sup_file)
        _, model_emb = load_interpret_model(model_config)
        region_embs, _ = embed_region_processed(
            dl=dl, model_emb=model_emb, odir=odir, oname=oname
        )

    return region_embs

def run(args, return_data=False):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    region2 = getattr(args, "region2", None)

    if region2 is None:
        # ----- Enhancer–promoter (single region set + gene TSS) -----
        check_files(
            files_dict,
            required_keys=[
                "chrombert_region_file",
                "hdf5_file",
                "gene_meta_tsv",
            ],
        )
        overlap_bed = check_region_file(args.region, files_dict, odir)
        overlap_bed = _normalize_chrom_int(overlap_bed, args.genome)

        gene_tss = pd.read_csv(files_dict["gene_meta_tsv"], sep="\t")
        gene_tss = gene_tss[
            ["chrom", "start", "end", "build_region_index", "tss", "gene_name", "gene_id"]
        ]
        gene_tss = _normalize_chrom_int(gene_tss, args.genome)
        gene_tss = gene_tss.sort_values(by="build_region_index").reset_index(drop=True)

        model_input = (
            pd.concat(
                [overlap_bed, gene_tss[["chrom", "start", "end", "build_region_index"]]]
            )
            .drop_duplicates(subset="build_region_index")
            .sort_values(by="build_region_index")
            .reset_index(drop=True)
        )
        sup_file = os.path.join(odir, "model_input.tsv")
        model_input.to_csv(sup_file, sep="\t", index=False)
        union_idx = model_input["build_region_index"].to_numpy(dtype=np.int64)
        region_embs = get_union_embeddings(args, files_dict, sup_file, union_idx)

        window = getattr(args, "distance_window", 250_000)
        pairs_cos = cal_sim_tss_region_pairs(
            overlap_bed, gene_tss, union_idx, region_embs, window=window
        )
        out_tsv = os.path.join(odir, "tss_region_pairs_cos.tsv")
        pairs_cos.to_csv(out_tsv, sep="\t", index=False)
        print("Finished!")
        print(f"Enhancer–promoter style pairs saved to: {out_tsv}")
        if return_data:
            return pairs_cos
        return None

    # ----- Two region sets: cosine similarity within genomic distance window -----
    check_files(
        files_dict,
        required_keys=[
            "chrombert_region_file",
            "hdf5_file",
            # "pretrain_ckpt",
        ],
    )
    d_odir = os.path.join(odir, "dataset")
    d1 = os.path.join(d_odir, "region1")
    d2 = os.path.join(d_odir, "region2")
    os.makedirs(d1, exist_ok=True)
    os.makedirs(d2, exist_ok=True)

    overlap1 = check_region_file(args.region, files_dict, d1)
    overlap2 = check_region_file(region2, files_dict, d2)
    overlap1 = _normalize_chrom_int(overlap1, args.genome)
    overlap2 = _normalize_chrom_int(overlap2, args.genome)

    model_input = (
        pd.concat(
            [
                overlap1[["chrom", "start", "end", "build_region_index"]],
                overlap2[["chrom", "start", "end", "build_region_index"]],
            ]
        )
        .drop_duplicates(subset="build_region_index")
        .sort_values(by="build_region_index")
        .reset_index(drop=True)
    )
    sup_file = os.path.join(odir, "model_input.tsv")
    model_input.to_csv(sup_file, sep="\t", index=False)
    union_idx = model_input["build_region_index"].to_numpy(dtype=np.int64)
    region_embs = get_union_embeddings(args, files_dict, sup_file, union_idx)

    window = getattr(args, "distance_window", 250_000)
    pairs_cos = cross_region_set_cosine_pairs(
        overlap1,
        overlap2,
        union_idx,
        region_embs,
        max_genomic_dist_bp=window,
    )
    out_tsv = os.path.join(odir, "region_set_pairs_cos.tsv")
    pairs_cos.to_csv(out_tsv, sep="\t", index=False)
    print("Finished!")
    print(
        f"Set1 x set2 region-pair cosines (same chrom, genomic_dist_bp <= {window}) saved to: {out_tsv}"
    )
    if return_data:
        return pairs_cos
    return None


@click.command(
    name="interpret_region_region_interactions",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--region",
    "region",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Region BED file (set 1, e.g. candidate enhancer regions).",
)
@click.option(
    "--region2",
    "region2",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Optional second BED. If omitted: infer region1–promoter interaction pairs "
    "If set: same-chromosome pairs (region2 regions vs region1 regions) only ",
)
@click.option("--odir", default="./output", show_default=True, type=click.Path(file_okay=False))
@click.option(
    "--genome",
    default="hg38",
    show_default=True,
    type=click.Choice(["hg38", "mm10"], case_sensitive=False),
)
@click.option(
    "--resolution",
    default="1kb",
    show_default=True,
    type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False),
)
@click.option(
    "--distance-window",
    "distance_window",
    default=250_000,
    show_default=True,
    type=int,
    help="Max distance (bp).  "
    "keep pairs with interval gap <= this (0 if overlap); cross-chrom dropped.",
)
@click.option("--batch-size", "batch_size", default=4, show_default=True, type=int)
@click.option(
    "--ft-ckpt",
    "ft_ckpt",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Optional fine-tuned checkpoint (same as interpret_regulator_interactions).",
)
@click.option(
    "--ignore-regulator",
    "ignore_regulator",
    type=str,
    default=None,
    help="Optional; passed to build_interpret_config when computing embeddings from model.",
)
@click.option("--gep", "gep", is_flag=True, default=False, show_default=True,
              help="Use GEP model (multi-flank-window). Default: False.")
@click.option("--flank-window", "flank_window",
              type=int,
              required=False, default=4, show_default=True,
              help="Flank window size for gep model.")
@click.option(
    "--chrombert-cache-dir",
    "chrombert_cache_dir",
    default="~/.cache/chrombert/data",
    show_default=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "--chrombert-region-file",
    "chrombert_region_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--chrombert-region-emb-file",
    "chrombert_region_emb_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)

def interpret_region_region_interactions(
    region,
    region2,
    odir,
    genome,
    resolution,
    chrombert_cache_dir,
    chrombert_region_file,
    chrombert_region_emb_file,
    batch_size,
    ft_ckpt,
    ignore_regulator,
    gep,
    flank_window,
    distance_window,
):
    """Region embedding similarities: enhancer–promoter (one BED) or two BEDs within --distance-window."""
    args = SimpleNamespace(
        region=region,
        region2=region2,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
        batch_size=batch_size,
        ft_ckpt=ft_ckpt,
        ignore_regulator=ignore_regulator,
        gep=gep,
        flank_window=flank_window,
        distance_window=distance_window,
    )
    run(args)


if __name__ == "__main__":
    interpret_region_region_interactions()
