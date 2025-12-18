import os
import click
import pickle
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
import torch
from tqdm import tqdm

import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
from .utils import resolve_paths, check_files
from .utils_train_cell import make_dataset, model_train
from .utils import model_eval, cal_metrics_regression, model_embedding
from .utils import overlap_gene_map_region


def _parse_focus_genes(gene_str: str):
    """Parse user-provided genes; separated by ';'."""
    focus = [g.strip().lower() for g in gene_str.split(";") if g.strip()]
    return focus

def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    if args.ft_ckpt is None and (args.cell_type_bw is None or args.cell_type_peak is None):
        raise ValueError("If you not provide --ft-ckpt, you should provide --cell-type-bw and --cell-type-peak to train a cell-specific model.")
    files_dict = resolve_paths(args)
    # Only check files needed by embed_cell_gene
    check_files(files_dict, required_keys=[
        "chrombert_region_file",
        "hdf5_file",
        "pretrain_ckpt",
        "mtx_mask"
    ])

    # ---------- load gene meta ----------
    gene_meta_tsv = files_dict["gene_meta_tsv"]
    if not os.path.exists(gene_meta_tsv):
        raise FileNotFoundError(
            f"Gene meta file not found: {gene_meta_tsv}\n"
            "Please run `chrombert_prepare_env` or provide --chrombert-gene-meta."
        )
    
    gene_meta = pd.read_csv(gene_meta_tsv, sep="\t")
    gene_meta["gene_id"] = gene_meta["gene_id"].astype(str).str.lower()
    gene_meta["gene_name"] = gene_meta["gene_name"].astype(str).str.lower()

    # ---------- parse requested genes ----------
    focus_genes = _parse_focus_genes(args.gene)

    overlap_genes, not_found_genes, gene_to_region_idx = overlap_gene_map_region(gene_meta, focus_genes, odir)

    # ---------- prepare cell-specific model ----------
    if args.cell_type_bw is not None and args.cell_type_peak is not None and args.ft_ckpt is None:
        d_odir = f"{odir}/dataset"
        os.makedirs(d_odir, exist_ok=True)
        train_odir = f"{odir}/train"
        os.makedirs(train_odir, exist_ok=True)
        print("Stage 1: Preparing the dataset for cell-specific model")
        make_dataset(args.cell_type_peak, args.cell_type_bw, d_odir, files_dict)
        print("Finished stage 1")
    else:
        print("Finished stage 1")

    # 2) train or load fine-tuned model
    if args.ft_ckpt is not None:
        print(f"Stage 2: Using provided fine-tuned ChromBERT checkpoint: {args.ft_ckpt}")
        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task="general",
            dropout=0,
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            finetune_ckpt=args.ft_ckpt,
        )
        model_tuned = model_config.init_model()
        train_odir = None
        print("Finished stage 2")
    else:
        print("Stage 2: Fine-tuning the model for cell-specific embeddings")
        data_module, model_config = model_train(d_odir, train_odir, args, files_dict)
        print("Finished stage 2: Got a cell-specific ChromBERT model")
        print("Evaluating the fine-tuned model performance")
        model_tuned = model_eval(args, train_odir, data_module, model_config, cal_metrics_regression)

    # ---------- compute gene embeddings using cell-specific model ----------
    print("Stage 3: Computing cell-specific gene embeddings")
    
    # Get needed region indices
    needed_idx = np.unique(np.concatenate([np.asarray(v).astype(int).reshape(-1) for v in gene_to_region_idx.values()]))

    # Load ChromBERT region bed and subset required indices
    chrombert_region_bed = files_dict["chrombert_region_file"]
    region_df = pd.read_csv(
        chrombert_region_bed,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "build_region_index"],
    )
    sub_df = region_df[region_df["build_region_index"].isin(needed_idx)].copy()
    sub_df = sub_df.sort_values("build_region_index").reset_index(drop=True)

    if sub_df.shape[0] == 0:
        raise ValueError("No required build_region_index found in chrombert_region_file. Check gene_meta vs region.bed.")

    # Save model input
    sub_df.to_csv(f"{odir}/model_input.csv", index=False)

    # Get embedding manager from fine-tuned model
    model_emb = model_embedding(train_odir=train_odir, model_config=model_config, ft_ckpt=args.ft_ckpt, model_tuned=model_tuned)

    # Prepare dataloader for gene regions
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=f"{odir}/model_input.csv",
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=8,
    )
    dl = data_config.init_dataloader()

    # Compute region embeddings
    region_embs = []
    with torch.no_grad():
        for batch in tqdm(dl, total=len(dl), desc="Computing region embeddings"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            model_emb(batch)  # initialize cache
            region_embs.append(model_emb.get_region_embedding().float().cpu().detach())

    region_embs = torch.cat(region_embs, dim=0).numpy()  # (len(sub_df), 768)

    # build index -> embedding mapping
    idx2emb = {int(i): region_embs[j] for j, i in enumerate(sub_df["build_region_index"].tolist())}

    # compute gene embedding by mean pooling its TSS-related region embeddings
    gene_emb_dict = {}
    for g, idxs in gene_to_region_idx.items():
        idxs = np.asarray(idxs).astype(int).reshape(-1)
        embs = [idx2emb[i] for i in idxs if i in idx2emb]
        if len(embs) == 0:
            raise ValueError(f"No region embeddings found for gene={g}.")
        gene_emb_dict[g] = np.mean(np.stack(embs, axis=0), axis=0)

    print("Finished stage 3")

    # ---------- save outputs ----------
    with open(f"{odir}/cell_specific_gene_embs_dict.pkl", "wb") as f:
        pickle.dump(gene_emb_dict, f)

    # ---------- report ----------
    print("\nFinished all stages!")
    print("Note: All gene names were converted to lowercase for matching.")
    print(
        f"Gene count summary - requested: {len(focus_genes)}, "
        f"matched: {len(overlap_genes)}, "
        f"not found: {len(not_found_genes)}"
    )
    print(f"Cell-specific ChromBERT model saved to: {train_odir}")
    print(f"Cell-specific gene embeddings saved to: {odir}/cell_specific_gene_embs_dict.pkl")
    print(f"Matched gene meta saved to: {odir}/overlap_genes_meta.tsv")


@click.command(name="embed_cell_gene", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--gene", "gene",
              type=str,
              required=True,
              help="Gene symbols or IDs. e.g. ENSG00000170921;TANC2;DPYD. Use ';' to separate multiple genes.")
@click.option("--cell-type-bw", "cell_type_bw",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, help="Cell type accessibility BigWig file.")
@click.option("--cell-type-peak", "cell_type_peak",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, help="Cell type accessibility Peak BED file.")
@click.option("--ft-ckpt", "ft_ckpt",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, default=None, show_default=True,
              help="Fine-tuned ChromBERT checkpoint. If provided, skip fine-tuning and use this ckpt. If you not provide, you should provide --cell-type-bw and --cell-type-peak to train a cell-specific model.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False),
              help="Resolution. Mouse only supports 1kb resolution.")
@click.option("--mode", default="fast", show_default=True,
              type=click.Choice(["fast", "full"], case_sensitive=False),
              help="Fast: downsample regions to 20k for training; Full: use all regions.")
@click.option("--batch-size", "batch_size", default=4, show_default=True, type=int,
              help="Batch size.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default=os.path.expanduser("~/.cache/chrombert/data"),
              show_default=True,
              type=click.Path(file_okay=False),
              help="ChromBERT cache dir. If you use `chrombert_prepare_env`, you usually don't need to provide this.")
@click.option("--chrombert-region-file", "chrombert_region_file",
              default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="ChromBERT region BED file. If not provided, use the default {genome}_{nd}_{resolution}_region.bed in the cache dir.")
@click.option("--chrombert-gene-meta", "chrombert_gene_meta",
              default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="ChromBERT gene meta TSV. If not provided, try {genome}_{resolution}_gene_meta.tsv in the cache dir.")


def embed_cell_gene(gene, cell_type_bw, cell_type_peak, ft_ckpt, odir, genome, resolution,
                    mode, batch_size, chrombert_cache_dir, chrombert_region_file, chrombert_gene_meta):

    args = SimpleNamespace(
        gene=gene,
        cell_type_bw=cell_type_bw,
        cell_type_peak=cell_type_peak,
        ft_ckpt=ft_ckpt,
        odir=odir,
        genome=genome,
        resolution=resolution,
        mode=mode,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_gene_meta=chrombert_gene_meta,
    )
    run(args)


if __name__ == "__main__":
    embed_cell_gene()

