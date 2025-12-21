import os
import click
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
from .utils import resolve_paths, check_files, check_region_file
from .utils_train_cell import make_dataset, retry_train
from .utils import cal_metrics_regression, model_embedding


def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    # Validate input arguments
    if args.ft_ckpt is None and (args.cell_type_bw is None or args.cell_type_peak is None):
        raise ValueError(
            "If you do not provide --ft-ckpt, you should provide --cell-type-bw and "
            "--cell-type-peak to train a cell-specific model."
        )

    files_dict = resolve_paths(args)
    check_files(files_dict, required_keys=[
        "chrombert_region_file",
        "hdf5_file",
        "pretrain_ckpt",
        "mtx_mask"
    ])

    # ---------- Stage 1: Overlap focus regions with ChromBERT regions ----------
    print("Stage 1: Overlapping focus regions with ChromBERT regions")
    focus_region = args.region
    overlap_bed = check_region_file(focus_region,files_dict,odir)
    print(f"Found {overlap_bed.shape[0]} overlapping regions")
    print("Finished stage 1")

    # ---------- Stage 2: Prepare cell-specific model ----------
    if args.cell_type_bw is not None and args.cell_type_peak is not None and args.ft_ckpt is None:
        d_odir = f"{odir}/dataset"
        os.makedirs(d_odir, exist_ok=True)
        train_odir = f"{odir}/train"
        os.makedirs(train_odir, exist_ok=True)
        
        print("Stage 2a: Preparing the dataset for cell-specific model")
        make_dataset(args.cell_type_peak, args.cell_type_bw, d_odir, files_dict)
        print("Finished stage 2a")
        
        print("Stage 2b: Fine-tuning the model for cell-specific embeddings")
        model_tuned, train_odir, model_config, data_config = retry_train(args, files_dict, 
            cal_metrics_regression, metcic='pearsonr', min_threshold=0.4
        )
        print("Finished stage 2b: Got a cell-specific ChromBERT model")
    elif args.ft_ckpt is not None:
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
        raise ValueError("Should not reach here due to earlier validation")

    # ---------- Stage 3: Compute cell-specific region embeddings ----------
    print("Stage 3: Computing cell-specific region embeddings")
    
    # Get embedding manager from cell-specific model
    model_emb = model_embedding(
        train_odir=train_odir, 
        model_config=model_config, 
        ft_ckpt=args.ft_ckpt, 
        model_tuned=model_tuned
    )
    
    # Prepare dataloader for focus regions
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=f"{odir}/model_input.tsv",
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=8,
    )
    dl = data_config.init_dataloader()

    # Compute region embeddings using cell-specific model
    region_embs = []
    with torch.no_grad():
        for batch in tqdm(dl, total=len(dl), desc="Computing region embeddings"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            model_emb(batch)  # initialize cache
            region_embs.append(model_emb.get_region_embedding().float().cpu().detach())

    region_embs = torch.cat(region_embs, dim=0).numpy()  # (len(overlap_bed), 768)
    
    # Save embeddings
    np.save(f"{odir}/cell_specific_overlap_region_emb.npy", region_embs)
    
    print("Finished stage 3")

    # ---------- Report ----------
    total_focus = sum(1 for _ in open(focus_region))
    no_overlap_region_len = sum(1 for _ in open(f"{odir}/no_overlap_region.bed"))
    
    print("\nFinished all stages!")
    print(
        f"Focus region summary - total: {total_focus}, "
        f"overlapping with ChromBERT: {overlap_bed.shape[0]}, "
        f"non-overlapping: {no_overlap_region_len}"
    )
    print("Note: It is possible for a single region to overlap multiple ChromBERT regions.")
    if train_odir is not None:
        print(f"Cell-specific ChromBERT model saved to: {train_odir}")
    print(f"Overlapping regions BED file: {odir}/overlap_region.bed")
    print(f"Non-overlapping regions BED file: {odir}/no_overlap_region.bed")
    print(f"Cell-specific region embeddings saved to: {odir}/cell_specific_overlap_region_emb.npy")


@click.command(name="embed_cell_region", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, 
              help="Region file to compute embeddings for.")
@click.option("--cell-type-bw", "cell_type_bw",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, 
              help="Cell type accessibility BigWig file. Required if --ft-ckpt is not provided.")
@click.option("--cell-type-peak", "cell_type_peak",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, 
              help="Cell type accessibility Peak BED file. Required if --ft-ckpt is not provided.")
@click.option("--ft-ckpt", "ft_ckpt",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, default=None, show_default=True,
              help="Fine-tuned ChromBERT checkpoint. If provided, skip fine-tuning and use this ckpt. "
                   "If not provided, you must provide --cell-type-bw and --cell-type-peak to train a cell-specific model.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), 
              help="Output directory.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), 
              help="Genome.")
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


def embed_cell_region(region, cell_type_bw, cell_type_peak, ft_ckpt, 
                      odir, genome, resolution, mode, batch_size, 
                      chrombert_cache_dir, chrombert_region_file):
    '''
    Extract cell-specific region embeddings
    '''
    args = SimpleNamespace(
        region=region,
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
    )
    run(args)


if __name__ == "__main__":
    embed_cell_region()

