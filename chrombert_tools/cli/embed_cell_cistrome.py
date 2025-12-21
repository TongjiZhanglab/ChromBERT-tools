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
from chrombert.scripts.utils import HDF5Manager
from .utils import resolve_paths, check_files, check_region_file, chrom_to_int_series, overlap_cistrome_func
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
        "meta_file",
        "hdf5_file",
        "pretrain_ckpt",
        "mtx_mask"
    ])

    # ---------- Stage 1: Prepare regions ----------
    print("Stage 1: Preparing regions")
    focus_region = args.region
    overlap_bed = check_region_file(focus_region,files_dict,args.odir)

    
    # Chromosome mapping to integer
    first_chrom = str(overlap_bed["chrom"].iloc[0])
    if "chr" in first_chrom.lower():
        overlap_bed["chrom"] = chrom_to_int_series(overlap_bed["chrom"].astype(str), args.genome)

    overlap_bed = overlap_bed.dropna(subset=["chrom"]).copy()
    overlap_bed["chrom"] = overlap_bed["chrom"].astype(int)
    overlap_bed.to_csv(f"{args.odir}/model_input.tsv", sep="\t", index=False)
    
    # Overlap cistrome (meta)
    _, _, cistrome_gsmid_dict = overlap_cistrome_func(args.cistrome, files_dict["meta_file"])
    if len(cistrome_gsmid_dict) == 0:
        raise ValueError("No requested cistromes matched ChromBERT meta. Nothing to embed.")
    
    print(f"Found {len(cistrome_gsmid_dict)} cistromes: {list(cistrome_gsmid_dict.keys())}")
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
        model_tuned, train_odir, model_config, data_config = retry_train(
            args, files_dict, cal_metrics_regression, metcic='pearsonr', min_threshold=0.4
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

    # ---------- Stage 3: Compute cell-specific cistrome embeddings ----------
    print("Stage 3: Computing cell-specific cistrome embeddings")
    
    # Get embedding manager from cell-specific model
    model_emb = model_embedding(
        train_odir=train_odir, 
        model_config=model_config, 
        ft_ckpt=args.ft_ckpt, 
        model_tuned=model_tuned
    )
    
    # Prepare dataloader
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=f"{args.odir}/model_input.tsv",
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()

    # Save cistrome embedding on specific region
    shapes = {f"emb/{k}": [(len(ds), 768), np.float16] for k in cistrome_gsmid_dict}
    total_counts = 0
    cistrome_sums = {name: np.zeros(768, dtype=np.float64) for name in cistrome_gsmid_dict}
    
    out_h5 = f"{args.odir}/cell_specific_cistrome_emb_on_region.hdf5"
    with HDF5Manager(out_h5, region=[(len(ds), 4), np.int64], **shapes) as h5:
        with torch.no_grad():
            for batch in tqdm(dl, total=len(dl), desc="Computing cistrome embeddings"):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()

                model_emb(batch)  # initialize cache

                bs = batch["region"].shape[0]
                start_idx = total_counts
                total_counts += bs
                end_idx = total_counts

                batch_index = batch["build_region_index"].long().cpu().numpy().reshape(-1)
                region = overlap_bed.iloc[start_idx:end_idx].values
                assert (batch_index == region[:, -1].reshape(-1)).all(), \
                    "Batch index and region index do not match"

                # Get cistrome embeddings from cell-specific model
                embs = {
                    f"emb/{k}": model_emb.get_cistrome_embedding(v).float().cpu().numpy()
                    for k, v in cistrome_gsmid_dict.items()
                }
                h5.insert(region=region, **embs)
                
                # Accumulate for mean calculation
                for reg_name, reg_idx in cistrome_gsmid_dict.items():
                    emb = model_emb.get_cistrome_embedding(reg_idx)
                    emb_np = emb.float().cpu().numpy()
                    cistrome_sums[reg_name] += emb_np.sum(axis=0)

    # Calculate mean embeddings
    cistrome_means = {
        reg_name: (sum_vec / total_counts)
        for reg_name, sum_vec in cistrome_sums.items()
    }
    
    # Save mean embeddings
    out_pkl = os.path.join(args.odir, "cell_specific_mean_cistrome_emb.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(cistrome_means, f)
    
    print("Finished stage 3")

    # ---------- Report ----------
    print("\nFinished all stages!")
    if train_odir is not None:
        print(f"Cell-specific ChromBERT model saved to: {train_odir}")
    print(f"Cell-specific mean cistrome embeddings saved to: {out_pkl}")
    print(f"Cell-specific cistrome embeddings on regions saved to: {out_h5}")
    print(f"Number of cistromes embedded: {len(cistrome_gsmid_dict)}")


@click.command(name="embed_cell_cistrome", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, 
              help="Region file where cistrome embeddings will be computed.")
@click.option("--cistrome", required=True,
              help="GSM/ENCODE id or factor:cell, e.g. ENCSR... or GSM... or ATAC-seq:HEK293T or BCL11A:GM12878. Use ';' to separate multiple.")
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
@click.option("--num-workers", "num_workers", default=8, show_default=True, type=int,
              help="Dataloader workers.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default=os.path.expanduser("~/.cache/chrombert/data"),
              show_default=True,
              type=click.Path(file_okay=False),
              help="ChromBERT cache dir (contains config/ checkpoint/ etc).")


def embed_cell_cistrome(region, cistrome, cell_type_bw, cell_type_peak, ft_ckpt, 
                        odir, genome, resolution, mode, batch_size, num_workers, 
                        chrombert_cache_dir):
    '''
    Extract cell-specific cistrome embeddings on specified regions
    '''
    args = SimpleNamespace(
        region=region,
        cistrome=cistrome,
        cell_type_bw=cell_type_bw,
        cell_type_peak=cell_type_peak,
        ft_ckpt=ft_ckpt,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        mode=mode,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    embed_cell_cistrome()

