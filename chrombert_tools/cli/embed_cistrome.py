import os
import re
import json
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

def run(args, return_data=False):
    os.makedirs(args.odir, exist_ok=True)

    files_dict = resolve_paths(args)
    # Only check files needed by embed_cistrome
    check_files(files_dict, required_keys=[
        "chrombert_region_file",
        "meta_file",
        "hdf5_file",
        "pretrain_ckpt",
        "mtx_mask"
    ])

    # overlap region
    focus_region = args.region
    overlap_bed = check_region_file(focus_region,files_dict,args.odir)

    # chromosome mapping to integer
    first_chrom = str(overlap_bed["chrom"].iloc[0])
    if "chr" in first_chrom.lower():
        overlap_bed["chrom"] = chrom_to_int_series(overlap_bed["chrom"].astype(str), args.genome)
    overlap_bed = overlap_bed.dropna(subset=["chrom"]).copy()
    overlap_bed["chrom"] = overlap_bed["chrom"].astype(int)
    overlap_bed.to_csv(f"{args.odir}/model_input.tsv", sep="\t", index=False)

    # overlap cistrome (meta)
    _, _, cistrome_gsmid_dict = overlap_cistrome_func(args.cistrome, files_dict["meta_file"])
    if len(cistrome_gsmid_dict) == 0:
        raise ValueError("No requested cistromes matched ChromBERT meta. Nothing to embed.")

    # dataloader
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=f"{args.odir}/model_input.tsv",
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()

    # model
    model_config = ChromBERTFTConfig(
        genome=args.genome,
        dropout=0,
        task="general",
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        mtx_mask=files_dict["mtx_mask"],
    )
    model = model_config.init_model().get_embedding_manager().cuda().bfloat16()

    # save cistrome embedding on specific region
    shapes = {f"emb/{k}": [(len(ds), 768), np.float16] for k in cistrome_gsmid_dict}
    total_counts = 0
    cistrome_sums = {name: np.zeros(768, dtype=np.float64) for name in cistrome_gsmid_dict}
    
    # Store embeddings if needed for return_data
    cistrome_emb_dict = {}

    out_h5 = f"{args.odir}/cistrome_emb_on_region.hdf5"
    with HDF5Manager(out_h5, region=[(len(ds), 4), np.int64], **shapes) as h5:
        with torch.no_grad():
            for batch in tqdm(dl, total=len(dl)):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()

                model(batch)  # init cache

                bs = batch["region"].shape[0]
                start_idx = total_counts
                total_counts += bs
                end_idx = total_counts

                batch_index = batch["build_region_index"].long().cpu().numpy().reshape(-1)
                region = overlap_bed.iloc[start_idx:end_idx].values
                assert (batch_index == region[:, -1].reshape(-1)).all(), "Batch index and region index do not match"

                embs = {
                    f"emb/{k}": model.get_cistrome_embedding(v).float().cpu().numpy()
                    for k, v in cistrome_gsmid_dict.items()
                }
                h5.insert(region=region, **embs)
                
                # Store for return if needed
                if return_data:
                    for k, v in embs.items():
                        cis_name = k.replace("emb/", "")
                        if cis_name not in cistrome_emb_dict:
                            cistrome_emb_dict[cis_name] = []
                        cistrome_emb_dict[cis_name].append(v)
                
                for reg_name, reg_idx in cistrome_gsmid_dict.items():
                    emb = model.get_cistrome_embedding(reg_idx)
                    emb_np = emb.float().cpu().numpy()            
                    cistrome_sums[reg_name] += emb_np.sum(axis=0)
    
    # Concatenate collected data if return_data
    if return_data:
        for k in cistrome_emb_dict:
            cistrome_emb_dict[k] = np.concatenate(cistrome_emb_dict[k], axis=0)
            
    cistrome_means = {
        reg_name: (sum_vec / total_counts)
        for reg_name, sum_vec in cistrome_sums.items()
    }
    out_pkl = os.path.join(args.odir, "mean_cistrome_emb.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(cistrome_means, f)
    print("Finished!")
    print("Saved mean cistrome embeddings to pickle file:", out_pkl)
    print("Saved cistrome embeddings to hdf5 file:", out_h5)
    
    if return_data:
        return cistrome_means, cistrome_emb_dict, overlap_bed
    return None

@click.command(name="embed_cistrome", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region file.")
@click.option("--cistrome", required=True,
              help="GSM/ENCODE id or factor:cell, e.g. ENCSR... or GSM... or ATAC-seq:HEK293T or BCL11A:GM12878. Use ';' to separate multiple.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False), help="Resolution.")
@click.option("--batch-size", default=64, show_default=True, type=int, help="Batch size.")
@click.option("--num-workers", default=8, show_default=True, type=int, help="Dataloader workers.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir", default=os.path.expanduser("~/.cache/chrombert/data"),
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache dir (contains config/ checkpoint/ etc).")

def embed_cistrome(region, cistrome, odir, genome, resolution, batch_size, num_workers,
        chrombert_cache_dir):
    '''
    Extract general cistrome embeddings on specified regions
    '''

    args = SimpleNamespace(
        region=region,
        cistrome=cistrome,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    embed_cistrome()
