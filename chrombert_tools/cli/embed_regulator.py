import os
import re
import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
from chrombert.scripts.utils import HDF5Manager
import pickle
from .utils import resolve_paths, check_files, check_region_file, chrom_to_int_series, overlap_regulator_func


def run(args, return_data=False):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    # mm10 resolution limit (only have 1kb data)
    if args.genome.lower() == "mm10" and args.resolution != "1kb":
        raise ValueError("mm10 currently only supports 1kb in this cache layout (adjust if you have more).")

    files_dict = resolve_paths(args)
    # Only check files needed by embed_regulator
    check_files(files_dict, required_keys=[
        "chrombert_region_file",
        "chrombert_regulator_file", 
        "hdf5_file",
        "pretrain_ckpt"
    ])

    focus_region = args.region
    overlap_bed = check_region_file(focus_region,files_dict,odir)

    # chromosome mapping to integer
    first_chrom = str(overlap_bed["chrom"].iloc[0])
    if "chr" in first_chrom.lower():
        overlap_bed["chrom"] = chrom_to_int_series(overlap_bed["chrom"].astype(str), args.genome)
    overlap_bed = overlap_bed.dropna(subset=["chrom"]).copy()
    overlap_bed["chrom"] = overlap_bed["chrom"].astype(int)
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)

    _, _, regulator_idx_dict = overlap_regulator_func(args.regulator, files_dict["chrombert_regulator_file"])
    if len(regulator_idx_dict) == 0:
        raise ValueError("No requested regulators matched ChromBERT regulator list. Nothing to embed.")

    # dataloader
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=f"{odir}/model_input.tsv",
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

    # save HDF5
    shapes = {f"emb/{k}": [(len(ds), 768), np.float16] for k in regulator_idx_dict}
    total_counts = 0
    # save mean regulator emb
    reg_sums = {name: np.zeros(768, dtype=np.float64) for name in regulator_idx_dict}
    
    reg_emb_dict = {}
    with HDF5Manager(f"{odir}/regulator_emb_on_region.hdf5",
                     region=[(len(ds), 4), np.int64],
                     **shapes) as h5:
        with torch.no_grad():
            for batch in tqdm(dl, total=len(dl)):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()

                model(batch)  # init cache inside embedding_manager

                bs = batch["region"].shape[0]
                start_idx = total_counts
                total_counts += bs
                end_idx = total_counts

                batch_index = batch["build_region_index"].long().cpu().numpy().reshape(-1)
                region = overlap_bed.iloc[start_idx:end_idx].values
                assert (batch_index == region[:, -1].reshape(-1)).all(), "Batch index and region index do not match"

                embs = {
                    f"emb/{k}": model.get_regulator_embedding(k).float().cpu().numpy()
                    for k in regulator_idx_dict
                }
                h5.insert(region=region, **embs)
                
                # Store for return if needed
                if return_data:
                    for k, v in embs.items():
                        reg_name = k.replace("emb/", "")
                        if reg_name not in reg_emb_dict:
                            reg_emb_dict[reg_name] = []
                        reg_emb_dict[reg_name].append(v)
                
                for reg_name, reg_idx in regulator_idx_dict.items():
                    emb = model.get_regulator_embedding(reg_name)
                    emb_np = emb.float().cpu().numpy()            
                    reg_sums[reg_name] += emb_np.sum(axis=0)
                    
    # Concatenate collected data if return_data
    if return_data:
        for k in reg_emb_dict:
            reg_emb_dict[k] = np.concatenate(reg_emb_dict[k], axis=0)
            
    reg_means = {
        reg_name: (sum_vec / total_counts)
        for reg_name, sum_vec in reg_sums.items()
    }
    out_pkl = os.path.join(odir, "mean_regulator_emb.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(reg_means, f)
    print("Finished!")  
    print("Saved mean regulator embeddings to pickle file:", out_pkl)
    print("Saved regulator embeddings to hdf5 file:", f"{odir}/regulator_emb_on_region.hdf5")
    
    if return_data:
        return reg_means, reg_emb_dict, overlap_bed

@click.command(name="embed_regulator", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region file.")
@click.option("--regulator", required=True,
              help="Regulators of interest, e.g. EZH2 or EZH2;BRD4. Use ';' to separate multiple regulators.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False), help="Resolution.")
@click.option("--batch-size", default=64, show_default=True, type=int, help="Batch size.")
@click.option("--num-workers", default=8, show_default=True, type=int, help="Dataloader workers.")

@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default=os.path.expanduser("~/.cache/chrombert/data"),
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache dir (contains config/ checkpoint/ etc).")

def embed_regulator(region, regulator, odir, genome, resolution, batch_size, num_workers,
        chrombert_cache_dir):
    '''
    Extract general regulator embeddings on specified regions
    '''
    args = SimpleNamespace(
        region=region,
        regulator=regulator,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    embed_regulator()
