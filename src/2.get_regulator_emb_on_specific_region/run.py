'''
This script is used to extract embeddings for specified regulators on specific regions.

Two typical usage patterns:

1) Use a prepared ChromBERT cache directory (recommended):

python run.py \
    --regulator EZH2;BRD4;CTCF;FOXA3 \
    --region_bed /path/to/region.bed \
    --odir output \
    --chrombert_cache_dir /path/to/chrombert/data
    
output:

'''

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import subprocess as sp
import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
from chrombert.scripts.utils import HDF5Manager
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ChromBERT regulator embeddings for user-specified regulators."
    )
    parser.add_argument(
        "--region_bed",
        type=str,
        required=True,
        help="Region BED file.",
    )
    parser.add_argument(
        "--regulator",
        type=str,
        required=True,
        help="Regulators of interest, e.g. EZH2 or EZH2;BRD4. Use ';' to separate multiple regulators.",
    )
    parser.add_argument(
        "--odir",
        type=str,
        required=False,
        help="Output directory.",
        default="./output",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Batch size. if you have enough GPU memory, you can set it to a larger value.",
        default=64,
    )

    # Mode 1: use a ChromBERT cache dir (standard layout)
    parser.add_argument(
        "--chrombert_cache_dir",
        type=str,
        required=False,
        default=os.path.expanduser('~/.cache/chrombert/data'),
        help="ChromBERT cache directory (containing config/ and anno/ subfolders).",
    )
    return parser.parse_args()


def resolve_paths(args):
    """Resolve regulator list file and embedding paths based on cache dir / overrides."""
    assert args.chrombert_cache_dir is not None, "ChromBERT cache directory must be provided"

    # 1) ChromBERT regulator list file
    chrombert_regulator_file = os.path.join(args.chrombert_cache_dir, "config", "hg38_6k_regulators_list.txt")
    
    # 2) ChromBERT input hdf5 file
    hdf5_file = os.path.join(args.chrombert_cache_dir, "hg38_6k_1kb.hdf5")
    
    # 3) ChromBERT pretrain ckpt file
    pretrain_ckpt = os.path.join(args.chrombert_cache_dir,"checkpoint", "hg38_6k_1kb_pretrain.ckpt")
    
    # 4) ChromBERT matcix mask file:
    mtx_mask = os.path.join(args.chrombert_cache_dir,"config", "hg38_6k_mask_matrix.tsv")
    
    # 5) ChromBERT region file:
    chrombert_region_file = os.path.join(args.chrombert_cache_dir, "config", "hg38_6k_1kb_region.bed")
    
    return {
        "chrombert_regulator_file": chrombert_regulator_file,
        "hdf5_file": hdf5_file,
        "pretrain_ckpt": pretrain_ckpt,
        "mtx_mask": mtx_mask,
        "chrombert_region_file": chrombert_region_file,
    }


def check_files(files_dict, args):
    """Check that required ChromBERT files exist, and give helpful hints if not."""

    for key, value in files_dict.items():
        if not os.path.exists(value):
            msg = (f"ChromBERT {key} file not found: {value}.\n "
                    "You can download all required files by running the command "
                "`chrombert_prepare_env`")
            print(msg)
            raise FileNotFoundError(msg)

def overlap_region(region_bed, chrombert_region_file, odir):
    # ---------- overlapping focus regions ----------
    cmd_overlap = f"""
    cut -f 1-3 {region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools intersect -F 0.5 -wa -wb -a {chrombert_region_file} -b - \
    | awk 'BEGIN{{OFS="\\t"}}{{print $5,$6,$7,$4}}' \
    > {odir}/overlap_focus.bed
    """
    sp.run(cmd_overlap, shell=True, check=True, executable="/bin/bash")

    overlap_bed = pd.read_csv(
        f"{odir}/overlap_focus.bed",
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "build_region_index"],
    )
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)
    overlap_idx = overlap_bed["build_region_index"].to_numpy()

    # ---------- non-overlapping focus regions ----------
    cmd_no = f"""
    cut -f 1-3 {region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools intersect -f 0.5 -v -a - -b {chrombert_region_file} \
    > {odir}/no_overlap_focus.bed
    """
    sp.run(cmd_no, shell=True, check=True, executable="/bin/bash")
    
    total_focus = sum(1 for _ in open(region_bed))
    no_overlap_region_len = sum(1 for _ in open(f"{odir}/no_overlap_focus.bed"))
    print(
        f"Focus region summary - total: {total_focus}, "
        f"overlapping with ChromBERT: {overlap_bed.shape[0]}, It is possible for a single focus region to overlap multiple ChromBERT regions,"
        f"non-overlapping: {no_overlap_region_len}"
    )
    print("Overlapping focus regions BED file:", f"{odir}/overlap_focus.bed")
    print("Overlapping focus regions will be used for ChromBERT embedding.")
    print("Non-overlapping focus regions BED file:", f"{odir}/no_overlap_focus.bed")
    return overlap_bed, overlap_idx

def overlap_regulator_func(regulator, chrombert_regulator_file):
    # load regulator list from ChromBERT
    chrombert_regulator = pd.read_csv(
        chrombert_regulator_file,
        sep="\t",
        header=None,
        names=["regulator"],
    )["regulator"].tolist()
    chrombert_regulator = [i.lower() for i in chrombert_regulator]

    # parse user-provided regulators
    focus_regulator_list = [r.strip().lower() for r in regulator.split(";") if r.strip()]

    overlap_regulator = list(set(chrombert_regulator) & set(focus_regulator_list))
    not_overlap_regulator = list(set(focus_regulator_list) - set(chrombert_regulator))
    
    regulator_dict = {tmp_regulator:chrombert_regulator.index(tmp_regulator) for tmp_regulator in overlap_regulator}
    
    print("Note: All regulator names were converted to lowercase for matching.")
    print(
        f"Regulator count summary - requested: {len(focus_regulator_list)}, "
        f"matched in ChromBERT: {len(overlap_regulator)}, "
        f"not found: {len(not_overlap_regulator)}"
    )
    print(f"ChromBERT regulator list loaded from: {chrombert_regulator_file}")
    return overlap_regulator, not_overlap_regulator, regulator_dict

def main():
    args = parse_args()
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    focus_regulator = args.regulator
    files_dict = resolve_paths(args)
    check_files(files_dict, args)

    # overlap chrombert region and user-provided region
    maping_chrom_dict = {"chr1": 1,"chr2": 2,"chr3": 3,"chr4": 4,"chr5": 5, "chr6": 6,"chr7": 7,"chr8": 8,"chr9": 9,"chr10": 10,"chr11": 11,"chr12": 12,"chr13": 13,"chr14": 14,"chr15": 15,"chr16": 16,"chr17": 17,"chr18": 18,"chr19": 19,"chr20": 20,"chr21": 21,"chr22": 22,"chrX": 23,"chrY": 24}

    overlap_bed, overlap_idx = overlap_region(args.region_bed, files_dict["chrombert_region_file"], odir)
    overlap_bed["chrom"] = overlap_bed["chrom"].map(maping_chrom_dict)
    
    # overlap chrombert regulator and user-provided regulator
    overlap_regulator, not_overlap_regulator, regulator_idx_dict = overlap_regulator_func(focus_regulator, files_dict["chrombert_regulator_file"])
    
    # init datamodule
    data_config = DatasetConfig(
        kind = "GeneralDataset",
        supervised_file = f"{odir}/model_input.tsv",
        hdf5_file = files_dict["hdf5_file"],
        batch_size = args.batch_size,
        num_workers = 8,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()
    
    # init chrombert
    model_config = chrombert.get_preset_model_config(
        basedir = args.chrombert_cache_dir,
        genome = "hg38",
        dropout = 0,
        preset = "general",
        pretrain_ckpt = files_dict["pretrain_ckpt"],
        mtx_mask = files_dict["mtx_mask"],
    )
    model = model_config.init_model().get_embedding_manager().cuda().bfloat16()
    
    total_counts = 0
    shapes = {f"emb/{k}": [(len(ds),768), np.float16] for k in regulator_idx_dict}
    with HDF5Manager(f'{odir}/save_regulator_emb_on_specific_region.hdf5', region=[(len(ds),4), np.int64],**shapes) as h5:
        with torch.no_grad():
            for batch in tqdm(dl, total = len(dl)):
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                model(batch) # initialize the cache 
                # region = np.concatenate([
                #     batch["region"].long().cpu().numpy(), 
                #     batch["build_region_index"].long().cpu().unsqueeze(-1).numpy()
                #     ], axis = 1
                # )
                bs = batch["region"].shape[0]
                statr_idx = total_counts
                total_counts += bs
                end_idx = total_counts
                batch_index = batch["build_region_index"].long().cpu().unsqueeze(-1).numpy()
                region = overlap_bed.iloc[statr_idx:end_idx][:].values
                assert (batch_index.reshape(-1) == region[:, -1].reshape(-1)).all(), "Batch index and region index do not match"
                embs = {
                    f"emb/{k}": model.get_regulator_embedding(k).float().cpu().detach().numpy()
                    for k,v in regulator_idx_dict.items()
                }
                h5.insert(region = region, **embs)

    print("Finished!")  
    print("Saved regulator embeddings on specific region to hdf5:", f'{odir}/save_regulator_emb_on_specific_region.hdf5')

if __name__ == "__main__":
    main()
