'''
This script is used to impute cistromes on specific regions (by dnase prompts).

Two typical usage patterns:

1) Use a prepared ChromBERT cache directory (recommended):

python run.py \
    --cistrome BCL11A:GM12878;BRD4:MCF7;CTCF:HepG2 \
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
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Impute cistromes on specific regions (by dnase prompts)."
    )
    parser.add_argument(
        "--region_bed",
        type=str,
        required=True,
        help="Region BED file.",
    )
    parser.add_argument(
        "--cistrome",
        type=str,
        required=True,
        help="factor:cell e.g. BCL11A:GM12878;BRD4:MCF7;CTCF:HepG2. Use ';' to separate multiple cistromes.",
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
    """Resolve cistrome list file and embedding paths based on cache dir / overrides."""
    assert args.chrombert_cache_dir is not None, "ChromBERT cache directory must be provided"

    # 1) ChromBERT cistrome list file
    chrombert_regulator_file = os.path.join(args.chrombert_cache_dir, "config", "hg38_6k_regulators_list.txt")
    
    # 2) ChromBERT input hdf5 file
    hdf5_file = os.path.join(args.chrombert_cache_dir, "hg38_6k_1kb.hdf5")
    
    # 3) ChromBERT pretrain ckpt file
    pretrain_ckpt = os.path.join(args.chrombert_cache_dir,"checkpoint", "hg38_6k_1kb_pretrain.ckpt")
    
    # 4) ChromBERT matcix mask file:
    mtx_mask = os.path.join(args.chrombert_cache_dir,"config", "hg38_6k_mask_matrix.tsv")
    
    # 5) ChromBERT region file:
    chrombert_region_file = os.path.join(args.chrombert_cache_dir, "config", "hg38_6k_1kb_region.bed")
    
    # 6) ChromBERT meta file:
    meta_file = os.path.join(args.chrombert_cache_dir, "config", "hg38_6k_meta.json")
    
    #7) ChromBERT finetuned ckpt file:
    finetuned_ckpt = os.path.join(args.chrombert_cache_dir,"checkpoint", "hg38_6k_1kb_prompt_cistrome.ckpt")

    return {
        "chrombert_regulator_file": chrombert_regulator_file,
        "hdf5_file": hdf5_file,
        "pretrain_ckpt": pretrain_ckpt,
        "mtx_mask": mtx_mask,
        "chrombert_region_file": chrombert_region_file,
        "meta_file": meta_file,
        "finetuned_ckpt": finetuned_ckpt,
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

def overlap_cistrome_func(cistrome, chrombert_meta_file):

    focus_cistrome_list = [r.strip().lower() for r in cistrome.split(";") if r.strip()]
    
    overlap_cistormes = []
    not_overlap_cistromes = []
    cistrome_gsmid_dict={}    
    with open(chrombert_meta_file) as f:
        meta = json.load(f)
    for cis in focus_cistrome_list:
        reg,ct = cis.split(":")
        ct = ct.lower()
        reg = reg.lower()
        if reg not in meta:
            print(f"regulator: {reg} not found in ChromBERT meta file.")
            not_overlap_cistromes.append(cis)
            continue
        if not ct.startswith("gsm") and not ct.startswith("enc"):
            tmp_ct = f'dnase:{ct}'
        else:
            tmp_ct = ct # citrome
        if tmp_ct not in meta:
            not_overlap_cistromes.append(cis)
            print(f"celltype: {ct} has not corresponding wide type dnase prompt/cistrome in ChromBERT.")
            continue
        else:
            overlap_cistormes.append(cis)
            cistrome_gsmid_dict[cis]=f"{reg}:{meta[tmp_ct]}"
                    
    print("Note: All cistromes names were converted to lowercase for matching.")
    print(
        f"Cistromes count summary - requested: {len(focus_cistrome_list)}, "
        f"matched in ChromBERT: {len(overlap_cistormes)}, "
        f"not found: {len(not_overlap_cistromes)}"
    )
    print(f"ChromBERT cistromes metas: {chrombert_meta_file.replace('.json','.tsv')}")
    return overlap_cistormes, not_overlap_cistromes, cistrome_gsmid_dict

def main():
    args = parse_args()
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    focus_cistrome = args.cistrome
    files_dict = resolve_paths(args)
    check_files(files_dict, args)

    # overlap chrombert region and user-provided region
    maping_chrom_dict = {"chr1": 1,"chr2": 2,"chr3": 3,"chr4": 4,"chr5": 5, "chr6": 6,"chr7": 7,"chr8": 8,"chr9": 9,"chr10": 10,"chr11": 11,"chr12": 12,"chr13": 13,"chr14": 14,"chr15": 15,"chr16": 16,"chr17": 17,"chr18": 18,"chr19": 19,"chr20": 20,"chr21": 21,"chr22": 22,"chrX": 23,"chrY": 24}
    
    overlap_bed, overlap_idx = overlap_region(args.region_bed, files_dict["chrombert_region_file"], odir)
    # overlap_bed["chrom"] = overlap_bed["chrom"].map(maping_chrom_dict)
    
    
    # overlap chrombert cistrome and user-provided cistrome
    overlap_cistrome, not_overlap_cistrome, cistrome_gsmid_dict = overlap_cistrome_func(focus_cistrome, files_dict["meta_file"])
    
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
    
    # init chrombert embed model
    model_config = chrombert.get_preset_model_config(
        preset = "general",
        basedir = args.chrombert_cache_dir,
        genome = "hg38",
        dropout = 0,
        pretrain_ckpt =  files_dict["pretrain_ckpt"],
        mtx_mask = files_dict["mtx_mask"],
    )
    model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
    
    
    # init impute model
    mc = chrombert.get_preset_model_config(
        preset ="prompt_cistrome", 
        dropout = 0,
        basedir = args.chrombert_cache_dir,
        genome = "hg38",
        pretrain_ckpt =  files_dict["pretrain_ckpt"],
        finetune_ckpt = files_dict["finetuned_ckpt"],
        mtx_mask = files_dict["mtx_mask"],
        ) 
    model_impute = mc.init_model().cuda().bfloat16().eval()
    
    
    # forward
    results_probs_dict = {}
    logits_results_cis = {}
    chrombert_regions = []
    input_regions = []
    total_counts=0
    for cis in cistrome_gsmid_dict.keys():
        logits_results_cis[cis] = []
    with torch.no_grad():
        for idx,batch in tqdm(enumerate(dl), total = len(dl)):
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            model_emb(batch) # init embedding manager for each batch
            
            bs = batch["region"].shape[0]
            
            statr_idx = total_counts
            total_counts += bs
            end_idx = total_counts
            chrombert_region = batch["region"].long().cpu().numpy()
            chrombert_regions.append(chrombert_region)
            batch_index = batch["build_region_index"].long().cpu().unsqueeze(-1).numpy()
            input_region = overlap_bed.iloc[statr_idx:end_idx][:].values
            input_regions.append(input_region)
            assert (batch_index.reshape(-1) == input_region[:, -1].reshape(-1)).all(), "Batch index and region index do not match"
            
            emb_dict={}
            for cis,mod_cis in cistrome_gsmid_dict.items():
                reg,ct = mod_cis.split(":")
                if ct not in emb_dict:
                    emb_dict[ct] = model_emb.get_cistrome_embedding(ct)
                if reg not in emb_dict:
                    emb_dict[reg] = model_emb.get_regulator_embedding(reg)
                if "all" not in emb_dict:
                    emb_dict['all'] = model_emb.get_region_embedding()
                ct_emb = emb_dict[ct]
                reg_emb = emb_dict[reg]
                all_emb = emb_dict["all"]
                header_out = model_impute.ft_header(ct_emb,reg_emb,all_emb)
                logits_results_cis[cis].append(header_out.detach().cpu())
                
    for key,value in logits_results_cis.items():
        logits = torch.concat(value)
        probs = torch.sigmoid(logits).to(dtype=torch.float32).numpy()
        results_probs_dict[key] = probs
        
    
    chrombert_regions_array = np.concatenate(chrombert_regions, axis=0)[:,1:]
    input_regions_array = np.concatenate(input_regions, axis=0)
    region_df = pd.DataFrame(np.concatenate([input_regions_array,chrombert_regions_array], axis=1), columns = ["input_chrom", "input_start", "input_end", "chrombert_build_region_index","chrombert_start", "chrombert_end"])

    results_pro_df = pd.DataFrame(results_probs_dict)
    results_pro_df = pd.concat([region_df,results_pro_df], axis=1)
    results_pro_df.to_csv(f'{odir}/results_prob_df.csv',index = False)

    print("Finished imputing cistromes on specific regions.")
    print(f"Results saved to {odir}/results_prob_df.csv")
    
if __name__ == "__main__":
    main()
