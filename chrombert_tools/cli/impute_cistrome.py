import os
import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
import subprocess as sp
import torch
from tqdm import tqdm
import json

import chrombert
from chrombert import DatasetConfig,ChromBERTFTConfig
from .utils import resolve_paths, check_files, check_region_file


def overlap_cistrome_func(cistrome, chrombert_meta_file):
    focus_cistrome_list = [r.strip().lower() for r in cistrome.split(";") if r.strip()]
    
    overlap_cistormes = []
    not_overlap_cistromes = []
    cistrome_gsmid_dict = {}
    
    with open(chrombert_meta_file) as f:
        meta = json.load(f)
    
    for cis in focus_cistrome_list:
        reg, ct = cis.split(":")
        ct = ct.lower()
        reg = reg.lower()
        if reg not in meta:
            print(f"regulator: {reg} not found in ChromBERT meta file.")
            not_overlap_cistromes.append(cis)
            continue
        if not ct.startswith("gsm") and not ct.startswith("enc"):
            tmp_ct = f'dnase:{ct}'
        else:
            tmp_ct = ct  # cistrome
        if tmp_ct not in meta:
            not_overlap_cistromes.append(cis)
            print(f"celltype: {ct} has not corresponding wide type dnase prompt/cistrome in ChromBERT.")
            continue
        else:
            overlap_cistormes.append(cis)
            cistrome_gsmid_dict[cis] = f"{reg}:{tmp_ct}"
                    
    print("Note: All cistromes names were converted to lowercase for matching.")
    print(
        f"Cistromes count summary - requested: {len(focus_cistrome_list)}, "
        f"matched in ChromBERT: {len(overlap_cistormes)}, "
        f"not found: {len(not_overlap_cistromes)}"
    )
    print(f"ChromBERT cistromes metas: {chrombert_meta_file.replace('.json', '.tsv')}")
    return overlap_cistormes, not_overlap_cistromes, cistrome_gsmid_dict

def model_emb_func(args,files_dict,odir):
    # init datamodule
    data_config = DatasetConfig(
        kind = "GeneralDataset",
        supervised_file = f"{odir}/model_input.tsv",
        hdf5_file = files_dict["hdf5_file"],
        batch_size = 4,
        num_workers = 8,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()

    # init chrombert
    model_config = ChromBERTFTConfig(
        genome = args.genome,
        dropout = 0,
        task = "general",
        pretrain_ckpt = files_dict["pretrain_ckpt"],
        mtx_mask = files_dict["mtx_mask"],
    )
    model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
    return ds, dl, model_emb

def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    
    files_dict = resolve_paths(args)
    check_files(files_dict, required_keys=[
        "chrombert_region_file",
        "hdf5_file",
        "pretrain_ckpt",
        "prompt_ckpt"])

    # overlap chrombert region and user-provided region

    
    focus_region = args.region
    overlap_bed = check_region_file(focus_region,files_dict,odir)
    overlap_idx = overlap_bed["build_region_index"].to_numpy()
    
    # overlap chrombert cistrome and user-provided cistrome
    focus_cistrome = args.cistrome
    overlap_cistrome, not_overlap_cistrome, cistrome_gsmid_dict = overlap_cistrome_func(focus_cistrome, files_dict["meta_file"])
    
    # init model embedding
    ds, dl, model_emb = model_emb_func(args,files_dict,odir)
    
    # init impute model
    mc = ChromBERTFTConfig(
        task="prompt",
        prompt_kind="cistrome",
        dropout=0,
        genome=args.genome,
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        finetune_ckpt=files_dict["prompt_ckpt"],
        mtx_mask=files_dict["mtx_mask"],
    )
    model_impute = mc.init_model().cuda().bfloat16().eval()
    
    # forward
    results_probs_dict = {}
    logits_results_cis = {}
    chrombert_regions = []
    input_regions = []
    total_counts = 0
    for cis in cistrome_gsmid_dict.keys():
        logits_results_cis[cis] = []
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dl), total=len(dl)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            model_emb(batch)  # init embedding manager for each batch
            
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
            
            emb_dict = {}
            for cis, mod_cis in cistrome_gsmid_dict.items():
                reg, ct = mod_cis.split(":")
                if ct not in emb_dict:
                    emb_dict[ct] = model_emb.get_cistrome_embedding(ct)
                if reg not in emb_dict:
                    emb_dict[reg] = model_emb.get_regulator_embedding(reg)
                if "all" not in emb_dict:
                    emb_dict['all'] = model_emb.get_region_embedding()
                ct_emb = emb_dict[ct]
                reg_emb = emb_dict[reg]
                all_emb = emb_dict["all"]
                header_out = model_impute.ft_header(ct_emb, reg_emb, all_emb)
                logits_results_cis[cis].append(header_out.detach().cpu())
                
    for key, value in logits_results_cis.items():
        logits = torch.concat(value)
        probs = torch.sigmoid(logits).to(dtype=torch.float32).numpy()
        results_probs_dict[key] = probs
        
    chrombert_regions_array = np.concatenate(chrombert_regions, axis=0)[:, 1:]
    input_regions_array = np.concatenate(input_regions, axis=0)
    region_df = pd.DataFrame(
        np.concatenate([input_regions_array, chrombert_regions_array], axis=1),
        columns=["input_chrom", "input_start", "input_end", "chrombert_build_region_index", "chrombert_start", "chrombert_end"]
    )

    results_pro_df = pd.DataFrame(results_probs_dict)
    results_pro_df = pd.concat([region_df, results_pro_df], axis=1)
    results_pro_df.to_csv(f'{odir}/results_prob_df.csv', index=False)

    print("Finished imputing cistromes on specific regions.")
    print(f"Results saved to {odir}/results_prob_df.csv")


@click.command(name="impute_cistrome", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True,
              help="Region BED file.")
@click.option("--cistrome", required=True,
              help="factor:cell e.g. BCL11A:GM12878;BRD4:MCF7;CTCF:HepG2. Use ';' to separate multiple cistromes.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False),
              help="Output directory.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb"], case_sensitive=False), help="Resolution. Only supports 1kb resolution in imputing cistromes task.")
@click.option("--batch-size", "batch_size", default=4, show_default=True, type=int,
              help="Batch size. if you have enough GPU memory, you can set it to a larger value.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default=os.path.expanduser('~/.cache/chrombert/data'),
              show_default=True,
              type=click.Path(file_okay=False),
              help="ChromBERT cache directory (containing config/ and anno/ subfolders).")


def impute_cistrome(region, cistrome, odir, genome, resolution, batch_size, chrombert_cache_dir):
    '''
    Impute cistromes on specified regions
    '''
    args = SimpleNamespace(
        region=region,
        cistrome=cistrome,
        odir=odir,
        genome=genome,
        resolution=resolution,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    impute_cistrome()
