import os
import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json

from chrombert_hf import ChromBERTFTConfig
from .utils import resolve_paths, check_files, check_region_file, get_model_name
from .utils_embed import build_dataloader

import pyBigWig
import bioframe as bf


# =========================
# cistrome utils
# =========================

def overlap_cistrome_func(cistrome, chrombert_meta_file):
    focus_cistrome_list = [r.strip().lower() for r in cistrome.split(";") if r.strip()]

    overlap_cistromes = []
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

        if ct.startswith("gsm") or ct.startswith("enc"):
            cistrome_id = ct
        else:
            dnase_key = f'dnase:{ct}'
            if dnase_key not in meta:
                print(f"celltype: {ct} has no corresponding wild type dnase data in ChromBERT.")
                not_overlap_cistromes.append(cis)
                continue
            cistrome_id = meta[dnase_key]

        if cistrome_id not in meta:
            print(f"cistrome ID: {cistrome_id} not found in ChromBERT meta file.")
            not_overlap_cistromes.append(cis)
            continue

        overlap_cistromes.append(cis)
        cistrome_gsmid_dict[cis] = f"{reg}:{cistrome_id}"

    print("Note: All cistromes names were converted to lowercase for matching.")
    print(
        f"Cistromes count summary - requested: {len(focus_cistrome_list)}, "
        f"matched in ChromBERT: {len(overlap_cistromes)}, "
        f"not found: {len(not_overlap_cistromes)}, "
        f"not found cistromes: {not_overlap_cistromes}"
    )
    print(f"ChromBERT cistromes metas: {chrombert_meta_file.replace('.json', '.tsv')}")
    return overlap_cistromes, not_overlap_cistromes, cistrome_gsmid_dict


def validate_args(args):
    if args.region is None:
        raise ValueError("You must provide --region.")
    if args.cistrome is None:
        raise ValueError("You must provide --cistrome.")


def get_required_keys(args):
    return [
        "chrombert_region_file",
        "hdf5_file",
        # "pretrain_ckpt",
        "prompt_ckpt",
    ]


# =========================
# prepare
# =========================

def prepare_region_and_cistrome(args, files_dict, odir):
    overlap_bed = check_region_file(args.region, files_dict, odir)
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False) # [chrom, start, end, build_region_index, start_input, end_input, label]

    overlap_cistromes, not_overlap_cistromes, cistrome_gsmid_dict = overlap_cistrome_func(
        args.cistrome, files_dict["meta_file"]
    )
    if len(cistrome_gsmid_dict) == 0:
        raise ValueError("No requested cistromes matched ChromBERT cistrome list. Nothing to impute.")

    return overlap_bed, cistrome_gsmid_dict


def build_impute_model(args, files_dict):
    mc = ChromBERTFTConfig(
        task="prompt",
        prompt_kind="cistrome",
        dropout=0,
        genome=args.genome,
        pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        finetune_ckpt=files_dict["prompt_ckpt"],
        mtx_mask=files_dict["mtx_mask"],
    )
    return mc.init_model().cuda().bfloat16().eval()


# =========================
# impute
# =========================

def run_impute(args, files_dict, odir, return_data=False):
    overlap_bed, cistrome_gsmid_dict = prepare_region_and_cistrome(args, files_dict, odir)

    ds, dl = build_dataloader(
        supervised_file=f"{odir}/model_input.tsv",
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=getattr(args, "num_workers", 8),
    )
    model_impute = build_impute_model(args, files_dict)
    model_emb = model_impute.get_embedding_manager().cuda().bfloat16()

    results_probs_dict = {}
    logits_results_cis = {cis: [] for cis in cistrome_gsmid_dict}
    chrombert_regions = []
    input_regions = []
    total_counts = 0

    with torch.no_grad():
        for batch in tqdm(dl, total=len(dl), desc="Imputing cistromes"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            model_emb(batch)

            bs = batch["region"].shape[0]
            start_idx = total_counts
            total_counts += bs
            end_idx = total_counts

            chrombert_regions.append(batch["region"].long().cpu().numpy())
            batch_index = batch["build_region_index"].long().cpu().unsqueeze(-1).numpy()
            input_region = overlap_bed[["chrom","start_input","end_input","build_region_index"]].iloc[start_idx:end_idx][:].values
            input_regions.append(input_region)
            assert (batch_index.reshape(-1) == input_region[:, -1].reshape(-1)).all(), \
                "Batch index and region index do not match"

            emb_dict = {}
            for cis, mod_cis in cistrome_gsmid_dict.items():
                reg, ct = mod_cis.split(":")
                if ct not in emb_dict:
                    emb_dict[ct] = model_emb.get_cistrome_embedding(ct)
                if reg not in emb_dict:
                    emb_dict[reg] = model_emb.get_regulator_embedding(reg)
                if "all" not in emb_dict:
                    emb_dict["all"] = model_emb.get_region_embedding()
                header_out = model_impute.ft_header(emb_dict[ct], emb_dict[reg], emb_dict["all"])
                logits_results_cis[cis].append(header_out.detach().cpu())

    for key, value in logits_results_cis.items():
        logits = torch.concat(value)
        results_probs_dict[key] = torch.sigmoid(logits).to(dtype=torch.float32).numpy()

    results_pro_df = compute_and_save_results(
        chrombert_regions, input_regions, results_probs_dict, overlap_bed, odir, args.genome
    )

    report_impute(args, odir, overlap_bed)

    if return_data:
        return results_pro_df


# =========================
# output
# =========================

def compute_and_save_results(chrombert_regions, input_regions, results_probs_dict, overlap_bed, odir, genome="hg38"):
    chrombert_regions_array = np.concatenate(chrombert_regions, axis=0)[:, 1:]
    input_regions_array = np.concatenate(input_regions, axis=0)
    region_df = pd.DataFrame(
        np.concatenate([input_regions_array, chrombert_regions_array], axis=1),
        columns=["input_chrom", "input_start", "input_end",
                 "chrombert_build_region_index", "chrombert_start", "chrombert_end"],
    )

    raw_results_pro_df = pd.DataFrame(results_probs_dict)
    results_pro_df = pd.concat([region_df, raw_results_pro_df], axis=1)
    results_pro_df.to_csv(f"{odir}/results_prob_df.csv", index=False)

    bw_results_pro_df = results_pro_df.drop(
        columns=["input_start", "input_end", "chrombert_build_region_index"]
    )
    bw_results_pro_df.rename(
        columns={"input_chrom": "chrom", "chrombert_start": "start", "chrombert_end": "end"},
        inplace=True,
    )
    get_bw(bw_results_pro_df, odir, genome=genome)

    return results_pro_df


def get_bw(results_pro_df, odir, genome="hg38"):
    chrom_sizes = bf.fetch_chromsizes(genome)
    chrom_order = list(chrom_sizes.keys())

    results_pro_df["chrom"] = pd.Categorical(
        results_pro_df["chrom"], categories=chrom_order, ordered=True
    )
    results_pro_df = results_pro_df.sort_values(by=["chrom", "start"])

    for column in results_pro_df.columns:
        if column in ["chrom", "start", "end"]:
            continue
        cell = column.split(":")[-1]
        factor = column.split(":")[0]
        opath = f"{odir}/{factor}_{cell}.bw"
        bw = pyBigWig.open(opath, "w")
        bw.addHeader(list(chrom_sizes.items()))

        chrom_list = results_pro_df["chrom"].astype(str).tolist()
        start_list = results_pro_df["start"].astype(int).tolist()
        end_list = results_pro_df["end"].astype(int).tolist()
        values_list = results_pro_df[column].astype(float).tolist()

        bw.addEntries(chrom_list, start_list, end_list, values=values_list)
        bw.close()


# =========================
# report
# =========================

def report_impute(args, odir, overlap_bed):
    total_focus = sum(1 for _ in open(args.region))
    no_overlap_region_len = (
        sum(1 for _ in open(f"{odir}/no_overlap_region.bed"))
        if os.path.exists(f"{odir}/no_overlap_region.bed")
        else 0
    )

    print("\nFinished imputing cistromes on specific regions.")
    print(
        f"Focus region summary - total: {total_focus}, "
        f"overlapping with ChromBERT: {overlap_bed.shape[0]}, "
        f"non-overlapping: {no_overlap_region_len}"
    )
    print("Overlapping regions BED file:", f"{odir}/overlap_region.bed")
    print("Non-overlapping regions BED file:", f"{odir}/no_overlap_region.bed")
    print("Results saved to:", f"{odir}/results_prob_df.csv")
    print("Results track files saved to:", f"{odir}/*.bw")


# =========================
# entry
# =========================

def run(args, return_data=False):
    for attr, default in [
        ("num_workers", 8),
        ("resolution", "1kb"),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    validate_args(args)
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    check_files(files_dict, required_keys=get_required_keys(args))

    return run_impute(args, files_dict, odir, return_data=return_data)


# =========================
# CLI
# =========================

@click.command(name="predict_tf_binding_regions", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region BED file.")
@click.option("--cistrome", required=True,
              help="factor:cell e.g. BCL11A:GM12878;BRD4:MCF7;CTCF:HepG2. Use ';' to separate multiple cistromes.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--oname", default="cistrome_impute", show_default=True,
              type=str, help="Output name prefix.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb"], case_sensitive=False),
              help="Resolution. Only supports 1kb resolution in imputing cistromes task.")
@click.option("--batch-size", "batch_size", default=4, show_default=True, type=int,
              help="Batch size.")
@click.option("--num-workers", "num_workers", default=8, show_default=True, type=int,
              help="Dataloader workers.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default="~/.cache/chrombert/data", show_default=True,
              type=click.Path(file_okay=False),
              help="ChromBERT cache directory (containing config/ and anno/ subfolders).")
def predict_tf_binding_regions(region, cistrome, odir, oname, genome, resolution, batch_size, num_workers, chrombert_cache_dir):
    '''
    Predict TF binding on specified regions across cell types.
    '''
    args = SimpleNamespace(
        region=region,
        cistrome=cistrome,
        odir=odir,
        oname=oname,
        genome=genome.lower(),
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    predict_tf_binding_regions()
