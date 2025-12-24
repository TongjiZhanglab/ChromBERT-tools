import os
import pandas as pd
import numpy as np
import subprocess as sp
import torchmetrics as tm
import json
import bbi
import glob
import torch
import re
import random
from sklearn.metrics.pairwise import cosine_similarity

def _nd_from_genome(genome: str) -> str:
    genome = genome.lower()
    if genome == "hg38":
        return "6k"
    elif genome == "mm10":
        return "5k"
    raise ValueError(f"Genome {genome} not supported!")

def resolve_paths(args):
    """
    Resolve all required ChromBERT files based on:
      - genome: hg38/mm10
      - resolution: 1kb/200bp/2kb/4kb
      - optional overrides: --chrombert-xxx-file
    """
    n_d = _nd_from_genome(args.genome)
    cache_path = os.path.expanduser(args.chrombert_cache_dir)
    # region bed
    chrombert_region_file = os.path.join(cache_path, f"config/{args.genome}_{n_d}_{args.resolution}_region.bed")

    # regulator file
    chrombert_regulator_file = os.path.join(cache_path, f"config/{args.genome}_{n_d}_regulators_list.txt")
    
    # factor file
    chrombert_factor_file = os.path.join(cache_path, f"config/{args.genome}_{n_d}_factors_list.txt")

    # hdf5
    hdf5_file = os.path.join(cache_path, f"{args.genome}_{n_d}_{args.resolution}.hdf5")
    
    # ckpt
    pretrain_ckpt = os.path.join(cache_path, "checkpoint", f"{args.genome}_{n_d}_{args.resolution}_pretrain.ckpt")
    
    # mask matrix
    mtx_mask = os.path.join(cache_path, "config", f"{args.genome}_{n_d}_mask_matrix.tsv")
    
    # region embedding file
    region_emb_file = os.path.join(cache_path, f"anno/{args.genome}_{args.resolution}_region_emb.npy")
    
    # chrombert_gene_meta
    gene_meta_tsv = os.path.join(cache_path, f"anno/{args.genome}_{args.resolution}_gene_meta.tsv")
    
    # chrombert base chromatin accessibility signal
    base_ca_signal = os.path.join(cache_path, "anno", f"{args.genome}_{args.resolution}_accessibility_signal_mean.npy")
    
    # chrombert meta file
    meta_file = os.path.join(cache_path, "config", f"{args.genome}_{n_d}_meta.json")
    
    # prompt ckpt
    prompt_ckpt = os.path.join(cache_path, "checkpoint", f"{args.genome}_{n_d}_{args.resolution}_prompt_cistrome.ckpt")
    
    # Override with user-provided paths if available
    chrombert_region_file = getattr(args, "chrombert_region_file", None) or chrombert_region_file
    chrombert_regulator_file = getattr(args, "chrombert_regulator_file", None) or chrombert_regulator_file
    chrombert_factor_file = getattr(args, "chrombert_factor_file", None) or chrombert_factor_file
    hdf5_file = getattr(args, "hdf5_file", None) or hdf5_file
    pretrain_ckpt = getattr(args, "pretrain_ckpt", None) or pretrain_ckpt
    mtx_mask = getattr(args, "mtx_mask", None) or mtx_mask
    region_emb_file = getattr(args, "chrombert_region_emb_file", None) or region_emb_file
    gene_meta_tsv = getattr(args, "chrombert_gene_meta", None) or gene_meta_tsv
    base_ca_signal = getattr(args, "base_ca_signal", None) or base_ca_signal
    meta_file = getattr(args, "meta_file", None) or meta_file

    return {
        "chrombert_region_file": chrombert_region_file,
        "chrombert_regulator_file": chrombert_regulator_file,
        "chrombert_factor_file": chrombert_factor_file,
        "hdf5_file": hdf5_file,
        "pretrain_ckpt": pretrain_ckpt,
        "mtx_mask": mtx_mask,
        "region_emb_npy": region_emb_file,
        "gene_meta_tsv": gene_meta_tsv,
        "base_ca_signal": base_ca_signal,
        "meta_file": meta_file,
        "prompt_ckpt": prompt_ckpt,
    }

def check_files(files_dict, required_keys=None):
    """
    Check if required files exist.
    
    Args:
        files_dict: Dictionary of all file paths
        required_keys: List of keys to check. If None, check all files.
    """
    if required_keys is None:
        # Check all files in the dict
        items_to_check = files_dict.items()
    else:
        # Only check specified keys
        items_to_check = [(k, files_dict[k]) for k in required_keys if k in files_dict]
    
    missing = [f"{k}: {v}" for k, v in items_to_check if not os.path.exists(v) and v is not None]
    if missing:
        msg = (
            "ChromBERT required file(s) not found:\n  - "
            + "\n  - ".join(missing)
            + "\nHint: run `chrombert_prepare_env` or pass the missing path(s) explicitly."
        )
        raise FileNotFoundError(msg)
    
def check_region_file(focus_region,files_dict,odir):
    """
    Check if the region file is valid.
    """
    if focus_region.endswith(".bed"):
        overlap_bed = overlap_region(focus_region, files_dict["chrombert_region_file"], odir)
        if overlap_bed.shape[0] == 0:
            raise ValueError("No overlap found between your regions and ChromBERT regions.")
        
    elif focus_region.endswith(".csv") or focus_region.endswith(".tsv"):
        df = pd.read_csv(focus_region) if focus_region.endswith(".csv") else pd.read_csv(focus_region, sep="\t")
        required = ["chrom", "start", "end"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in CSV/TSV: {missing}. Required columns: {required}."
            )
        if "build_region_index" in df.columns:
            overlap_bed = df[required + ['build_region_index']].copy()
            overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)
        else:
            # write a temp BED and reuse overlap_region
            tmp_bed = f"{odir}/tmp_focus_regions.bed"
            df[required].to_csv(tmp_bed, sep="\t", header=False, index=False)
            overlap_bed = overlap_region(tmp_bed, files_dict["chrombert_region_file"], odir)
            if overlap_bed.shape[0] == 0:
                raise ValueError("No overlap found between your regions and ChromBERT regions.")
    else:
        raise ValueError(f"Unsupported region file format: {focus_region}. Only .bed, .csv, and .tsv are supported.")
    return overlap_bed

def overlap_region(region_bed, chrombert_region_file, odir):
    os.makedirs(odir, exist_ok=True)

    # overlapping focus regions
    cmd_overlap = f"""
    cut -f 1-3 {region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools intersect -F 0.5 -wa -wb -a {chrombert_region_file} -b - \
    | awk 'BEGIN{{OFS="\\t"}}{{print $5,$6,$7,$4}}' \
    > {odir}/overlap_region.bed
    """
    sp.run(cmd_overlap, shell=True, check=True, executable="/bin/bash")

    overlap_bed = pd.read_csv(
        f"{odir}/overlap_region.bed",
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "build_region_index"],
    )
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)

    # non-overlapping regions
    cmd_no = f"""
    cut -f 1-3 {region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools intersect -f 0.5 -v -a - -b {chrombert_region_file} \
    > {odir}/no_overlap_region.bed
    """
    sp.run(cmd_no, shell=True, check=True, executable="/bin/bash")

    total_region = sum(1 for _ in open(region_bed))
    no_overlap_len = sum(1 for _ in open(f"{odir}/no_overlap_region.bed"))
    print(
        f"Region summary - total: {total_region}, "
        f"overlapping with ChromBERT: {overlap_bed.shape[0]} (one region may overlap multiple ChromBERT regions), "
        f"non-overlapping: {no_overlap_len}"
    )
    return overlap_bed


def overlap_regulator_func(regulator: str, chrombert_regulator_file: str):
    chrombert_regulator = pd.read_csv(
        chrombert_regulator_file,
        sep="\t",
        header=None,
        names=["regulator"],
    )["regulator"].tolist()
    chrombert_regulator = [i.lower() for i in chrombert_regulator]

    focus_regulator_list = [r.strip().lower() for r in regulator.split(";") if r.strip()]
    overlap_regulator = list(set(chrombert_regulator) & set(focus_regulator_list))
    not_overlap_regulator = list(set(focus_regulator_list) - set(chrombert_regulator))
    regulator_dict = {r: chrombert_regulator.index(r) for r in overlap_regulator}

    print("Note: All regulator names were converted to lowercase for matching.")
    print(
        f"Regulator count summary - requested: {len(focus_regulator_list)}, "
        f"matched in ChromBERT: {len(overlap_regulator)}, "
        f"not found: {len(not_overlap_regulator)}, "
        f"not found regulator: {not_overlap_regulator}"
    )
    print(f"ChromBERT regulators: {chrombert_regulator_file}")
    return overlap_regulator, not_overlap_regulator, regulator_dict


def overlap_cistrome_func(cistrome: str, chrombert_meta_file: str):
    focus_list = [r.strip().lower() for r in cistrome.split(";") if r.strip()]
    gsm_ids = [i for i in focus_list if ":" not in i]
    reg_ids = [i for i in focus_list if ":" in i]

    with open(chrombert_meta_file) as f:
        meta = json.load(f)

    overlap = []
    not_overlap = []
    cistrome_gsmid_dict = {}

    for i in gsm_ids:
        if i in meta:
            overlap.append(i)
            cistrome_gsmid_dict[i] = i
        else:
            not_overlap.append(i)

    for i in reg_ids:
        if i in meta:
            overlap.append(i)
            cistrome_gsmid_dict[i] = meta[i]  # map factor:cell -> GSM/ENCODE id
        else:
            not_overlap.append(i)

    print("Note: All cistrome names were converted to lowercase for matching.")
    print(
        f"Cistromes count summary - requested: {len(focus_list)}, "
        f"matched in ChromBERT meta: {len(overlap)}, "
        f"not found: {len(not_overlap)}, "
        f"not found cistromes: {not_overlap}"
    )
    print(f"ChromBERT cistromes metas: {chrombert_meta_file.replace('.json', '.tsv')}")
    return overlap, not_overlap, cistrome_gsmid_dict

def overlap_gene_map_region(gene_meta, focus_genes, odir):
    overlap_genes = []
    not_found_genes = []
    gene_to_region_idx = {}  # gene -> np.array of build_region_index

    # gene meta is expected to contain build_region_index column
    if "build_region_index" not in gene_meta.columns:
        raise ValueError("gene_meta.tsv must contain 'build_region_index' column.")

    for g in focus_genes:
        if g.startswith("ensg") or g.startswith("ensmusg"):
            if g in set(gene_meta["gene_id"].tolist()):
                overlap_genes.append(g)
                gene_to_region_idx[g] = gene_meta.loc[gene_meta["gene_id"] == g, "build_region_index"].values
            else:
                not_found_genes.append(g)
        else:
            if g in set(gene_meta["gene_name"].tolist()):
                overlap_genes.append(g)
                gene_to_region_idx[g] = gene_meta.loc[gene_meta["gene_name"] == g, "build_region_index"].values
            else:
                not_found_genes.append(g)

    # save matched gene meta for debugging
    overlap_meta = gene_meta[(gene_meta["gene_id"].isin(overlap_genes)) | (gene_meta["gene_name"].isin(overlap_genes))].copy()
    overlap_meta.to_csv(f"{odir}/overlap_genes_meta.tsv", sep="\t", index=False)

    if len(gene_to_region_idx) == 0:
        raise ValueError("No requested genes matched gene_meta. Nothing to embed.")
    return overlap_genes, not_found_genes, gene_to_region_idx

def chrom_to_int_series(chrom_series: pd.Series, genome: str) -> pd.Series:
    """hg38: 1-22,X=23,Y=24; mm10: 1-19,X=20,Y=21"""
    genome = genome.lower()
    if genome == "hg38":
        x_id, y_id, max_auto = 23, 24, 22
    elif genome == "mm10":
        x_id, y_id, max_auto = 20, 21, 19
    else:
        raise ValueError(f"Genome {genome} not supported for chrom mapping")

    def _map_one(c):
        if pd.isna(c):
            return np.nan
        c = str(c).strip()
        c = c[3:] if c.lower().startswith("chr") else c
        if c.upper() == "X":
            return x_id
        if c.upper() == "Y":
            return y_id
        if re.fullmatch(r"\d+", c):
            v = int(c)
            return v if 1 <= v <= max_auto else np.nan
        return np.nan

    return chrom_series.map(_map_one)


def bw_getSignal_bins(bw, regions: pd.DataFrame,scale:bool=False,name:str="signal"):
    regions = regions.copy()
    with bbi.open(str(bw)) as bwf:
        mtx = bwf.stackup(regions["chrom"], regions["start"], regions["end"], bins=1, missing=0)
        if scale:
            mean= bwf.info["summary"]["mean"]
            mtx = mtx/mean
    df_signal = pd.DataFrame(data=mtx, columns=[name])
    return df_signal


def split_data(df, name, odir):
    columns = ["chrom", "start", "end", "build_region_index", "label"]
    train = df.sample(frac=0.8, random_state=55)
    test = df.drop(train.index).sample(frac=0.5, random_state=55)
    valid = df.drop(train.index).drop(test.index)
    train[columns].to_csv(f"{odir}/train{name}.csv", index=False)
    test[columns].to_csv(f"{odir}/test{name}.csv", index=False)
    valid[columns].to_csv(f"{odir}/valid{name}.csv", index=False)
    
def cal_metrics_regression(preds, labels):
    metrics_pearsonr = tm.PearsonCorrCoef()
    metrics_spearmanr = tm.SpearmanCorrCoef()
    metrics_mse = tm.MeanSquaredError()
    metrics_mae = tm.MeanAbsoluteError()
    metrics_r2 = tm.R2Score()

    score_pearsonr = metrics_pearsonr(preds, labels).item()
    score_spearmanr = metrics_spearmanr(preds, labels).item()
    score_mse = metrics_mse(preds, labels).item()
    score_mae = metrics_mae(preds, labels).item()
    score_r2 = metrics_r2(preds, labels).item()

    metrics_pearsonr.reset()
    metrics_spearmanr.reset()
    metrics_mse.reset()
    metrics_mae.reset()
    metrics_r2.reset()

    metrics = {
        "pearsonr": score_pearsonr,
        "spearmanr": score_spearmanr,
        "mse": score_mse,
        "mae": score_mae,
        "r2": score_r2,
    }
    return metrics

def cal_metrics_binary(preds, labels):
    metrics_auroc = tm.AUROC(task="binary", ignore_index=-1)
    metrics_auprc = tm.AveragePrecision(task="binary", ignore_index=-1)
    metrics_mcc = tm.MatthewsCorrCoef(task="binary", ignore_index=-1)
    metrics_f1 = tm.F1Score(task="binary", ignore_index=-1)
    metrics_precision = tm.Precision(task="binary", ignore_index=-1)
    metrics_recall = tm.Recall(task="binary", ignore_index=-1)

    score_auroc = metrics_auroc(preds, labels).item()
    score_auprc = metrics_auprc(preds, labels).item()
    score_mcc = metrics_mcc(preds, labels).item()
    score_f1 = metrics_f1(preds, labels).item()
    score_precision = metrics_precision(preds, labels).item()
    score_recall = metrics_recall(preds, labels).item()

    metrics_auroc.reset()
    metrics_auprc.reset()
    metrics_mcc.reset()
    metrics_f1.reset()
    metrics_precision.reset()
    metrics_recall.reset()

    metrics = {
        "auroc": score_auroc,
        "auprc": score_auprc,
        "mcc": score_mcc,
        "f1": score_f1,
        "precision": score_precision,
        "recall": score_recall,
    }
    return metrics

def model_eval(train_odir, data_module, model_config, cal_metrics):
    ckpts = glob.glob(f"{train_odir}/**/checkpoints/*.ckpt", recursive=True)
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint found under {train_odir}. Please verify that training completed successfully."
        )
    ft_ckpt = os.path.abspath(max(ckpts, key=os.path.getmtime))

    dc_test = data_module.test_config
    dl_test = dc_test.init_dataloader(batch_size=4)

    model_tuned = model_config.init_model(finetune_ckpt=ft_ckpt, dropout=0).eval().cuda()

    test_preds = []
    test_labels = []
    for batch in dl_test:
        with torch.no_grad():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            preds = model_tuned(batch).cpu()
            test_labels.append(batch["label"].cpu())
            test_preds.append(preds)

    test_preds = torch.cat(test_preds, dim=0).reshape(-1)
    test_labels = torch.cat(test_labels, axis=0).reshape(-1)

    test_metrics = cal_metrics(test_preds, test_labels)
    print(f"ft_ckpt: {ft_ckpt}, test_metrics: {test_metrics}")
    test_metrics['ft_ckpt'] = ft_ckpt
    with open(os.path.join(train_odir, "eval_performance.json"), "w") as f:
        json.dump(test_metrics, f)
    return model_tuned, test_metrics


def model_embedding(train_odir=None, model_config=None, ft_ckpt=None, model_tuned=None):
    if model_tuned is None:
        if ft_ckpt is None:
            ckpts = glob.glob(f"{train_odir}/lightning_logs/**/checkpoints/*.ckpt", recursive=True)
            if not ckpts:
                raise FileNotFoundError(
                    f"No ckpt found under {train_odir}/lightning_logs, Please check you train finished"
                )
            ft_ckpt = os.path.abspath(max(ckpts, key=os.path.getmtime))
        else:
            ft_ckpt = ft_ckpt
            
        model_tuned = model_config.init_model(finetune_ckpt=ft_ckpt, dropout=0).eval().cuda()

    model_emb = model_tuned.get_embedding_manager().cuda()
    return model_emb


def factor_rank(emb1,emb2,regulator,odir):
    cos_sim = [
        cosine_similarity(emb1[i].reshape(1, -1), emb2[i].reshape(1, -1))[0, 0]
        for i in range(emb2.shape[0])
    ]
    cos_sim_df = (
        pd.DataFrame({"factors": regulator, "similarity": cos_sim})
        .sort_values(by="similarity")
        .reset_index(drop=True)
    )
    cos_sim_df = cos_sim_df[cos_sim_df["factors"] != "input"].reset_index(drop=True)
    cos_sim_df["rank"] = cos_sim_df.index + 1
    cos_sim_df.to_csv(f"{odir}/factor_importance_rank.csv", index=False)
    # print(cos_sim_df.head(n=25))
    return cos_sim_df


def set_seed(seed: int):
    """Set seeds for reproducibility while keeping training stochastic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep these settings for speed; set deterministic=True if you want strict determinism.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True