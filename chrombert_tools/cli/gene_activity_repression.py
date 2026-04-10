"""Driver factors from gene expression changes (cell state transition)."""
import json
import os
import pickle

import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from chrombert_hf import ChromBERTFTConfig, DatasetConfig

from .utils import resolve_paths, check_files, check_region_file
from .utils import cal_metrics_regression
from .utils import get_model_name
from .utils_train_cell import retry_train


def _semicolon_paths(s):
    if s is None:
        return []
    return [p.strip() for p in str(s).split(";") if p.strip()]


def _read_tpm_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "gene_id" not in df.columns or "tpm" not in df.columns:
        raise ValueError(f"{path}: CSV must contain `gene_id` and `tpm` columns.")
    return df[["gene_id", "tpm"]].copy()


def mean_tpm_by_gene(csv_paths):
    """Average TPM per gene across multiple replicate CSV files.

    Replicates are aligned with ``merge(..., how="inner")`` on ``gene_id``: only
    genes present in every file are kept. Within a file, duplicate ``gene_id``
    rows are averaged first.
    """
    if not csv_paths:
        raise ValueError("At least one TPM CSV path is required.")
    parts = []
    for p in csv_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"TPM file not found: {p}")
        df = _read_tpm_csv(p).groupby("gene_id", as_index=False)["tpm"].mean()
        parts.append(df)
    if len(parts) == 1:
        return parts[0]
    merged = parts[0].rename(columns={"tpm": "tpm_0"})
    for i, df in enumerate(parts[1:], start=1):
        merged = merged.merge(
            df.rename(columns={"tpm": f"tpm_{i}"}),
            on="gene_id",
            how="inner",
        )
    tpm_cols = [f"tpm_{j}" for j in range(len(parts))]
    merged["tpm"] = merged[tpm_cols].mean(axis=1)
    return merged[["gene_id", "tpm"]].copy()


def make_exp_dataset(args, files_dict, data_odir):
    paths1 = _semicolon_paths(args.exp_tpm1)
    paths2 = _semicolon_paths(args.exp_tpm2)
    dual_state = len(paths2) > 0

    if not paths1:
        return args, False

    gene_meta = (
        pd.read_csv(files_dict["gene_meta_tsv"], sep="\t")
        .query("gene_biotype == 'protein_coding'")[
            ["chrom", "start", "end", "build_region_index", "gene_id", "tss"]
        ]
    )

    meta_path = os.path.join(data_odir, "dataset_meta.json")
    if os.path.exists(f"{data_odir}/total.csv"):
        print(f"Expression dataset already exists in {data_odir}")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                args.dual_state = json.load(f).get("dual_state", True)
        else:
            args.dual_state = True
        return args, True

    print("Processing stage 1: prepare expression dataset")
    if dual_state:
        print(
            "  Mode: two states (log1p TPM fold change; per state, genes inner-merged across ';' files then TPM mean)"
        )
    else:
        print(
            "  Mode: state 1 only (label = log1p(mean TPM); genes inner-merged across ';' replicates then TPM mean)"
        )

    tpm1_mean = mean_tpm_by_gene(paths1)
    tpm1_mean["log1p_tpm_1"] = np.log1p(tpm1_mean["tpm"])

    if dual_state:
        tpm2_mean = mean_tpm_by_gene(paths2)
        tpm2_mean["log1p_tpm_2"] = np.log1p(tpm2_mean["tpm"])
        merge_exp = pd.merge(
            tpm1_mean[["gene_id", "log1p_tpm_1"]],
            tpm2_mean[["gene_id", "log1p_tpm_2"]],
            on="gene_id",
            how="inner",
        ).reset_index(drop=True)
        merge_exp["label"] = merge_exp["log1p_tpm_2"] - merge_exp["log1p_tpm_1"]
        if args.direction != "2-1":
            merge_exp["label"] = -merge_exp["label"]

        merge_exp_anno = pd.merge(
            gene_meta,
            merge_exp[["gene_id", "label", "log1p_tpm_1", "log1p_tpm_2"]],
            on="gene_id",
            how="inner",
        )[
            [
                "chrom",
                "start",
                "end",
                "build_region_index",
                "label",
                "gene_id",
                "tss",
                "log1p_tpm_1",
                "log1p_tpm_2",
            ]
        ]
    else:
        merge_exp_anno = pd.merge(
            gene_meta,
            tpm1_mean[["gene_id", "log1p_tpm_1"]],
            on="gene_id",
            how="inner",
        )
        merge_exp_anno["label"] = merge_exp_anno["log1p_tpm_1"]
        merge_exp_anno = merge_exp_anno[
            [
                "chrom",
                "start",
                "end",
                "build_region_index",
                "label",
                "gene_id",
                "tss",
                "log1p_tpm_1",
            ]
        ]

    merge_exp_anno.to_csv(f"{data_odir}/total.csv", index=False)

    train_data = merge_exp_anno.sample(frac=0.8, random_state=55)
    test_data = merge_exp_anno.drop(train_data.index).sample(frac=0.5, random_state=55)
    valid_data = merge_exp_anno.drop(train_data.index).drop(test_data.index)
    train_data.to_csv(f"{data_odir}/train.csv", index=False)
    test_data.to_csv(f"{data_odir}/test.csv", index=False)
    valid_data.to_csv(f"{data_odir}/valid.csv", index=False)

    if dual_state:
        up_data = (
            merge_exp_anno[merge_exp_anno["label"] > 1]
            .sort_values("label", ascending=False)
            .head(1000)
            .reset_index(drop=True)
        )
        tmp = merge_exp_anno.copy()
        tmp["abs_label"] = np.abs(tmp["label"])
        nochange_data = (
            tmp[(tmp["label"] > -0.5) & (tmp["label"] < 0.5)]
            .sort_values("abs_label")
            .reset_index(drop=True)
            .iloc[0:1000]
        )

        up_data.to_csv(f"{data_odir}/up.csv", index=False)
        nochange_data.to_csv(f"{data_odir}/nochange.csv", index=False)

    args.dual_state = dual_state
    with open(meta_path, "w") as f:
        json.dump({"dual_state": dual_state}, f)

    return args, True


def load_train_model_gep(args, files_dict, odir):
    if args.ft_ckpt is not None:
        print(f"Use fine-tuned ChromBERT checkpoint file: {args.ft_ckpt}")
        data_config = DatasetConfig(
            kind="MultiFlankwindowDataset",
            supervised_file=None,
            hdf5_file=files_dict["hdf5_file"],
            batch_size=args.batch_size,
            num_workers=2,
            meta_file=files_dict["meta_file"],
            flank_window=args.flank_window,
        )
        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task="gep",
            dropout=0,
            pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            finetune_ckpt=args.ft_ckpt,
            mtx_mask=files_dict["mtx_mask"],
            gep_flank_window=args.flank_window,
        )
        model_tuned = model_config.init_model().cuda()
        print("Finished stage 2 (loaded checkpoint)")
    else:
        model_tuned, train_try_odir, model_config, data_config = retry_train(
            args,
            files_dict,
            cal_metrics_regression,
            metcic="pearsonr",
            min_threshold=0.2,
            train_kind="regression",
            task="gep",
            odir=odir,
        )
        print("Finished stage 2 (trained)")
    return model_tuned, data_config

def prepare_dataset(args, files_dict, data_odir):
    os.makedirs(data_odir, exist_ok=True)
    args, ok = make_exp_dataset(args, files_dict, data_odir)
    if not ok:
        raise ValueError(
            "Provide --exp-tpm1 (one or more CSV paths separated by ';'). "
            "Optional --exp-tpm2 for two-state fold change; omit for single-state (label = log1p mean TPM)."
        )


# =========================
# prediction (GEP regression)
# =========================


def _resolve_predict_file(args, d_odir):
    """Priority: --predict-file > test split (sampled or full)."""
    predict_file = getattr(args, "predict_file", None)
    if predict_file is not None:
        if not os.path.exists(predict_file):
            raise FileNotFoundError(f"--predict-file not found: {predict_file}")
        return predict_file
    # GEP training (init_datamodule, task="gep") always uses train/test/valid.csv — no *_sampled splits.
    test_file = os.path.join(d_odir, "test.csv")
    if os.path.exists(test_file):
        return test_file
    raise FileNotFoundError(f"No predict file or test split found in {d_odir}")


def predict(args, model_tuned, data_config, files_dict, d_odir):
    """Run GEP regression on regions/genes; save predictions.csv."""
    predict_odir = os.path.join(args.odir, "predict")
    os.makedirs(predict_odir, exist_ok=True)

    src = _resolve_predict_file(args, d_odir)
    check_region_file(src, files_dict, predict_odir)
    model_input = os.path.join(predict_odir, "model_input.tsv")
    print(f"  Predict input: {model_input}")

    data_config.supervised_file = model_input
    dl = data_config.init_dataloader(batch_size=args.batch_size)
    meta_df = pd.read_csv(model_input, sep="\t")

    model_tuned = model_tuned.eval()
    all_pred = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Predicting"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            out = model_tuned(batch).cpu()
            all_pred.append(out.reshape(-1))
            if "label" in batch:
                all_labels.append(batch["label"].cpu().reshape(-1))

    all_pred = torch.cat(all_pred, dim=0).numpy()
    meta_cols = [
        c
        for c in ["chrom", "start", "end", "build_region_index", "gene_id", "tss"]
        if c in meta_df.columns
    ]
    result = meta_df[meta_cols].copy()
    result["predicted_value"] = all_pred
    if all_labels:
        result["true_label"] = torch.cat(all_labels, dim=0).numpy()

    out_path = os.path.join(predict_odir, "predictions.csv")
    result.to_csv(out_path, index=False)
    print(f"  Predictions saved: {out_path}  ({len(result)} rows)")
    return result


def _is_predict_only(args):
    return (
        getattr(args, "ft_ckpt", None) is not None
        and getattr(args, "predict_file", None) is not None
    )


def _load_model_for_predict_gep(args, files_dict):
    """Load fine-tuned GEP model for predict-only (no TPM dataset)."""
    fw = int(getattr(args, "flank_window", 4))
    data_config = DatasetConfig(
        kind="MultiFlankwindowDataset",
        supervised_file=None,
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=2,
        meta_file=files_dict["meta_file"],
        flank_window=fw,
    )
    model_config = ChromBERTFTConfig(
        genome=args.genome,
        task="gep",
        dropout=0,
        pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        finetune_ckpt=args.ft_ckpt,
        mtx_mask=files_dict["mtx_mask"],
        gep_flank_window=fw,
    )
    model_tuned = model_config.init_model().cuda()
    return model_tuned, data_config




def run(args):
    for attr, default in [
        ("ft_ckpt", None),
        ("predict_file", None),
        ("mode", "fast"),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    predict_only = _is_predict_only(args)

    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    required_keys = [
        "chrombert_region_file",
        "hdf5_file",
        "meta_file",
        # "pretrain_ckpt",
        # "mtx_mask",
    ]
    if not predict_only:
        required_keys.append("gene_meta_tsv")
    check_files(files_dict, required_keys=required_keys)

    data_odir = os.path.join(odir, "dataset")
    train_odir = None

    if predict_only:
        print("Predict-only mode (--ft-ckpt + --predict-file)")
        model_tuned, data_config = _load_model_for_predict_gep(args, files_dict)
        print("Loaded GEP checkpoint for prediction")
    else:
        if not _semicolon_paths(args.exp_tpm1):
            raise ValueError(
                "Provide --exp-tpm1 for training, or use predict-only with "
                "both --ft-ckpt and --predict-file."
            )
        print("Stage 1: prepare expression dataset")
        os.makedirs(data_odir, exist_ok=True)
        prepare_dataset(args, files_dict, data_odir)
        print("Finished stage 1")

        os.makedirs(os.path.join(odir, "train"), exist_ok=True)
        if getattr(args, "dual_state", True):
            print("Stage 2: train ChromBERT (expression fold change, two states)")
        else:
            print("Stage 2: train ChromBERT (state-1 expression as log1p TPM)")

        model_tuned, data_config = load_train_model_gep(args, files_dict, odir)
        
    print("Stage 3: Predicting")
    predict(args, model_tuned, data_config, files_dict, data_odir)
    print("Finished stage 3")

    print("\n" + "=" * 60)
    print("All stages completed!")
    print("=" * 60)
    if predict_only:
        print(f"Fine-tuned checkpoint used: {args.ft_ckpt}")
    elif args.ft_ckpt:
        print(f"Fine-tuned checkpoint used: {args.ft_ckpt}")
    elif train_odir is not None:
        print(f"Fine-tuned model saved under: {odir}/train/")
    print(f"Predictions: {odir}/predict/predictions.csv")


@click.command(
    name="gene_activity_repression",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--exp-tpm1",
    "exp_tpm1",
    default=None,
    required=False,
    help="State-1 TPM CSV path(s); use ';' for replicates. Columns: gene_id, tpm. "
         "Replicates are inner-joined on gene_id, then TPM is averaged. "
         "Required for training unless predict-only (--ft-ckpt + --predict-file).",
)
@click.option(
    "--exp-tpm2",
    "exp_tpm2",
    default=None,
    required=False,
    help="Optional state-2 TPM CSV path(s), same format as --exp-tpm1. Omit for single-state (no fold change).",
)
@click.option(
    "--direction",
    default="2-1",
    show_default=True,
    type=click.Choice(["2-1", "1-2"], case_sensitive=False),
    help="Only for two states: flip fold-change sign between 2-1 and 1-2.",
)
@click.option(
    "--predict-file",
    "predict_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="TSV/CSV for prediction (chrom, start, end, build_region_index, gene_id, tss; optional label). "
         "With --ft-ckpt, skips TPM dataset and training. If omitted after training, uses dataset/test.csv.",
)
@click.option(
    "--odir",
    default="./output",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory.",
)
@click.option(
    "--genome",
    default="hg38",
    show_default=True,
    type=click.Choice(["hg38", "mm10"], case_sensitive=False),
    help="Reference genome.",
)
@click.option(
    "--resolution",
    default="1kb",
    show_default=True,
    type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
    help="ChromBERT resolution.",
)
@click.option(
    "--ft-ckpt",
    "ft_ckpt",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Fine-tuned checkpoint (skips training if set).",
)
@click.option(
    "--chrombert-cache-dir",
    "chrombert_cache_dir",
    default="~/.cache/chrombert/data",
    show_default=True,
    type=click.Path(file_okay=False),
    help="ChromBERT cache directory.",
)
@click.option(
    "--batch-size",
    "batch_size",
    default=4,
    show_default=True,
    type=int,
    help="Batch size.",
)
@click.option(
    "--mode",
    default="fast",
    show_default=True,
    type=click.Choice(["fast", "full"], case_sensitive=False),
    help="Reserved for parity with other tools; gene expression splits always use full train/test/valid.csv.",
)
@click.option(
    "--flank-window",
    "flank_window",
    default=4,
    show_default=True,
    type=int,
    help="Flank window size for gene expression prediction.",
)
def gene_activity_repression(
    exp_tpm1,
    exp_tpm2,
    direction,
    predict_file,
    odir,
    genome,
    resolution,
    ft_ckpt,
    chrombert_cache_dir,
    batch_size,
    mode,
    flank_window,
):
    """
    Predict gene expression levels or fold changes between two states.

    Multiple files per state (separated by ';'): inner-merge on gene_id, then mean TPM, then log1p.
    Predict-only: pass both --ft-ckpt and --predict-file to load a GEP checkpoint and
    write {odir}/predict/predictions.csv without TPM inputs or training.
    """
    args = SimpleNamespace(
        exp_tpm1=exp_tpm1,
        exp_tpm2=exp_tpm2,
        direction=direction,
        predict_file=predict_file,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        mode=str(mode).lower(),
        ft_ckpt=ft_ckpt,
        chrombert_cache_dir=chrombert_cache_dir,
        batch_size=batch_size,
        flank_window=flank_window,
    )
    run(args)


if __name__ == "__main__":
    gene_activity_repression()
