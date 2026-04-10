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
from .utils import split_data, bw_getSignal_bins
from .utils import cal_metrics_regression
from .utils import factor_rank, overlap_region, get_model_name
from .utils_train_cell import retry_train


def _semicolon_paths(s):
    """Split ';'-separated paths; strip whitespace; drop empties."""
    if s is None:
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


def _union_peak_overlaps(peak_files, chrombert_region_file, data_odir, tag_prefix):
    """Each BED → overlap with ChromBERT bins; concat all and dedupe by build_region_index."""
    if not peak_files:
        return pd.DataFrame(columns=["chrom", "start", "end", "build_region_index"])
    dfs = []
    base = os.path.join(data_odir, f"{tag_prefix}_peak_overlap")
    for i, bed in enumerate(peak_files):
        if not os.path.isfile(bed):
            raise FileNotFoundError(f"Peak BED not found: {bed}")
        subdir = os.path.join(base, f"rep_{i:03d}")
        os.makedirs(subdir, exist_ok=True)
        df = overlap_region(
            bed, chrombert_region_file, subdir, tag=f"{tag_prefix}_{i}"
        )
        dfs.append(df)
    out = pd.concat(dfs, axis=0).drop_duplicates(subset="build_region_index").reset_index(drop=True)
    print(
        f"  [{tag_prefix}] merged {len(peak_files)} peak file(s) → "
        f"{len(out)} unique ChromBERT regions (after overlap + dedup)"
    )
    return out


def _mean_bw_signals(bw_files, regions, scale=True, col_name="signal"):
    """Mean scaled BigWig signal across replicates at each region."""
    if not bw_files:
        raise ValueError("At least one BigWig path is required.")
    mats = []
    for j, bw in enumerate(bw_files):
        if not os.path.isfile(bw):
            raise FileNotFoundError(f"BigWig not found: {bw}")
        df = bw_getSignal_bins(bw, regions, scale=scale, name=f"_bw{j}")
        mats.append(df.iloc[:, 0].to_numpy())
    stacked = np.column_stack(mats)
    mean_sig = stacked.mean(axis=1)
    return pd.DataFrame({col_name: mean_sig})


def make_acc_dataset(args, files_dict, data_odir):
    acc_peak1_list = _semicolon_paths(args.acc_peak1)
    acc_peak2_list = _semicolon_paths(args.acc_peak2)
    acc_signal1_list = _semicolon_paths(args.acc_signal1)
    acc_signal2_list = _semicolon_paths(args.acc_signal2)
    dual_state = len(acc_signal2_list) > 0
    include_bg = getattr(args, "include_tss_background", False)

    if not acc_peak1_list or not acc_signal1_list:
        return args, False

    if (
        len(acc_peak1_list) > 0
        and len(acc_signal1_list) > 0
        and (not dual_state or len(acc_signal2_list) > 0)
        and (not acc_peak2_list or len(acc_peak2_list) > 0)
    ):
        meta_path = os.path.join(data_odir, "dataset_meta.json")
        if os.path.exists(f"{data_odir}/total.csv"):
            print(f"Chromatin accessibility dataset already exists in {data_odir}")
            if os.path.exists(f"{data_odir}/total_sampled.csv"):
                args.mode = "fast"
            else:
                args.mode = "full"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                args.dual_state = meta.get("dual_state", True)
            else:
                args.dual_state = True
            return args, True

        print("Processing stage 1: prepare chromatin accessibility dataset")
        if dual_state:
            print("  Mode: two states (fold-change label)")
        else:
            print("  Mode: state 1 only (label = log2(1 + cell1 signal))")
        if include_bg:
            tss_flank = getattr(args, "tss_flank", 10000)
            print(f"  TSS ± flank background regions: enabled, flank distance: {tss_flank} bp")
        crf = files_dict["chrombert_region_file"]

        peak1_union = _union_peak_overlaps(acc_peak1_list, crf, data_odir, "state1")
        if acc_peak2_list:
            peak2_union = _union_peak_overlaps(acc_peak2_list, crf, data_odir, "state2")
            total_peak_process = (
                pd.concat([peak1_union, peak2_union], axis=0)
                .drop_duplicates(subset="build_region_index")
                .reset_index(drop=True)
            )
        else:
            total_peak_process = peak1_union

        if include_bg:
            gene_meta = pd.read_csv(files_dict["gene_meta_tsv"], sep="\t").query(
                "gene_biotype == 'protein_coding'"
            )
            gene_tss_region = pd.DataFrame(
                {
                    "chrom": gene_meta["chrom"],
                    "start": gene_meta["tss"] - tss_flank,
                    "end": gene_meta["tss"] + tss_flank,
                }
            )
            gene_tss_bed = f"{data_odir}/gene_tss_{tss_flank // 1000}kb.bed"
            gene_tss_region.to_csv(gene_tss_bed, sep="\t", index=False, header=None)

            tss_overlap_odir = f"{data_odir}/tss_overlap"
            os.makedirs(tss_overlap_odir, exist_ok=True)
            gene_tss_process = overlap_region(
                gene_tss_bed, files_dict["chrombert_region_file"], tss_overlap_odir, tag="tss_bg"
            ).drop_duplicates(subset="build_region_index").reset_index(drop=True)

            total_region_processed = pd.concat(
                [total_peak_process, gene_tss_process], axis=0
            ).drop_duplicates().reset_index(drop=True)
        else:
            total_region_processed = total_peak_process.copy()

        total_region_processed.to_csv(f"{data_odir}/total_region_processed.csv", index=False)

        cell1_signal = _mean_bw_signals(
            acc_signal1_list, total_region_processed, scale=True, col_name="cell1_signal"
        )

        if dual_state:
            cell2_signal = _mean_bw_signals(
                acc_signal2_list, total_region_processed, scale=True, col_name="cell2_signal"
            )
            total_region_signal = pd.concat(
                [total_region_processed, cell1_signal, cell2_signal], axis=1
            )
            total_region_signal["log2_cell1_signal"] = np.log2(1 + total_region_signal["cell1_signal"])
            total_region_signal["log2_cell2_signal"] = np.log2(1 + total_region_signal["cell2_signal"])
            total_region_signal["label"] = (
                total_region_signal["log2_cell2_signal"] - total_region_signal["log2_cell1_signal"]
            )
            if args.direction != "2-1":
                total_region_signal["label"] = -total_region_signal["label"]
        else:
            total_region_signal = pd.concat(
                [total_region_processed, cell1_signal], axis=1
            )
            total_region_signal["log2_cell1_signal"] = np.log2(1 + total_region_signal["cell1_signal"])
            total_region_signal["label"] = total_region_signal["log2_cell1_signal"]

        total_region_signal.to_csv(f"{data_odir}/total.csv", index=False)

        if args.mode == "fast" and len(total_region_signal) > 20000:
            total_region_signal_sampled = (
                total_region_signal.sample(n=20000, random_state=55).reset_index(drop=True)
            )
            total_region_signal_sampled.to_csv(f"{data_odir}/total_sampled.csv", index=False)
            split_data(total_region_signal_sampled, "_sampled", data_odir)
            print(f" total region: {len(total_region_signal)}")
            print(f"  Fast mode: downsampled to ~{20000} regions")
        else:
            args.mode = "full"
            split_data(total_region_signal, "", data_odir)
            print(f"  Full mode: using all {len(total_region_signal)} regions")
        if dual_state:
            up_region = (
                total_region_signal[total_region_signal["label"] > 1]
                .sort_values("label", ascending=False)
                .head(1000)
                .reset_index(drop=True)
            )
            total_region_signal = total_region_signal.copy()
            total_region_signal["abs_label"] = np.abs(total_region_signal["label"])
            nochange_region = (
                total_region_signal.query("cell1_signal > 0 or cell2_signal > 0")
                .query("label <1 and label > -1")
                .sort_values("abs_label")
                .reset_index(drop=True)
                .iloc[0:1000]
            )

            up_region.to_csv(f"{data_odir}/up.csv", index=False)
            nochange_region.to_csv(f"{data_odir}/nochange.csv", index=False)

        args.dual_state = dual_state
        with open(meta_path, "w") as f:
            json.dump(
                {"dual_state": dual_state, "include_tss_background": include_bg},
                f,
            )

        return args, True

    return args, False


def prepare_dataset(args, files_dict, data_odir):
    args, ok = make_acc_dataset(args, files_dict, data_odir)
    if not ok:
        raise ValueError(
            "Required: --acc-peak1 and --acc-signal1 (use ';' for multiple files). "
            "Omit --acc-signal2 for single-state mode (label = log2(1 + state1 signal), no fold change). "
            "With two states, provide --acc-signal2; optional --acc-peak2. "
            "Use --include-tss-background to add TSS±flank background (default: no background)."
        )
    return args

def load_train_model_acc(args, files_dict, odir):
    if args.ft_ckpt is not None:
        print(f"Using fine-tuned checkpoint: {args.ft_ckpt}")
        data_config = DatasetConfig(
            kind="GeneralDataset",
            supervised_file=None,
            hdf5_file=files_dict["hdf5_file"],
            batch_size=args.batch_size,
            num_workers=8,
            meta_file=files_dict["meta_file"],
        )
        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task="general",
            dropout=0,
            pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            finetune_ckpt=args.ft_ckpt,
        )
        model_tuned = model_config.init_model().cuda()
        print("Finished stage 2 (loaded checkpoint)")
    else:
        model_tuned, train_try_odir, _, data_config = retry_train(
            args,
            files_dict,
            cal_metrics_regression,
            metcic="pearsonr",
            min_threshold=0.2,
            train_kind="regression",
            task="general",
            odir=odir,
        )
        train_odir = train_try_odir
        print("Finished stage 2 (trained)")
    return model_tuned, data_config

# =========================
# prediction (regression)
# =========================


def _resolve_predict_file(args, d_odir):
    """Priority: --predict-file > test split (sampled if fast, else full)."""
    predict_file = getattr(args, "predict_file", None)
    if predict_file is not None:
        if not os.path.exists(predict_file):
            raise FileNotFoundError(f"--predict-file not found: {predict_file}")
        return predict_file
    suffix = "_sampled" if args.mode == "fast" else ""
    test_file = os.path.join(d_odir, f"test{suffix}.csv")
    if os.path.exists(test_file):
        return test_file
    raise FileNotFoundError(f"No predict file or test split found in {d_odir}")


def predict(args, model_tuned, data_config, files_dict, d_odir):
    """Run accessibility regression; write {odir}/predict/predictions.csv."""
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
    region_cols = [
        c
        for c in [
            "chrom",
            "start",
            "end",
            "build_region_index",
            "start_input",
            "end_input",
        ]
        if c in meta_df.columns
    ]
    result = meta_df[region_cols].copy()
    result["predicted_value"] = all_pred
    if all_labels:
        result["true_label"] = torch.cat(all_labels, dim=0).numpy()

    out_path = os.path.join(predict_odir, "predictions.csv")
    result.to_csv(out_path, index=False)
    print(f"  Predictions saved: {out_path}  ({len(result)} regions)")
    return result


def _is_predict_only(args):
    return (
        getattr(args, "ft_ckpt", None) is not None
        and getattr(args, "predict_file", None) is not None
    )


def _load_model_for_predict_acc(args, files_dict):
    """Load fine-tuned general/regression checkpoint for predict-only."""
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=None,
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=8,
        meta_file=files_dict["meta_file"],
    )
    model_config = ChromBERTFTConfig(
        genome=args.genome,
        task="general",
        dropout=0,
        dim_output=1,
        pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        mtx_mask=files_dict["mtx_mask"],
        finetune_ckpt=args.ft_ckpt,
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
        # "pretrain_ckpt",
        # "mtx_mask",
    ]
    if not predict_only and getattr(args, "include_tss_background", False):
        required_keys.append("gene_meta_tsv")
    check_files(files_dict, required_keys=required_keys)

    data_odir = os.path.join(odir, "dataset")
    train_odir = None

    if predict_only:
        print("Predict-only mode (--ft-ckpt + --predict-file)")
        model_tuned, data_config = _load_model_for_predict_acc(args, files_dict)
        print("Loaded checkpoint for prediction")
    else:
        if not _semicolon_paths(args.acc_peak1) or not _semicolon_paths(args.acc_signal1):
            raise ValueError(
                "Provide --acc-peak1 and --acc-signal1 for training, or use predict-only with "
                "both --ft-ckpt and --predict-file."
            )
        print("Stage 1: prepare chromatin accessibility dataset")
        os.makedirs(data_odir, exist_ok=True)
        args = prepare_dataset(args, files_dict, data_odir)
        print("Finished stage 1")

        os.makedirs(os.path.join(odir, "train"), exist_ok=True)
        if getattr(args, "dual_state", True):
            print("Stage 2: train ChromBERT (accessibility fold change, two states)")
        else:
            print("Stage 2: train ChromBERT (state-1 accessibility as log2 signal)")
            
        model_tuned, data_config = load_train_model_acc(args, files_dict, odir)

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
    name="region_activity_regression",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--acc-peak1",
    "acc_peak1",
    default=None,
    required=False,
    help="BED peak file(s) for state 1; use ';' for multiple. Required for training unless predict-only "
         "(--ft-ckpt + --predict-file).",
)
@click.option(
    "--acc-peak2",
    "acc_peak2",
    required=False,
    default=None,
    help="Optional BED peak file(s) for state 2 (same ';' / overlap / union as state 1). Omit to use only state-1 peaks in the peak union.",
)
@click.option(
    "--acc-signal1",
    "acc_signal1",
    default=None,
    required=False,
    help="BigWig(s) for state 1; use ';' for replicates. Required for training unless predict-only.",
)
@click.option(
    "--predict-file",
    "predict_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="CSV/TSV for prediction (chrom, start, end, build_region_index; optional label). "
         "With --ft-ckpt, skips dataset prep and training. If omitted after training, uses dataset/test[_sampled].csv.",
)
@click.option(
    "--acc-signal2",
    "acc_signal2",
    default=None,
    required=False,
    help="Optional. BigWig(s) for state 2 (';' for replicates). Omit for single-state mode (predict state-1 log2 signal only).",
)
@click.option(
    "--direction",
    default="2-1",
    show_default=True,
    type=click.Choice(["2-1", "1-2"], case_sensitive=False),
    help="Only used with two states: '2-1' vs '1-2' flips fold-change sign.",
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
    "--mode",
    default="fast",
    show_default=True,
    type=click.Choice(["fast", "full"], case_sensitive=False),
    help="'fast' downsamples to 20k regions; 'full' uses all.",
)
@click.option(
    "--ft-ckpt",
    "ft_ckpt",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Fine-tuned checkpoint (skips training if set).",
)
@click.option(
    "--tss-flank",
    "tss_flank",
    default=10000,
    show_default=True,
    type=int,
    help="Flanking distance (bp) around TSS when --include-tss-background is set.",
)
@click.option(
    "--include-tss-background",
    "include_tss_background",
    is_flag=True,
    default=False,
    help="Add protein-coding TSS±flank bins as background (needs gene_meta). Default: no background.",
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
def region_activity_regression(
    acc_peak1,
    acc_peak2,
    acc_signal1,
    acc_signal2,
    predict_file,
    direction,
    odir,
    genome,
    resolution,
    mode,
    ft_ckpt,
    tss_flank,
    include_tss_background,
    chrombert_cache_dir,
    batch_size,
):
    """
    Predict region activity from chromatin accessibility, or fold changes between two states.

    Multiple files per state (separated by ';') are averaged per region before log2.
    Predict-only: pass both --ft-ckpt and --predict-file to load a checkpoint and write
    {odir}/predict/predictions.csv without peaks/BigWigs or training.
    """
    args = SimpleNamespace(
        acc_peak1=acc_peak1,
        acc_peak2=acc_peak2,
        acc_signal1=acc_signal1,
        acc_signal2=acc_signal2,
        predict_file=predict_file,
        direction=direction,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        mode=str(mode).lower(),
        ft_ckpt=ft_ckpt,
        tss_flank=tss_flank,
        include_tss_background=include_tss_background,
        chrombert_cache_dir=chrombert_cache_dir,
        batch_size=batch_size,
    )
    run(args)


if __name__ == "__main__":
    region_activity_regression()
