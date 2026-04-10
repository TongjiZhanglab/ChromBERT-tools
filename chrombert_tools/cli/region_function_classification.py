import os
import pickle
import itertools
from collections import defaultdict

import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import networkx as nx
import nxviz as nv
from nxviz import annotate
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from chrombert_hf import ChromBERTFTConfig, DatasetConfig, ChromBERTConfig

from .utils import resolve_paths, check_files, overlap_regulator_func, overlap_region
from .utils import split_data, cal_metrics_binary, cal_metrics_multiclass, get_model_name,check_region_file
from .utils_train_cell import retry_train
from .utils_classfication import validate_args, prepare_dataset


# =========================
# model training / loading
# =========================

def load_or_train_model(args, files_dict, d_odir, ignore, ignore_object):
    """Load fine-tuned model or train from scratch."""
    ignore_index = None
    train_odir = None
    n_classes = len(args.function_names)

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
        if ignore:
            data_config.ignore = ignore
            data_config.ignore_object = ignore_object
            ds = data_config.init_dataset(
                supervised_file=os.path.join(d_odir, "total.csv")
            )
            ignore_index = ds[0]["ignore_index"]

        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task="general",
            dropout=0,
            dim_output=n_classes if n_classes > 2 else 1,
            pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            finetune_ckpt=args.ft_ckpt,
            ignore=ignore,
            ignore_index=ignore_index,
        )
        model_tuned = model_config.init_model().cuda()
        return model_tuned, data_config, train_odir

    # No ft_ckpt → train classifier
    if n_classes > 2:
        print(f"Stage 2: Fine-tuning (multiclass, {n_classes} classes)")
        cal_metrics = cal_metrics_multiclass
        train_kind = "multiclass"
        metcic = "macro_f1"
        dim_output = n_classes
    else:
        print("Stage 2: Fine-tuning (binary classification)")
        cal_metrics = cal_metrics_binary
        train_kind = "classification"
        metcic = "auprc"
        dim_output = 1

    model_tuned, train_odir, model_config, data_config = retry_train(
        args, files_dict, cal_metrics,
        metcic=metcic, min_threshold=0.2,
        train_kind=train_kind,
        ignore_object=ignore_object,
        dim_output=dim_output,
    )
    return model_tuned, data_config, train_odir



# =========================
# prediction
# =========================

def _resolve_predict_file(args, d_odir):
    """Return the path to the prediction input file.

    Priority: --predict-file > test split (sampled or full).
    """
    predict_file = getattr(args, "predict_file", None)
    if predict_file is not None:
        if not os.path.exists(predict_file):
            raise FileNotFoundError(f"--predict-file not found: {predict_file}")
        return predict_file

    suffix = "_sampled" if args.mode == "fast" else ""
    test_file = os.path.join(d_odir, f"test{suffix}.csv")
    if os.path.exists(test_file):
        return test_file
    raise FileNotFoundError(f"No predict file or test file found in {d_odir}")


def predict(args, model_tuned, data_config, files_dict,d_odir):
    """Run prediction, save probabilities and predicted labels."""
    n_classes = len(args.function_names)
    names = args.function_names
    predict_odir = os.path.join(args.odir, "predict")
    os.makedirs(predict_odir, exist_ok=True)

    predict_file = _resolve_predict_file(args, d_odir)
    check_region_file(predict_file,files_dict,predict_odir)
    predict_file = os.path.join(predict_odir, "model_input.tsv")
    print(f"  Predict input: {predict_file}")

    
    data_config.supervised_file = predict_file
    dl = data_config.init_dataloader(batch_size=args.batch_size)
    meta_df = pd.read_csv(predict_file,sep="\t")

    model_tuned = model_tuned.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Predicting"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            logits = model_tuned(batch).cpu()
            all_logits.append(logits)
            if "label" in batch:
                all_labels.append(batch["label"].cpu())
                all_label_state = True
            else:
                all_label_state = False

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).reshape(-1) if all_label_state else None

    # Build result DataFrame starting from region metadata
    region_cols = [c for c in ["chrom", "start", "end", "build_region_index"] if c in meta_df.columns]
    result = meta_df[region_cols].copy()
    if all_label_state:
        result["true_label"] = all_labels.numpy() 

    if n_classes > 2:
        # multiclass: softmax → per-class probabilities
        probs = torch.softmax(all_logits.view(-1, n_classes), dim=-1).numpy()
        for i, name in enumerate(names):
            result[f"prob_{name}"] = probs[:, i]
        result["predicted_label"] = np.argmax(probs, axis=1)
    else:
        # binary: sigmoid → single probability
        probs = torch.sigmoid(all_logits.reshape(-1)).numpy()
        result[f"prob_{names[0]}"] = probs
        result["predicted_label"] = (probs > 0.5).astype(int)

    result["predicted_name"] = result["predicted_label"].map(
        {i: n for i, n in enumerate(names)}
    )

    out_path = os.path.join(predict_odir, "predictions.csv")
    result.to_csv(out_path, index=False)
    print(f"  Predictions saved: {out_path}  ({len(result)} regions)")

    return result


# =========================
# report
# =========================

def report(args, train_odir):
    n_classes = len(args.function_names)
    names = args.function_names

    print("\n" + "=" * 60)
    print("All stages completed!")
    print("=" * 60)

    if args.ft_ckpt is not None:
        print(f"Fine-tuned checkpoint used: {args.ft_ckpt}")
    elif train_odir is not None:
        print(f"Fine-tuned model saved in: {train_odir}")

    print(f"Classes ({n_classes}): {', '.join(names)}")
    print(f"Predictions: {args.odir}/predict/predictions.csv")

# =========================
# main entry
# =========================

def _is_predict_only(args):
    """Predict-only mode: both ft_ckpt and predict_file are provided."""
    return (
        getattr(args, "ft_ckpt", None) is not None
        and getattr(args, "predict_file", None) is not None
    )


def _load_model_for_predict(args, files_dict, ignore=False, ignore_object=None):
    """Load a fine-tuned model for predict-only mode (no training data needed)."""
    n_classes = len(args.function_names)
    ignore_index = None

    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=None,
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=8,
        meta_file=files_dict["meta_file"],
    )

    if ignore:
        data_config.ignore = ignore
        data_config.ignore_object = ignore_object
        ds = data_config.init_dataset(supervised_file=args.predict_file)
        ignore_index = ds[0]["ignore_index"]

    model_config = ChromBERTFTConfig(
        genome=args.genome,
        task="general",
        dropout=0,
        dim_output=n_classes if n_classes > 2 else 1,
        pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        mtx_mask=files_dict["mtx_mask"],
        finetune_ckpt=args.ft_ckpt,
        ignore=ignore,
        ignore_index=ignore_index,
    )
    model_tuned = model_config.init_model().cuda()
    return model_tuned, data_config


def run(args):
    # Backward compatibility for API callers
    for attr, default in [
        ("ft_ckpt", None),
        ("predict_file", None),
        ("ignore_regulator", None),
        ("mode", "fast"),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    predict_only = _is_predict_only(args)

    if predict_only:
        # Predict-only: require function_names, skip function_beds validation
        if not args.function_names:
            raise ValueError(
                "--function-name is required in predict-only mode "
                "(when both --ft-ckpt and --predict-file are provided)."
            )
        args.function_names = list(args.function_names)
    else:
        if not args.function_beds:
            raise ValueError(
                "--function-bed is required unless both --ft-ckpt and "
                "--predict-file are provided for predict-only mode."
            )
        validate_args(args)

    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    check_files(files_dict, required_keys=[
        "chrombert_region_file", "hdf5_file", "meta_file",
    ])

    # Ignore regulator (needed for both predict-only and full mode)
    ignore_object = None
    if args.ignore_regulator is not None:
        overlap_ignore, _, _ = overlap_regulator_func(
            args.ignore_regulator, files_dict["chrombert_regulator_file"]
        )
        ignore_object = ";".join(overlap_ignore) if overlap_ignore else None
    ignore = ignore_object is not None

    if predict_only:
        # Skip Stage 1 & 2, directly load model and predict
        print("Predict-only mode (--ft-ckpt + --predict-file)")
        model_tuned, data_config = _load_model_for_predict(
            args, files_dict, ignore=ignore, ignore_object=ignore_object
        )
        train_odir = None
        d_odir = os.path.join(odir, "dataset")
    else:
        # Stage 1: Dataset
        print("Stage 1: Preparing dataset")
        d_odir = os.path.join(odir, "dataset")
        os.makedirs(d_odir, exist_ok=True)
        prepare_dataset(args, files_dict, d_odir)
        print("Finished stage 1")

        # Stage 2: Model
        model_tuned, data_config, train_odir = load_or_train_model(
            args, files_dict, d_odir, ignore, ignore_object
        )
        print("Finished stage 2")

    # Stage 3: Predict
    print("Stage 3: Predicting")
    predict(args, model_tuned, data_config, files_dict, d_odir)
    print("Finished stage 3")

    report(args, train_odir)

# =========================
# CLI
# =========================

@click.command(
    name="region_function_classification",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--function-bed", "function_beds",
    multiple=True, required=False, default=(),
    help="Function BED file(s) for one class. Repeat for each class. "
         "Use ';' to combine multiple BED files within one class. "
         "Not required when both --ft-ckpt and --predict-file are provided.",
)
@click.option(
    "--function-mode", "function_modes",
    multiple=True,
    type=click.Choice(["and", "or"], case_sensitive=False),
    help="Merge mode per class ('and'=intersection, 'or'=union). "
         "Provide once to apply to all, or repeat to match each --function-bed. "
         "Default: 'and'.",
)
@click.option(
    "--function-name", "function_names",
    multiple=True,
    help="Name per class. Repeat to match each --function-bed. "
         "Default: function_0, function_1, ...",
)

@click.option(
    "--predict-file", "predict_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="CSV/TSV file for prediction (must contain chrom, start, end, build_region_index, label). "
         "If not provided, the test split is used.",
)
@click.option(
    "--ignore-regulator", "ignore_regulator",
    default=None,
    help="Regulators to ignore during analysis. Use ';' to separate.",
)
@click.option(
    "--odir",
    default="./output", show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory.",
)
@click.option(
    "--genome",
    default="hg38", show_default=True,
    type=click.Choice(["hg38", "mm10"], case_sensitive=False),
    help="Genome version.",
)
@click.option(
    "--resolution",
    default="1kb", show_default=True,
    type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
    help="ChromBERT resolution.",
)
@click.option(
    "--ft-ckpt", "ft_ckpt",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Fine-tuned ChromBERT checkpoint. Skips training if provided.",
)
@click.option(
    "--batch-size", "batch_size",
    default=4, show_default=True, type=int,
    help="Batch size.",
)
@click.option(
    "--mode",
    default="fast", show_default=True,
    type=click.Choice(["fast", "full"], case_sensitive=False),
    help="'fast' downsamples to ~20k regions; 'full' uses all.",
)
@click.option(
    "--chrombert-cache-dir", "chrombert_cache_dir",
    default="~/.cache/chrombert/data", show_default=True,
    type=click.Path(file_okay=False),
    help="ChromBERT cache directory.",
)
def region_function_classification(
    function_beds,
    function_modes,
    function_names,
    predict_file,
    ignore_regulator,
    odir,
    genome,
    resolution,
    ft_ckpt,
    batch_size,
    mode,
    chrombert_cache_dir,
):
    """
    Classify genomic regions into N function classes.

    Each --function-bed defines one class. Supports 1-class (auto background),
    2-class (binary), and N-class (multiclass, N>=3).
    """
    args = SimpleNamespace(
        function_beds=function_beds,
        function_modes=list(function_modes),
        function_names=list(function_names),
        predict_file=predict_file,
        ignore_regulator=ignore_regulator,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        ft_ckpt=ft_ckpt,
        batch_size=batch_size,
        mode=str(mode).lower(),
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    region_function_classification()



    # \b
    # Workflow:
    #   Stage 1  Prepare dataset (overlap with ChromBERT regions, split)
    #   Stage 2  Fine-tune ChromBERT (or load --ft-ckpt)
    #   Stage 3  Predict probabilities & labels on test or --predict-file

    # \b
    # Example (1-class, background auto-generated):
    #   chrombert-tools predict_region_function_label \\
    #       --function-bed enhancer.bed \\
    #       --function-name enhancer

    # \b
    # Example (2-class):
    #   chrombert-tools predict_region_function_label \\
    #       --function-bed enhancer.bed \\
    #       --function-bed promoter.bed \\
    #       --function-name enhancer --function-name promoter

    # \b
    # Example (3-class, with multi-BED merge):
    #   chrombert-tools predict_region_function_label \\
    #       --function-bed "enhancer1.bed;enhancer2.bed" \\
    #       --function-bed promoter.bed \\
    #       --function-bed silencer.bed \\
    #       --function-mode and --function-mode or --function-mode and \\
    #       --function-name enhancer --function-name promoter --function-name silencer

    # \b
    # Example (predict on custom file with existing checkpoint):
    #   chrombert-tools predict_region_function_label \\
    #       --function-bed enhancer.bed --function-bed promoter.bed \\
    #       --ft-ckpt model.ckpt \\
    #       --predict-file my_regions.cpa

    # \b
    # Example (predict-only, no training data needed):
    #   chrombert-tools predict_region_function_label \\
    #       --ft-ckpt model.ckpt \\
    #       --predict-file my_regions.csv \\
    #       --function-name enhancer --function-name promoter

    # \b
    # Output:
    #   {odir}/dataset/          Train/test/valid splits (skipped in predict-only)
    #   {odir}/train/            Fine-tuned model checkpoints (skipped in predict-only)
    #   {odir}/predict/          predictions.csv (prob, predicted_label)