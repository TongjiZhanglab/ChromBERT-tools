import os
import itertools
import re

import click
from types import SimpleNamespace

import pandas as pd

from .utils import resolve_paths, check_files, overlap_regulator_func
from .utils_classfication import prepare_dataset, validate_args
from .embed_regulator import embed_regulator_processed_mean
from .region_function_classification import load_or_train_model
from .interpret_context_dependent_cofactor import context_dependent_cofactor_analysis


def _pair_results_subdir(name_a, name_b, i: int, j: int) -> str:
    """
    Create a filesystem-safe results subdirectory name for one label pair.
    """

    def slug(x) -> str:
        t = str(x).strip()
        t = re.sub(r"[^\w.\-]+", "_", t, flags=re.ASCII)
        t = t.strip("._") or "label"
        return t[:120]

    return f"{i}_{j}_{slug(name_a)}_vs_{slug(name_b)}"

def prepare_dataset_groups(args, files_dict, d_odir):
    
    prepare_dataset(args, files_dict, d_odir)

    # Use the test split to define the region groups that will be compared later.
    # In fast mode, the sampled test set is used; in full mode, the full test set is used.
    if args.mode == "fast":
        test_dataset = pd.read_csv(os.path.join(d_odir, "test_sampled.csv"))
    else:
        test_dataset = pd.read_csv(os.path.join(d_odir, "test.csv"))

    labels_ordered = list(test_dataset["label"].unique())
    if args.function_names and len(args.function_names) == len(labels_ordered):
        display_names = list(args.function_names)
    else:
        display_names = [str(x) for x in labels_ordered]
    if len(labels_ordered) < 2:
        raise ValueError(
            "At least two distinct labels are required for context comparison; "
            f"found labels: {labels_ordered!r}."
        )

    # Save one region file per label so embeddings can be generated independently
    # for each function class.
    region_files = []
    for label in labels_ordered:
        region = test_dataset[test_dataset["label"] == label]
        d_region = f"{d_odir}/region{label}"
        os.makedirs(d_region, exist_ok=True)
        region.to_csv(os.path.join(d_region, "model_input.tsv"), index=False,sep='\t')
        region_file = os.path.join(d_region, "model_input.tsv")
        region_files.append(region_file)
    return region_files, display_names, labels_ordered

def run(args):
    """
    Run context-specific cofactor analysis across region-function classes.

    Workflow overview
    -----------------
    1. Build a labeled dataset from user-provided function BED files.
    2. Load an existing fine-tuned model or train one if needed.
    3. Generate region-level embeddings for each function class.
    4. Compare every label pair to identify context-specific cofactors.

    Output
    ------
    Pairwise comparison results are written under:
        {odir}/results/<label_pair_subdir>/
    """
    # Fill in optional arguments when they are not provided by upstream callers.
    # This keeps the workflow robust even if args is created outside the CLI entry point.
    for attr, default in [
        ("ft_ckpt", None),
        ("predict_file", None),
        ("ignore_regulator", None),
        ("mode", "fast"),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    # At least one function BED is required to define a target region class.
    # If only one class is provided, a background class will be added automatically.
    if not args.function_beds:
        raise ValueError(
            "At least one --function-bed is required. "
            "Repeat --function-bed to define multiple region classes. "
            "If only one BED file is provided, a background class will be added automatically."
        )

    validate_args(args)

    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    # Resolve required ChromBERT resources and verify they exist before starting.
    files_dict = resolve_paths(args)
    check_files(
        files_dict,
        required_keys=[
            "chrombert_region_file",
            "hdf5_file",
            "meta_file",
        ],
    )

    ignore_object = None
    if args.ignore_regulator is not None:
        overlap_ignore, _, _ = overlap_regulator_func(
            args.ignore_regulator, files_dict["chrombert_regulator_file"]
        )
        ignore_object = ";".join(overlap_ignore) if overlap_ignore else None
    ignore = ignore_object is not None
    
    dual_regulator = args.dual_regulator
    overlap_dual_regulator = None
    if dual_regulator is not None:
        overlap_dual_regulator, _, _ = overlap_regulator_func(
            dual_regulator, files_dict["chrombert_regulator_file"]
        )

    print("Step 1/3: Preparing labeled region dataset...")
    d_odir = os.path.join(odir, "dataset")
    os.makedirs(d_odir, exist_ok=True)
    region_files, display_names, labels_ordered = prepare_dataset_groups(args, files_dict, d_odir)
    print("Finished step 1: labeled dataset prepared.")

    print("Step 2/3: Loading or training the model...")
    model_tuned, data_config, train_odir = load_or_train_model(
        args, files_dict, d_odir, ignore, ignore_object
    )
    model_emb = model_tuned.get_embedding_manager().cuda().bfloat16()
    print("Finished step 2: model ready for embedding generation.")
    
    print("Step 3/3: Generating embeddings and running pairwise cofactor comparisons...")
    emb_odir = f"{odir}/emb"
    os.makedirs(emb_odir, exist_ok=True)

    embed_by_label = []
    for idx, region_file in enumerate(region_files):
        tmp_dl = data_config.init_dataloader(supervised_file=region_file)
        embs_pool, regulators = embed_regulator_processed_mean(
            tmp_dl,
            model_emb,
            emb_odir,
            f"region{idx}",
        )
        embed_by_label.append((labels_ordered[idx], display_names[idx], embs_pool))

    for i, j in itertools.combinations(range(len(embed_by_label)), 2):
        lab_i, name_i, pool_i = embed_by_label[i]
        lab_j, name_j, pool_j = embed_by_label[j]
        pair_dir = _pair_results_subdir(name_i, name_j, i, j)

        print(f"Comparing {name_i!r} vs {name_j!r} ...")
        print(f"Results will be written to: results/{pair_dir}/")

        context_dependent_cofactor_analysis(
            args,
            overlap_dual_regulator,
            pool_i,
            pool_j,
            regulators,
            pair_results_subdir=pair_dir,
        )

    print("Finished step 3: all pairwise comparisons completed.")
    print(f"Compared {len(labels_ordered)} classes and generated {len(list(itertools.combinations(range(len(labels_ordered)), 2)))} pairwise results.")
    print(f"Final results are available under: {odir}/results/<label_pair_subdir>/")


@click.command(
    name="predict_regulator_context_cofactors",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--function-bed",
    "function_beds",
    multiple=True,
    required=False,
    default=(),
    help=(
        "BED file(s) defining one functional region class. "
        "Repeat this option to provide multiple classes. "
        "Use ';' to combine multiple BED files into the same class. "
        "If only one class is provided, a background class will be added automatically."
    ),
)
@click.option(
    "--function-mode",
    "function_modes",
    multiple=True,
    type=click.Choice(["and", "or"], case_sensitive=False),
    help=(
        "How to combine multiple BED files within one class: "
        "'and' for intersection, 'or' for union. "
        "Provide one value to reuse for all classes, or repeat to match each --function-bed. "
        "Default: and."
    ),
)
@click.option(
    "--function-name",
    "function_names",
    multiple=True,
    help=(
        "Optional display name for each function class. "
        "Repeat to match each --function-bed. "
        "If not provided, default names such as function_0, function_1, ... are used."
    ),
)
@click.option(
    "--dual-regulator",
    "dual_regulator",
    default=None,
    required=True,
    help=(
        "Regulator(s) of special interest for dual-function analysis. "
        "Use ';' to separate multiple regulators."
    ),
)
@click.option(
    "--ignore-regulator",
    "ignore_regulator",
    default=None,
    help=(
        "Regulator(s) to exclude from analysis. "
        "Use ';' to separate multiple regulators."
    ),
)
@click.option(
    "--odir",
    default="./output",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory for datasets, embeddings, trained models, and pairwise comparison results.",
)
@click.option(
    "--genome",
    default="hg38",
    show_default=True,
    type=click.Choice(["hg38", "mm10"], case_sensitive=False),
    help="Reference genome used by ChromBERT resources.",
)
@click.option(
    "--resolution",
    default="1kb",
    show_default=True,
    type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
    help="Resolution of the ChromBERT representation.",
)
@click.option(
    "--ft-ckpt",
    "ft_ckpt",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help=(
        "Path to a fine-tuned ChromBERT checkpoint. "
        "If provided, the workflow uses this checkpoint instead of training a new model."
    ),
)
@click.option(
    "--threshold",
    "threshold",
    default=0.1,
    show_default=True,
    type=float,
    help="Threshold for the embedding similarity difference between two regions used to identify a regulator as a context-dependent cofactor.",
)
@click.option("--quantile", default=0.95, show_default=True, type=float,
              help="Quantile threshold for cosine similarity edges. Default: 0.95.")
@click.option(
    "--batch-size",
    "batch_size",
    default=4,
    show_default=True,
    type=int,
    help="Batch size used for embedding generation and model inference.",
)
@click.option(
    "--mode",
    default="fast",
    show_default=True,
    type=click.Choice(["fast", "full"], case_sensitive=False),
    help=(
        "Run mode. "
        "'fast' uses a sampled subset of regions for quicker execution, "
        "while 'full' uses all eligible regions."
    ),
)
@click.option(
    "--chrombert-cache-dir",
    "chrombert_cache_dir",
    default=os.path.expanduser("~/.cache/chrombert/data"),
    show_default=True,
    type=click.Path(file_okay=False),
    help=(
        "Directory containing ChromBERT cached resources, such as config files, "
        "annotations, checkpoints, and metadata."
    ),
)
def predict_regulator_context_cofactors(
    function_beds,
    function_modes,
    function_names,
    dual_regulator,
    ignore_regulator,
    threshold,
    quantile,
    odir,
    genome,
    resolution,
    ft_ckpt,
    batch_size,
    mode,
    chrombert_cache_dir,
):
    """
    Identify context-specific cofactors across two or more region-function classes for user-specified dual-functional regulators.

    Typical usage

    Provide two or more functional region classes with --function-bed to define the region groups to be compared.

    Provide --dual-regulator to specify the dual-functional regulator of interest. The workflow then compares cofactor patterns across the defined region classes for that regulator.

    Output

    For every pair of classes, the workflow writes results to:
        {odir}/results/<label_pair_subdir>/
    """
    args = SimpleNamespace(
        function_beds=list(function_beds),
        function_modes=list(function_modes),
        function_names=list(function_names),
        dual_regulator=dual_regulator,
        ignore_regulator=ignore_regulator,
        threshold=threshold,
        quantile=quantile,
        odir=odir,
        genome=genome,
        resolution=resolution,
        ft_ckpt=ft_ckpt,
        batch_size=batch_size,
        mode=str(mode).lower(),
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    predict_regulator_context_cofactors()
    
