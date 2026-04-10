import os
import glob
import click
from types import SimpleNamespace

from .utils import resolve_paths, check_files
from .utils_embed import build_cell_model_emb
from .utils_interpret import build_interpret_config
from .interpret_regulator_effects_between_region_groups import regulator_effects_rank


def run(args):
    """
    Run key-regulator interpretation for a cell type or for two user-defined region sets.

    This command supports two main usage modes:

    Provide --cell-type-bw and --cell-type-peak.
    The workflow will build or load a cell-specific model, define region groups
    from the cell-type accessibility data, and rank candidate key regulators.

    a) If --ft-ckpt is also provided,
        the workflow will use the checkpoint together with the cell-type
        accessibility data to identify region groups automatically and then rank
        candidate key regulators.

    b) If --ft-ckpt is not provided,
        the workflow will build a cell-specific model, define region groups
        from the cell-type accessibility data, and rank candidate key regulators.

    Output
    ------
    Ranked candidate regulators are saved to:
        {odir}/results/factor_importance_rank.csv
    """
    odir = args.odir
    os.makedirs(odir, exist_ok=True)


    if args.ft_ckpt is None and (args.cell_type_bw is None or args.cell_type_peak is None):
        raise ValueError(
            "Please provide either --ft-ckpt, or both --cell-type-bw and --cell-type-peak."
        )
        
    files_dict = resolve_paths(args)
    required_keys = [
        "chrombert_region_file",
        "hdf5_file",
    ]
    check_files(files_dict, required_keys=required_keys)

    d_odir = f"{odir}/dataset"; os.makedirs(d_odir, exist_ok=True)
    train_odir = f"{odir}/train"; os.makedirs(train_odir, exist_ok=True)
    results_odir = f"{odir}/results"; os.makedirs(results_odir, exist_ok=True)
    emb_odir = f"{odir}/emb"; os.makedirs(emb_odir, exist_ok=True)

    print("Step 1/3: Building or loading a cell-specific model...")
    model_emb, _ = build_cell_model_emb(args, files_dict, odir)

    print("Step 2/3: Preparing region groups for interpretation...")
    data_config, _ = build_interpret_config(
        args, files_dict, f"{d_odir}/highly_accessible_region.csv"
    )

    print("Step 3/3: Ranking candidate key regulators...")
    cos_sim_df = regulator_effects_rank(data_config, model_emb, f"{d_odir}/highly_accessible_region.csv", f"{d_odir}/background_region.csv", emb_odir, results_odir)
    print("Top 25 candidate regulators:")
    print(cos_sim_df.head(n=25))
    print("Analysis finished.")

    if args.ft_ckpt is not None:
        print(f"Checkpoint used for interpretation: {args.ft_ckpt}")
    else:
        ckpts = glob.glob(f"{train_odir}/**/checkpoints/*.ckpt", recursive=True)
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoint found under {train_odir}/**/checkpoints/*.ckpt after training."
            )
        ft_ckpt = os.path.abspath(max(ckpts, key=os.path.getmtime))
        print(f"Checkpoint used for interpretation: {ft_ckpt}")

    print(f"Ranked regulators saved to: {results_odir}/factor_importance_rank.csv")


@click.command(
    name="predict_cell_type_master_regulators",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--cell-type-bw",
    "cell_type_bw",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help=(
        "BigWig file of chromatin accessibility for the target cell type. "
        "Use this together with --cell-type-peak when you do not already have "
        "a fine-tuned checkpoint."
    ),
)
@click.option(
    "--cell-type-peak",
    "cell_type_peak",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help=(
        "BED file of accessible peaks for the target cell type. "
        "Use this together with --cell-type-bw."
    ),
)
@click.option(
    "--ft-ckpt",
    "ft_ckpt",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    default=None,
    show_default=True,
    help=(
        "Path to a fine-tuned ChromBERT checkpoint. "
        "If provided, the workflow skips model fine-tuning and uses this checkpoint directly."
    ),
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
    "--odir",
    default="./output",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory for intermediate files, embeddings, and final ranking results.",
)
@click.option(
    "--mode",
    default="fast",
    show_default=True,
    type=click.Choice(["fast", "full"], case_sensitive=False),
    help=(
        "Run mode. "
        "'fast' uses a subset of regions for quicker execution, while "
        "'full' uses all eligible regions."
    ),
)
@click.option(
    "--batch-size",
    "batch_size",
    default=4,
    show_default=True,
    type=int,
    help="Batch size used for embedding generation and model inference.",
)
@click.option(
    "--chrombert-cache-dir",
    "chrombert_cache_dir",
    default="~/.cache/chrombert/data",
    show_default=True,
    type=click.Path(file_okay=False),
    help=(
        "Directory containing ChromBERT cached resources, such as configs, "
        "annotations, checkpoints, and reference embeddings."
    ),
)
def predict_cell_type_master_regulators(
    cell_type_bw,
    cell_type_peak,
    ft_ckpt,
    odir,
    mode,
    batch_size,
    chrombert_cache_dir,
    genome,
    resolution,
):
    """
    Find candidate key regulators for a target cell type or between two region sets.

    Typical usage

    Provide:
        --cell-type-bw
        --cell-type-peak

        a) If --ft-ckpt is provided,
        The workflow will use the checkpoint together with the cell-type accessibility data to identify region groups automatically and then rank candidate key regulators.

        b) If --ft-ckpt is not provided,
        The workflow will build a cell-specific model, define region groups from the cell-type accessibility data and then rank candidate key regulators.
    """
    args = SimpleNamespace(
        cell_type_bw=cell_type_bw,
        cell_type_peak=cell_type_peak,
        ft_ckpt=ft_ckpt,
        genome=genome,
        resolution=resolution,
        odir=odir,
        mode=mode,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    predict_cell_type_master_regulators()