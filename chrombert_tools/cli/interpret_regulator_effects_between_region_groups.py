import os
import click
from types import SimpleNamespace
from .utils import resolve_paths, check_files
from .utils import factor_rank, check_region_file
from .utils_interpret import (
    build_interpret_config,
    load_interpret_model,
)
from .embed_regulator import embed_regulator_processed_mean



def regulator_effects_rank(data_config,model_emb,region1_file,region2_file,emb_odir,results_odir):
    dl1 = data_config.init_dataloader(supervised_file=region1_file)
    dl2 = data_config.init_dataloader(supervised_file=region2_file)
    embs_pool_region1, regulators = embed_regulator_processed_mean(dl1, model_emb, emb_odir, "region1")
    embs_pool_region2, regulators = embed_regulator_processed_mean(dl2, model_emb, emb_odir, "region2")
    cos_sim_df = factor_rank(embs_pool_region1, embs_pool_region2, regulators, results_odir)
    return cos_sim_df

def run(args, return_data=True):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    required_keys=[
        "chrombert_region_file",
        "hdf5_file",
        # "pretrain_ckpt",
        # "mtx_mask"
    ]
    check_files(files_dict, required_keys=required_keys)


    # 1) make dataset
    d_odir = f"{odir}/dataset";  os.makedirs(d_odir, exist_ok=True)
    d_region1 = f"{d_odir}/region1";  os.makedirs(d_region1, exist_ok=True)
    d_region2 = f"{d_odir}/region2";  os.makedirs(d_region2, exist_ok=True)
    check_region_file(args.region1_file, files_dict, d_region1)
    check_region_file(args.region2_file, files_dict, d_region2)
    
    region1_file = f"{d_region1}/model_input.tsv"
    region2_file = f"{d_region2}/model_input.tsv"
    # train_odir = f"{odir}/train"; os.makedirs(train_odir, exist_ok=True)
    results_odir = f"{odir}/results"; os.makedirs(results_odir, exist_ok=True)
    emb_odir = f"{odir}/emb"; os.makedirs(emb_odir, exist_ok=True)


    # 2) load model (shared with interpret_regulator_interactions)
    data_config, model_config = build_interpret_config(
        args, files_dict, region1_file
    )
    _, model_emb = load_interpret_model(model_config)

    # 3) key regulators across regions
    cos_sim_df = regulator_effects_rank(data_config, model_emb, region1_file, region2_file, emb_odir, results_odir)
    print("Identify key regulators across regions(top 25)")
    print(cos_sim_df.head(n=25))

    print("Finished ")
    if args.ft_ckpt is not None:
        print(f"Used fine-tuned ChromBERT checkpoint: {args.ft_ckpt}")
    else:
        print(f"Used pre-trained ChromBERT")
    print(f"Key regulators across regions saved to: {results_odir}/factor_importance_rank.csv")

    if return_data:
        return cos_sim_df
    return None


@click.command(name="interpret_regulator_effects_between_region_groups", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region1-file", "region1_file",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region 1 file.")
@click.option("--region2-file", "region2_file",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region 2 file.")
@click.option("--ft-ckpt", "ft_ckpt",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, default=None, show_default=True,
              help="Fine-tuned ChromBERT checkpoint. If provided, using this ckpt to generate embeddings.")
@click.option("--ignore-regulator", "ignore_regulator",
              type=str,
              required=False, default=None, show_default=True,
              help="Ignore regulator. Use ';' to separate multiple regulators.")
@click.option("--gep", "gep", is_flag=True, default=False, show_default=True,
              help="Use GEP model (multi-flank-window). Default: False.")
@click.option("--flank-window", "flank_window",
              type=int,
              required=False, default=4, show_default=True,
              help="Flank window size for gep model.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False),
              help="Reference genome (hg38 or mm10).")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
              help="ChromBERT resolution.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--batch-size", "batch_size", default=4, show_default=True, type=int,
              help="Batch size.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default="~/.cache/chrombert/data",
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache directory (contains config/ anno/ checkpoint/ etc).")


def interpret_regulator_effects_between_region_groups(region1_file, region2_file, ft_ckpt, ignore_regulator, gep, flank_window, genome, resolution, odir, batch_size,
                   chrombert_cache_dir):
    '''
    Identify regulators that differ between two region sets via embedding shift.
    ''' 
    args = SimpleNamespace(
        region1_file=region1_file,
        region2_file=region2_file,
        ft_ckpt=ft_ckpt,
        ignore_regulator=ignore_regulator,
        gep=gep,
        flank_window=flank_window,
        genome=genome,
        resolution=resolution,
        odir=odir,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args, return_data=False)


if __name__ == "__main__":
    interpret_regulator_effects_between_region_groups()