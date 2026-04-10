import os
from types import SimpleNamespace
import click
from .utils import (
    resolve_paths,
    check_files,
    check_region_file,
    chrom_to_int_series,
    overlap_regulator_func,
)
from .utils_embed import (
    is_cell_specific, 
    get_required_keys, 
    build_cell_model_emb, 
    build_model_dataset_config,
    generate_regulator_embeddings,
    generate_regulator_embeddings_only_mean
)


# =========================
# regulator embedding
# =========================

def prepare_region_and_regulator(args, files_dict, odir):
    overlap_bed = check_region_file(args.region, files_dict, odir)

    first_chrom = str(overlap_bed["chrom"].iloc[0])
    if "chr" in first_chrom.lower():
        overlap_bed["chrom"] = chrom_to_int_series(overlap_bed["chrom"].astype(str), args.genome)
    overlap_bed = overlap_bed.dropna(subset=["chrom"]).copy()
    overlap_bed["chrom"] = overlap_bed["chrom"].astype(int)
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)

    _, _, regulator_idx_dict = overlap_regulator_func(args.regulator, files_dict["chrombert_regulator_file"])
    if len(regulator_idx_dict) == 0:
        raise ValueError("No requested regulators matched ChromBERT regulator list. Nothing to embed.")

    return overlap_bed, regulator_idx_dict

def validate_args(args):
    '''
    Validate arguments
    '''
    if args.region is None:
        raise ValueError("You must provide --region.")
    
    if args.regulator is None:
        raise ValueError("You must provide --regulator.")

    cell_mode = is_cell_specific(args)
    # cell-specific mode check
    if cell_mode:
        if args.ft_ckpt is None and (args.cell_type_bw is None or args.cell_type_peak is None):
            raise ValueError(
                "For cell-specific embedding, provide either --ft-ckpt "
                "or both --cell-type-bw and --cell-type-peak."
            )
                        
def embed_regulator_processed_mean(dl,model_emb,odir,oname):
    reg_means, regulators = generate_regulator_embeddings_only_mean(dl, model_emb, odir, oname)
    return reg_means, regulators

def run_regulator_general(args, files_dict, odir, return_data=False):
    '''
    Generate regulator embeddings for pretrained model
    '''
    overlap_bed, regulator_idx_dict = prepare_region_and_regulator(args, files_dict, odir)
    sup = f"{odir}/model_input.tsv"
    data_config, model_config = build_model_dataset_config(
        args, files_dict, supervised_file_for_ignore_idx=sup
    )
    model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
    ds = data_config.init_dataset(supervised_file=sup)
    dl = data_config.init_dataloader(batch_size=args.batch_size, supervised_file=sup)
    reg_means, reg_emb_dict = generate_regulator_embeddings(
        ds, dl, model_emb, overlap_bed, regulator_idx_dict, odir, args.oname, return_data
    )
    report_regulator(args, odir, overlap_bed, cell_specific=False)
    if return_data:
        return reg_means, reg_emb_dict, overlap_bed


def run_regulator_cell(args, files_dict, odir, return_data=False):
    '''
    Generate regulator embeddings for cell-type-specific model
    '''
    overlap_bed, regulator_idx_dict = prepare_region_and_regulator(args, files_dict, odir)
    model_emb, data_config = build_cell_model_emb(args, files_dict, odir)
    sup = f"{odir}/model_input.tsv"
    ds = data_config.init_dataset(supervised_file=sup)
    dl = data_config.init_dataloader(batch_size=args.batch_size, supervised_file=sup)
    reg_means, reg_emb_dict = generate_regulator_embeddings(
        ds, dl, model_emb, overlap_bed, regulator_idx_dict, odir, args.oname, return_data
    )
    report_regulator(args, odir, overlap_bed, cell_specific=True)
    if return_data:
        return reg_means, reg_emb_dict, overlap_bed


# =========================
# report
# =========================

def report_regulator(args, odir, overlap_bed, cell_specific=False):
    total_focus = sum(1 for _ in open(args.region))
    no_overlap_region_len = (
        sum(1 for _ in open(f"{odir}/no_overlap_region.bed"))
        if os.path.exists(f"{odir}/no_overlap_region.bed")
        else 0
    )

    print("\nFinished!")
    print(
        f"Focus region summary - total: {total_focus}, "
        f"overlapping with ChromBERT: {overlap_bed.shape[0]}, "
        f"non-overlapping: {no_overlap_region_len}"
    )
    print("Overlapping regions BED file:", f"{odir}/overlap_region.bed")
    print("Non-overlapping regions BED file:", f"{odir}/no_overlap_region.bed")
    print("Mean regulator embeddings saved to:", f"{odir}/mean_{args.oname}.pkl")
    print("Region-aware regulator embeddings saved to:", f"{odir}/region_aware_{args.oname}.hdf5")
    print("Embedding type:", "cell-specific" if cell_specific else "general")


def run(args, return_data=False):
    # Backward compatibility for API caller (which may not provide these attrs)
    for attr, default in [
        ("cell_type_bw", None),
        ("cell_type_peak", None),
        ("ft_ckpt", None),
        ("mode", "fast"),
        ("num_workers", 8),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    validate_args(args)
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    check_files(files_dict, required_keys=get_required_keys(args))

    cell_mode = is_cell_specific(args)
    if cell_mode:
        return run_regulator_cell(args, files_dict, odir, return_data=return_data)
    return run_regulator_general(args, files_dict, odir, return_data=return_data)


@click.command(name="embed_regulator", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region file.")
@click.option("--regulator", required=True,
              help="Regulators of interest, e.g. EZH2 or EZH2;BRD4. Use ';' to separate multiple regulators.")
@click.option("--cell-type-bw", "cell_type_bw",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False,
              help="Cell type accessibility BigWig file. Used for cell-specific mode.")
@click.option("--cell-type-peak", "cell_type_peak",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False,
              help="Cell type accessibility Peak BED file. Used for cell-specific mode.")
@click.option("--ft-ckpt", "ft_ckpt",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, default=None, show_default=True,
              help="Fine-tuned checkpoint. If provided, use cell-specific model and skip fine-tuning.")
# @click.option("--ignore-regulator", "ignore_regulator",
#               type=str,
#               required=False, default=None, show_default=True,
#               help="Ignore regulator. Use ';' to separate multiple regulators.")
# @click.option("--gep", "gep", is_flag=True, default=False, show_default=True,
#               help="Use GEP model (multi-flank-window). Default: False.")
# @click.option("--flank-window", "flank_window",
#               type=int,
#               required=False, default=4, show_default=True,
#               help="Flank window size for gep model.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--oname", default="regulator_emb", show_default=True,
              type=str, 
              help="Output name of the regulator embeddings.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False), help="Resolution.")
@click.option("--mode", default="fast", show_default=True,
              type=click.Choice(["fast", "full"], case_sensitive=False),
              help="Used when training cell-specific model.")
@click.option("--batch-size", default=4, show_default=True, type=int, help="Batch size.")
@click.option("--num-workers", default=8, show_default=True, type=int, help="Dataloader workers.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default="~/.cache/chrombert/data",
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache dir (contains config/ checkpoint/ etc).")

def embed_regulator(
    region,
    regulator,
    cell_type_bw,
    cell_type_peak,
    ft_ckpt,
    # ignore_regulator,
    # gep,
    # flank_window,
    odir,
    oname,
    genome,
    resolution,
    mode,
    batch_size,
    num_workers,
    chrombert_cache_dir,
):
    '''
    Extract regulator embeddings on specified regions.
    Supports both general and cell-specific modes.
    '''
    args = SimpleNamespace(
        region=region,
        regulator=regulator,
        cell_type_bw=cell_type_bw,
        cell_type_peak=cell_type_peak,
        ft_ckpt=ft_ckpt,
        # ignore_regulator=ignore_regulator,
        # gep=gep,
        # flank_window=flank_window,
        odir=odir,
        oname=oname,
        genome=genome.lower(),
        resolution=resolution,
        mode=mode,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    embed_regulator()
