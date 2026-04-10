import os
import pickle
from types import SimpleNamespace
import click
import numpy as np
import pandas as pd
from chrombert_hf.download_data import download
from .utils import (
    resolve_paths,
    check_files,
    check_region_file,
    overlap_gene_map_region,
)
from .utils_embed import (
    is_cell_specific, 
    get_required_keys, 
    build_cell_model_emb, 
    generate_embeddings, 
    build_model_dataset_config
)

# =========================
# embed utils
# =========================

def parse_focus_genes(gene_str: str):
    '''
    Parse focus genes from gene string
    '''
    return [g.strip().lower() for g in gene_str.split(";") if g.strip()]

def validate_args(args):
    '''
    Validate arguments
    '''
    if args.region is None and args.gene is None:
        raise ValueError("You must provide at least one of --region or --gene.")

    cell_mode = is_cell_specific(args)
    # cell-specific mode check
    if cell_mode:
        if args.ft_ckpt is None and (args.cell_type_bw is None or args.cell_type_peak is None):
            raise ValueError(
                "For cell-specific embedding, provide either --ft-ckpt "
                "or both --cell-type-bw and --cell-type-peak."
            )

def get_required_keys(args):
    '''
    Get required keys for the embedding
    '''
    # required = ["chrombert_region_file", "hdf5_file", "pretrain_ckpt"]

    # if is_cell_specific(args):
    #     required.append("mtx_mask")

    required = ["chrombert_region_file", "hdf5_file"]

    return required


# =========================
# region embedding
# =========================

def embed_region_processed(dl=None,model_emb=None,odir=None,oname=None,emb_npy_file=None, overlap_idx = None):
    
    if emb_npy_file is not None:
        all_embs = np.load(emb_npy_file)
        assert overlap_idx is not None, "overlap_idx is required when emb_npy_file is provided"
        region_embs = all_embs[overlap_idx]
    else:
        region_embs = generate_embeddings(dl, model_emb)
        if odir is not None:
            os.makedirs(odir, exist_ok=True)
            stem = oname if oname is not None else "region_emb"
            np.save(f"{odir}/{stem}.npy", region_embs)
    mean_region_embs = region_embs.mean(axis=0)
    if odir is not None:
        stem = oname if oname is not None else ""
        np.save(f"{odir}/mean_{stem}.npy", mean_region_embs)
    return region_embs,mean_region_embs
    
def run_region_general(args, files_dict, odir, return_data=False,model_emb=None):
    '''
    Generate region embeddings for general model
    '''
    focus_region = args.region

    overlap_bed = check_region_file(focus_region, files_dict, odir)
    overlap_idx = overlap_bed["build_region_index"].to_numpy()

    emb_npy_path = files_dict["region_emb_npy"]

    if os.path.exists(emb_npy_path) and getattr(args, "ignore_regulator", None) is None:
        print("Using cached region embeddings...")
        all_emb = np.load(emb_npy_path)
        region_embs = all_emb[overlap_idx]
    else:
        print("Cached region embeddings not found, computing by model...")
        sup = f"{odir}/model_input.tsv"
        data_config, model_config = build_model_dataset_config(
            args, files_dict, supervised_file_for_ignore_idx=sup
        )
        dl = data_config.init_dataloader(batch_size=args.batch_size, supervised_file=sup)
        if model_emb is None:
            model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
        region_embs = generate_embeddings(dl, model_emb)

    np.save(f"{odir}/region_emb_{args.oname}.npy", region_embs)
    report_region(args, odir, overlap_bed, cell_specific=False)

    if return_data:
        return region_embs, overlap_bed


def run_region_cell(args, files_dict, odir, return_data=False, model_emb=None):
    focus_region = args.region

    overlap_bed = check_region_file(focus_region, files_dict, odir)

    if model_emb is None:
        model_emb, _ = build_cell_model_emb(args, files_dict, odir)

    sup = f"{odir}/model_input.tsv"
    data_config, _ = build_model_dataset_config(
        args, files_dict, supervised_file_for_ignore_idx=sup
    )
    dl = data_config.init_dataloader(batch_size=args.batch_size, supervised_file=sup)
    region_embs = generate_embeddings(dl, model_emb)

    np.save(f"{odir}/region_emb_{args.oname}.npy", region_embs)
    report_region(args, odir, overlap_bed, cell_specific=True)

    if return_data:
        return region_embs, overlap_bed


# =========================
# gene embedding
# =========================

def load_gene_meta(args, files_dict):
    gene_meta_tsv = files_dict["gene_meta_tsv"]
    if not os.path.exists(gene_meta_tsv):
        print(f"Gene meta file not found: {gene_meta_tsv}, downloading...")
        download(basedir=os.path.expanduser(args.chrombert_cache_dir),genome=args.genome,resolution=args.resolution)
    gene_meta = pd.read_csv(gene_meta_tsv, sep="\t")
    gene_meta["gene_id"] = gene_meta["gene_id"].astype(str).str.lower()
    gene_meta["gene_name"] = gene_meta["gene_name"].astype(str).str.lower()
    return gene_meta, gene_meta_tsv


def prepare_gene_regions(args, files_dict, odir):
    gene_meta, gene_meta_tsv = load_gene_meta(args, files_dict)
    focus_genes = parse_focus_genes(args.gene)

    overlap_genes, not_found_genes, gene_to_region_idx = overlap_gene_map_region(
        gene_meta, focus_genes, odir
    )

    needed_idx = np.unique(
        np.concatenate(
            [np.asarray(v).astype(int).reshape(-1) for v in gene_to_region_idx.values()]
        )
    )

    chrombert_region_bed = files_dict["chrombert_region_file"]
    region_df = pd.read_csv(
        chrombert_region_bed,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "build_region_index"],
    )

    sub_df = region_df[region_df["build_region_index"].isin(needed_idx)].copy()
    sub_df = sub_df.sort_values("build_region_index").reset_index(drop=True)

    if sub_df.shape[0] == 0:
        raise ValueError("No required build_region_index found in chrombert_region_file.")

    sub_df.to_csv(f"{odir}/model_input_gene.tsv", index=False, sep="\t")

    return {
        "gene_meta_tsv": gene_meta_tsv,
        "focus_genes": focus_genes,
        "overlap_genes": overlap_genes,
        "not_found_genes": not_found_genes,
        "gene_to_region_idx": gene_to_region_idx,
        "sub_df": sub_df,
    }


def pool_gene_embeddings(region_embs, sub_df, gene_to_region_idx):
    idx2emb = {int(i): region_embs[j] for j, i in enumerate(sub_df["build_region_index"].tolist())}

    gene_emb_dict = {}
    for g, idxs in gene_to_region_idx.items():
        idxs = np.asarray(idxs).astype(int).reshape(-1)
        embs = [idx2emb[i] for i in idxs if i in idx2emb]
        if len(embs) == 0:
            raise ValueError(f"No region embeddings found for gene={g}.")
        gene_emb_dict[g] = np.mean(np.stack(embs, axis=0), axis=0)

    return gene_emb_dict


def run_gene_general(args, files_dict, odir, return_data=False,model_emb=None):
    info = prepare_gene_regions(args, files_dict, odir)

    emb_npy_path = files_dict["region_emb_npy"]

    if os.path.exists(emb_npy_path) and getattr(args, "ignore_regulator", None) is None:
        print("Using cached region embeddings for gene pooling...")
        all_emb = np.load(emb_npy_path)
        gene_emb_dict = {}
        for g, idxs in info["gene_to_region_idx"].items():
            idxs = np.asarray(idxs).astype(int)
            gene_emb_dict[g] = all_emb[idxs].mean(axis=0)
    else:
        print("Cached region embeddings not found, computing by model...")
        sup = f"{odir}/model_input_gene.tsv"
        data_config, model_config = build_model_dataset_config(
            args, files_dict, supervised_file_for_ignore_idx=sup
        )
        dl = data_config.init_dataloader(batch_size=args.batch_size, supervised_file=sup)
        if model_emb is None:
            model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
        region_embs = generate_embeddings(dl, model_emb)
        gene_emb_dict = pool_gene_embeddings(
            region_embs,
            info["sub_df"],
            info["gene_to_region_idx"],
        )

    with open(f"{odir}/gene_emb_{args.oname}.pkl", "wb") as f:
        pickle.dump(gene_emb_dict, f)

    report_gene(args, odir, info, cell_specific=False)
    if return_data:
        return gene_emb_dict


def run_gene_cell(args, files_dict, odir, return_data=False, model_emb=None):
    info = prepare_gene_regions(args, files_dict, odir)

    sup = f"{odir}/model_input_gene.tsv"
    if model_emb is None:
        model_emb, data_config = build_cell_model_emb(args, files_dict, odir)
    else:
        data_config, _ = build_model_dataset_config(
            args, files_dict, supervised_file_for_ignore_idx=sup
        )
    dl = data_config.init_dataloader(batch_size=args.batch_size, supervised_file=sup)
    region_embs = generate_embeddings(dl, model_emb)

    gene_emb_dict = pool_gene_embeddings(
        region_embs,
        info["sub_df"],
        info["gene_to_region_idx"],
    )

    with open(f"{odir}/gene_emb_{args.oname}.pkl", "wb") as f:
        pickle.dump(gene_emb_dict, f)

    report_gene(args, odir, info, cell_specific=True)

    if return_data:
        return gene_emb_dict

# =========================
# report
# =========================

def report_region(args, odir, overlap_bed, cell_specific=False):
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
    print("Note: It is possible for a single region to overlap multiple ChromBERT regions.")
    print("Overlapping regions BED file:", f"{odir}/overlap_region.bed")
    print("Non-overlapping regions BED file:", f"{odir}/no_overlap_region.bed")
    print("Region embeddings saved to:", f"{odir}/region_emb_{args.oname}.npy")
    print("Embedding type:", "cell-specific" if cell_specific else "general")


def report_gene(args, odir, info, cell_specific=False):
    print("\nFinished!")
    print("Note: All gene names were converted to lowercase for matching.")
    print(
        f"Gene count summary - requested: {len(info['focus_genes'])}, "
        f"matched: {len(info['overlap_genes'])}, "
        f"not found: {len(info['not_found_genes'])}"
    )
    print("Matched gene meta saved to:", f"{odir}/overlap_genes_meta.tsv")
    print("Gene embeddings saved to:", f"{odir}/gene_emb_{args.oname}.pkl")
    print("Embedding type:", "cell-specific" if cell_specific else "general")


# =========================
# embed_region parameters
# =========================

@click.command(name="embed_region", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False,
              help="Region BED file.")
@click.option("--gene",
              type=str,
              required=False,
              help="Gene symbols or IDs separated by ';'.")
@click.option("--cell-type-bw",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False,
              help="Cell type accessibility BigWig file.")
@click.option("--cell-type-peak",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False,
              help="Cell type accessibility Peak BED file.")
@click.option("--ft-ckpt",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False,
              default=None,
              show_default=True,
              help="Fine-tuned checkpoint. If provided, skip fine-tuning.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False))
@click.option("--oname", default="embedding", show_default=True, type=str)
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False))
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False))
@click.option("--mode", default="fast", show_default=True,
              type=click.Choice(["fast", "full"], case_sensitive=False),
              help="Used when training cell-specific model.")
@click.option("--batch-size", default=4, show_default=True, type=int)
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
@click.option("--chrombert-cache-dir", default="~/.cache/chrombert/data",
              show_default=True, type=click.Path(file_okay=False))
@click.option("--chrombert-region-file", default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--chrombert-region-emb-file", default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--chrombert-gene-meta", default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True))
def embed_region(
    region,
    gene,
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
    chrombert_cache_dir,
    chrombert_region_file,
    chrombert_region_emb_file,
    chrombert_gene_meta,
):
    '''
    Generate region embeddings for specified regions or gene promoter regions
    '''
    args = SimpleNamespace(
        region=region,
        gene=gene,
        cell_type_bw=cell_type_bw,
        cell_type_peak=cell_type_peak,
        ft_ckpt=ft_ckpt,
        # ignore_regulator=ignore_regulator,
        # gep=gep,
        # flank_window=flank_window,
        odir=odir,
        oname=oname,
        genome=genome,
        resolution=resolution,
        mode=mode,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
        chrombert_gene_meta=chrombert_gene_meta,
    )

    validate_args(args)

    os.makedirs(odir, exist_ok=True)
    files_dict = resolve_paths(args)
    check_files(files_dict, required_keys=get_required_keys(args))

    cell_mode = is_cell_specific(args)

    if cell_mode:
        model_emb, _ = build_cell_model_emb(args, files_dict, odir)
        if args.region is not None:
            run_region_cell(args, files_dict, odir, model_emb=model_emb)
        if args.gene is not None:
            run_gene_cell(args, files_dict, odir, model_emb=model_emb)
    else:
        if args.region is not None:
            run_region_general(args, files_dict, odir)
        if args.gene is not None:
            run_gene_general(args, files_dict, odir)


if __name__ == "__main__":
    embed_region()