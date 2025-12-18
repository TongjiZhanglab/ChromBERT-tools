import os
import click
import pickle
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
from .utils import resolve_paths, overlap_gene_map_region


def check_files(file_dicts, args):
    """Check required files. If region embedding is missing, we will fallback to model inference."""
    chrombert_region_file = file_dicts["chrombert_region_file"]
    if not os.path.exists(chrombert_region_file):
        if args.chrombert_region_file is not None:
            msg = (
                f"ChromBERT region BED file not found: {chrombert_region_file}.\n"
                "Please check the path you passed to --chrombert-region-file."
            )
        else:
            msg = (
                f"ChromBERT region BED file not found: {chrombert_region_file}.\n"
                "You can download all required files by running `chrombert_prepare_env`."
            )
        print(msg)
        raise FileNotFoundError(msg)

    gene_meta_tsv = file_dicts["gene_meta_tsv"]
    if not os.path.exists(gene_meta_tsv):
        if args.chrombert_gene_meta is not None:
            msg = (
                f"ChromBERT gene meta file not found: {gene_meta_tsv}.\n"
                "Please check the path you passed to --chrombert-gene-meta."
            )
        else:
            msg = (
                f"ChromBERT gene meta file not found: {gene_meta_tsv}.\n"
                "You can download all required files by running `chrombert_prepare_env`."
            )
        print(msg)
        raise FileNotFoundError(msg)

    # Region embedding file is optional. If missing, we will compute by model (same style as embed_region).
    emb_npy_path = file_dicts["region_emb_npy"]
    if not os.path.exists(emb_npy_path):
        print(f"ChromBERT region embedding file not found: {emb_npy_path}.")
        print("Fallback: load ChromBERT model to compute region embeddings for requested genes.")
        # sanity check for fallback requirements
        if (not os.path.exists(file_dicts["hdf5_file"])) or (not os.path.exists(file_dicts["pretrain_ckpt"])):
            print("Fallback requires hdf5 + ckpt. Please run `chrombert_prepare_env` or provide a correct cache dir.")


def model_emb_func(args, files_dict, odir):
    """Initialize dataloader and ChromBERT embedding manager for fallback region embedding computation."""
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=f"{odir}/model_input.csv",
        hdf5_file=files_dict["hdf5_file"],
        batch_size=4,
        num_workers=8,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()

    model_config = ChromBERTFTConfig(
        genome=args.genome,
        dropout=0,
        task="general",
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        mtx_mask=files_dict["mtx_mask"],
    )
    model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
    return ds, dl, model_emb


def _parse_focus_genes(gene_str: str):
    """Parse user-provided genes; separated by ';'."""
    focus = [g.strip().lower() for g in gene_str.split(";") if g.strip()]
    return focus


def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    check_files(files_dict, args)

    # ---------- load gene meta ----------
    gene_meta_tsv = files_dict["gene_meta_tsv"]
    gene_meta = pd.read_csv(gene_meta_tsv, sep="\t")
    gene_meta["gene_id"] = gene_meta["gene_id"].astype(str).str.lower()
    gene_meta["gene_name"] = gene_meta["gene_name"].astype(str).str.lower()

    # ---------- parse requested genes ----------
    focus_genes = _parse_focus_genes(args.gene)

    overlap_genes, not_found_genes, gene_to_region_idx = overlap_gene_map_region(gene_meta, focus_genes, odir)

    # ---------- gene embeddings ----------
    emb_npy_path = files_dict["region_emb_npy"]
    gene_emb_dict = {}

    if os.path.exists(emb_npy_path):
        # Fast path: directly index from cached region embedding matrix
        all_emb = np.load(emb_npy_path)  # shape: (N_regions, 768)
        for g, idxs in gene_to_region_idx.items():
            idxs = np.asarray(idxs).astype(int)
            gene_emb_dict[g] = all_emb[idxs].mean(axis=0)
    else:
        # Fallback: compute region embeddings only for required build_region_index
        needed_idx = np.unique(np.concatenate([np.asarray(v).astype(int).reshape(-1) for v in gene_to_region_idx.values()]))

        # Load ChromBERT region bed and subset required indices
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
            raise ValueError("No required build_region_index found in chrombert_region_file. Check gene_meta vs region.bed.")

        # embed_region template writes model_input.csv then runs model_emb_func
        sub_df.to_csv(f"{odir}/model_input.csv", index=False)

        ds, dl, model_emb = model_emb_func(args, files_dict, odir)

        region_embs = []
        with torch.no_grad():
            for batch in tqdm(dl, total=len(dl)):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                model_emb(batch)  # initialize cache
                region_embs.append(model_emb.get_region_embedding().float().cpu().detach())

        region_embs = torch.cat(region_embs, dim=0).numpy()  # (len(sub_df), 768)

        # build index -> embedding mapping
        idx2emb = {int(i): region_embs[j] for j, i in enumerate(sub_df["build_region_index"].tolist())}

        # compute gene embedding by mean pooling its TSS-related region embeddings
        for g, idxs in gene_to_region_idx.items():
            idxs = np.asarray(idxs).astype(int).reshape(-1)
            embs = [idx2emb[i] for i in idxs if i in idx2emb]
            if len(embs) == 0:
                raise ValueError(f"No region embeddings found for gene={g}.")
            gene_emb_dict[g] = np.mean(np.stack(embs, axis=0), axis=0)

    # ---------- save outputs ----------
    import pickle
    with open(f"{odir}/embs_dict.pkl", "wb") as f:
        pickle.dump(gene_emb_dict, f)

    # ---------- report ----------
    print("Finished!")
    print("Note: All gene names were converted to lowercase for matching.")
    print(
        f"Gene count summary - requested: {len(focus_genes)}, "
        f"matched: {len(overlap_genes)}, "
        f"not found: {len(not_found_genes)}"
    )
    print("Gene meta file:", gene_meta_tsv)
    print("Region embedding source:", emb_npy_path if os.path.exists(emb_npy_path) else "computed by ChromBERT model")
    print("Gene embeddings saved to:", f"{odir}/embs_dict.pkl")
    print("Matched gene meta saved to:", f"{odir}/overlap_genes_meta.tsv")


@click.command(name="embed_gene", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--gene", "gene",
              type=str,
              required=True,
              help="Gene symbols or IDs. e.g. ENSG00000170921;TANC2;DPYD. Use ';' to separate multiple genes.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False),
              help="Resolution. Mouse only supports 1kb resolution.")

@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default=os.path.expanduser("~/.cache/chrombert/data"),
              show_default=True,
              type=click.Path(file_okay=False),
              help="ChromBERT cache dir. If you use `chrombert_prepare_env`, you usually don't need to provide this.")

@click.option("--chrombert-region-file", "chrombert_region_file",
              default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="ChromBERT region BED file. If not provided, use the default {genome}_{nd}_{resolution}_region.bed in the cache dir.")
@click.option("--chrombert-region-emb-file", "chrombert_region_emb_file",
              default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="ChromBERT region embedding file. If not provided, use the default {genome}_{resolution}_region_emb.npy in the cache dir (fallback to model if missing).")
@click.option("--chrombert-gene-meta", "chrombert_gene_meta",
              default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="ChromBERT gene meta TSV. If not provided, try {genome}_{resolution}_gene_meta.tsv, then fallback to hm_1kb_gene_meta.tsv in the cache dir.")
def embed_gene(gene, odir, genome, resolution, chrombert_cache_dir,
               chrombert_region_file, chrombert_region_emb_file, chrombert_gene_meta):

    args = SimpleNamespace(
        gene=gene,
        odir=odir,
        genome=genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
        chrombert_gene_meta=chrombert_gene_meta,
    )
    run(args)


if __name__ == "__main__":
    embed_gene()
