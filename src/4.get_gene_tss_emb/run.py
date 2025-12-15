'''
This script is used to extract embeddings for user-specified focus regions.

Two typical usage patterns:

1) Use a prepared ChromBERT cache directory (recommended):

python run.py \
    --gene  \
    --odir output \
    --chrombert_cache_dir /path/to/chrombert/data

2) Specify region BED and embedding file explicitly:

python run.py \
    --gene  \
    --odir output \
    --chrombert_region_emb_file /path/to/anno/hm_1kb_all_region_emb.npy
    --chrmbert_gene_meta /path/to/anno/hm_1kb_gene_meta.tsv
'''

import argparse
import os
import subprocess as sp
import numpy as np
import pandas as pd
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ChromBERT TSS-based gene embeddings for user-specified genes."
    )

    parser.add_argument(
        "--gene",
        type=str,
        required=True,
        help="Gene symbols or IDs. e.g ENSG00000170921;TANC2;ENSG00000200997;DPYD. Use ; to separate multiple genes."
    )
    parser.add_argument(
        "--odir",
        type=str,
        required=False,
        help="Output directory.",
        default="./output",
    )

    # Mode 1: use a ChromBERT cache dir (standard layout)
    parser.add_argument(
        "--chrombert_cache_dir",
        type=str,
        required=True,
        help="ChromBERT cache directory (containing config/ and anno/ subfolders).",
    )

    # Mode 2: override default region embedding file and gene meta file
    parser.add_argument(
        "--chrombert_gene_meta",
        type=str,
        required=False,
        default=None,
        help="Full path to the ChromBERT gene meta file .tsv file.",
    )
    
    parser.add_argument(
        "--chrombert_region_emb_file",
        type=str,
        required=False,
        default=None,
        help="Full path to the ChromBERT region embedding .npy file.",
    )

    return parser.parse_args()


def resolve_paths(args):
    """Resolve region BED and embedding paths based on cache dir / overrides."""


    # 1) ChromBERT region embedding .npy
    if args.chrombert_region_emb_file is not None:
        emb_npy_path = args.chrombert_region_emb_file
    else:
        emb_npy_path = os.path.join(
            args.chrombert_cache_dir, "anno/hm_1kb_all_region_emb.npy"
        )
        
    # 2) ChromBERT gene meta .tsv
    if args.chrombert_gene_meta is not None:
        gene_meta_tsv = args.chrombert_gene_meta
    else:
        gene_meta_tsv = os.path.join(
            args.chrombert_cache_dir, "anno/hm_1kb_gene_meta.tsv"
        )

    return emb_npy_path, gene_meta_tsv


def check_files(emb_npy_path, gene_meta_tsv, args):
    """Check that required ChromBERT files exist, and give helpful hints if not."""

    if not os.path.exists(emb_npy_path):
        if args.chrombert_region_emb_file is not None:
            msg = (
                f"ChromBERT region embedding file not found: {emb_npy_path}.\n"
                "Please check the path you passed to --chrombert_region_emb_file."
            )
        else:
            msg = (
                f"ChromBERT region embedding file not found: {emb_npy_path}.\n"
                "You can download all required files by running the command "
                "`chrombert_prepare_env`, or download this embedding file directly from:\n"
                "  https://huggingface.co/TongjiZhanglab/chrombert/"
                "tree/main/data/hm_1kb_all_region_emb.npy"
            )
        print(msg)
        raise FileNotFoundError(msg)
    
    if not os.path.exists(gene_meta_tsv):
        if args.chrombert_gene_meta is not None:
            msg = (
                f"ChromBERT gene meta file not found: {gene_meta_tsv}.\n"
                "Please check the path you passed to --chrombert_gene_meta."
            )
        else:
            msg = (
                f"ChromBERT gene meta file not found: {gene_meta_tsv}.\n"
                "You can download all required files by running the command "
                "`chrombert_prepare_env`, or download this gene meta file directly from:\n"
                "  https://huggingface.co/TongjiZhanglab/chrombert/"
                "tree/main/data/hm_1kb_gene_meta.tsv"
            )
        print(msg)
        raise FileNotFoundError(msg)


def main():
    args = parse_args()

    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    emb_npy_path, gene_meta_tsv = resolve_paths(args)
    check_files(emb_npy_path, gene_meta_tsv, args)
    
    gene_meta = pd.read_csv(gene_meta_tsv, sep="\t")
    gene_meta["gene_id"] = gene_meta["gene_id"].str.lower()
    gene_meta["gene_name"] = gene_meta["gene_name"].str.lower()
    
    focus_genes = args.gene.split(";")
    focus_genes = [gene.strip().lower() for gene in focus_genes]
    overlap_genes = []
    non_overlap_genes = []
    gene_id_idx = {}
    for gene in focus_genes:
        if gene.startswith("ensg"):
            if gene in gene_meta["gene_id"].tolist():
                overlap_genes.append(gene)
                gene_id_idx[gene] = gene_meta.query("gene_id == @gene").build_region_index.values
            else:
                non_overlap_genes.append(gene)
        else:
            if gene in gene_meta["gene_name"].tolist():
                overlap_genes.append(gene)
                gene_id_idx[gene] = gene_meta.query("gene_name == @gene").build_region_index.values
            else:
                non_overlap_genes.append(gene)
    
    
    
    overlap_genes_meta = gene_meta.query("gene_id in @overlap_genes or gene_name in @overlap_genes")
    overlap_genes_meta = overlap_genes_meta.reset_index(drop=True)
    overlap_genes_meta.to_csv(f"{odir}/overlap_genes_meta.tsv", sep="\t", index=False)
    
    all_emb = np.load(emb_npy_path)
    gene_emb = {}
    for key,value in gene_id_idx.items():
        gene_emb[key] = all_emb[value].mean(axis=0) # if multiple regions for a gene, average the embeddings

    out_pkl = os.path.join(odir, "embs_dict.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(gene_emb, f)

    print("Finished!")
    print("Note: All gene names were converted to lowercase for matching.")
    print(
        f"Gene count summary - requested: {focus_genes}, "
        f"matched in ChromBERT: {len(overlap_genes)}, "
        f"not found: {len(non_overlap_genes)}"
    )
    print("Saved gene embeddings to pickle file:", out_pkl)
    print(f"Requested genes not found in ChromBERT list: {non_overlap_genes}")
    print(f"ChromBERT gene meta list loaded from: {gene_meta_tsv}")
if __name__ == "__main__":
    main()
