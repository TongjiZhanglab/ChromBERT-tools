'''
This script is used to extract embeddings for user-specified focus regions.

Two typical usage patterns:

1) Use a prepared ChromBERT cache directory (recommended):

python run.py \
    --region_bed /path/to/CTCF_ENCFF664UGR.bed \
    --odir output \
    --chrombert_cache_dir /path/to/chrombert/data

2) Specify region BED and embedding file explicitly:

python run.py \
    --region_bed /path/to/CTCF_ENCFF664UGR.bed \
    --odir output \
    --chrombert_region_dir /path/to/config \
    --chrombert_region_emb_file /path/to/anno/hm_1kb_all_region_emb.npy
'''

import argparse
import os
import subprocess as sp
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ChromBERT region embeddings for user-specified focus regions."
    )

    parser.add_argument(
        "--region_bed",
        type=str,
        required=True,
        help="BED file containing focus regions (to be intersected with ChromBERT regions).",
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

    # Mode 2: override default region BED / embedding file
    parser.add_argument(
        "--chrombert_region_file",
        type=str,
        required=False,
        default=None,
        help=(
            "Directory OR full path for the ChromBERT region BED file. "
            "If a directory is provided, the script assumes the file "
            "hg38_6k_1kb_region.bed is inside this directory."
        ),
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

    # 1) ChromBERT region BED
    if args.chrombert_region_file is not None:
        # Allow both a directory or a direct file path
        chrombert_region_bed = args.chrombert_region_file
    else:
        chrombert_region_bed = os.path.join(
            args.chrombert_cache_dir, "config/hg38_6k_1kb_region.bed"
        )

    # 2) ChromBERT region embedding .npy
    if args.chrombert_region_emb_file is not None:
        emb_npy_path = args.chrombert_region_emb_file
    else:
        emb_npy_path = os.path.join(
            args.chrombert_cache_dir, "anno/hm_1kb_all_region_emb.npy"
        )

    return chrombert_region_bed, emb_npy_path


def check_files(chrombert_region_bed, emb_npy_path, args):
    """Check that required ChromBERT files exist, and give helpful hints if not."""

    if not os.path.exists(chrombert_region_bed):
        if args.chrombert_region_file is not None:
            msg = (
                f"ChromBERT region BED file not found: {chrombert_region_bed}.\n"
                "Please check the path you passed to --chrombert_region_file "
                "or provide a correct BED file path."
            )
        else:
            msg = (
                f"ChromBERT region BED file not found: {chrombert_region_bed}.\n"
                "You can download all required files by running the command "
                "`chrombert_prepare_env`, or download this BED file directly from:\n"
                "  https://huggingface.co/TongjiZhanglab/chrombert/"
                "tree/main/data/hg38_6k_1kb_region.bed"
            )
        print(msg)
        raise FileNotFoundError(msg)

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


def main():
    args = parse_args()

    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    focus_region_bed = args.region_bed
    chrombert_region_bed, emb_npy_path = resolve_paths(args)
    check_files(chrombert_region_bed, emb_npy_path, args)

    # ---------- overlapping focus regions ----------
    cmd_overlap = f"""
    cut -f 1-3 {focus_region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools merge \
    | bedtools intersect -F 0.5 -wa -wb -a {chrombert_region_bed} -b - \
    | awk 'BEGIN{{OFS="\\t"}}{{print $5,$6,$7,$4}}' \
    > {odir}/overlap_focus.bed
    """
    sp.run(cmd_overlap, shell=True, check=True, executable="/bin/bash")

    overlap_bed = pd.read_csv(
        f"{odir}/overlap_focus.bed",
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "build_region_index"],
    )
    overlap_idx = overlap_bed["build_region_index"].to_numpy()

    # ---------- non-overlapping focus regions ----------
    cmd_no = f"""
    cut -f 1-3 {focus_region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools merge \
    | bedtools intersect -f 0.5 -v -a - -b {chrombert_region_bed} \
    > {odir}/no_overlap_focus.bed
    """
    sp.run(cmd_no, shell=True, check=True, executable="/bin/bash")

    # ---------- focus region embeddings ----------
    all_emb = np.load(emb_npy_path)
    overlap_emb = all_emb[overlap_idx]
    np.save(f"{odir}/overlap_focus_emb.npy", overlap_emb)

    # ---------- report ----------
    total_focus = sum(1 for _ in open(focus_region_bed))
    no_overlap_region_len = sum(1 for _ in open(f"{odir}/no_overlap_focus.bed"))

    print("Finished!")
    print(
        f"Focus region summary - total: {total_focus}, "
        f"overlapping with ChromBERT: {overlap_bed.shape[0]}, It is possible for a single focus region to overlap multiple ChromBERT regions,"
        f"non-overlapping: {no_overlap_region_len}"
    )
    print("Overlapping focus regions BED file:", f"{odir}/overlap_focus.bed")
    print("Non-overlapping focus regions BED file:", f"{odir}/no_overlap_focus.bed")
    print("Overlapping focus region embeddings saved to:", f"{odir}/overlap_focus_emb.npy")


if __name__ == "__main__":
    main()
