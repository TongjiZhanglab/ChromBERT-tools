'''
This script is used to infer trn for 1073 regulators on specific regions (regulators co-association).

Two typical usage patterns:

1) Use a prepared ChromBERT cache directory (recommended):


python run.py \
    --regulator EZH2;BRD4;CTCF \
    --region_bed /path/to/region.bed \
    --odir output \
    --chrombert_cache_dir /path/to/chrombert/data
'''

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import subprocess as sp
import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
from chrombert.scripts.utils import HDF5Manager
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer TRN for regulators on specific regions."
    )
    parser.add_argument(
        "--region_bed",
        type=str,
        required=True,
        help="Region BED file.",
    )
    parser.add_argument(
        "--regulator",
        type=str,
        required=False,
        default=None,
        help="Regulators of interest, e.g. EZH2 or BRD4 or CTCF or EZH2;BRD4;CTCF. Use ';' to separate multiple regulators.",
    )
    parser.add_argument(
        "--odir",
        type=str,
        required=False,
        help="Output directory.",
        default="./output",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Batch size. if you have enough GPU memory, you can set it to a larger value.",
        default=64,
    )

    parser.add_argument(
        "--chrombert_cache_dir",
        type=str,
        required=False,
        default=os.path.expanduser('~/.cache/chrombert/data'),
        help="ChromBERT cache directory (containing config/ and anno/ subfolders).",
    )
    
    parser.add_argument(
        "--quantile",
        type=float,
        required=False,
        default=0.99,
        help="quantile threshold for cosine similarity.",
    )
    parser.add_argument(
        "--k_hop",
        type=int,
        required=False,
        default=1,
        help="k-hop for subnetwork.",
    )
    return parser.parse_args()


def resolve_paths(args):
    """Resolve regulator list file and embedding paths based on cache dir / overrides."""
    assert args.chrombert_cache_dir is not None, "ChromBERT cache directory must be provided"

    # 1) ChromBERT regulator list file
    chrombert_regulator_file = os.path.join(args.chrombert_cache_dir, "config", "hg38_6k_regulators_list.txt")
    
    # 2) ChromBERT input hdf5 file
    hdf5_file = os.path.join(args.chrombert_cache_dir, "hg38_6k_1kb.hdf5")
    
    # 3) ChromBERT pretrain ckpt file
    pretrain_ckpt = os.path.join(args.chrombert_cache_dir,"checkpoint", "hg38_6k_1kb_pretrain.ckpt")
    
    # 4) ChromBERT matcix mask file:
    mtx_mask = os.path.join(args.chrombert_cache_dir,"config", "hg38_6k_mask_matrix.tsv")
    
    # 5) ChromBERT region file:
    chrombert_region_file = os.path.join(args.chrombert_cache_dir, "config", "hg38_6k_1kb_region.bed")
    
    return {
        "chrombert_regulator_file": chrombert_regulator_file,
        "hdf5_file": hdf5_file,
        "pretrain_ckpt": pretrain_ckpt,
        "mtx_mask": mtx_mask,
        "chrombert_region_file": chrombert_region_file,
    }


def check_files(files_dict, args):
    """Check that required ChromBERT files exist, and give helpful hints if not."""

    for key, value in files_dict.items():
        if not os.path.exists(value):
            msg = (f"ChromBERT {key} file not found: {value}.\n "
                    "You can download all required files by running the command "
                "`chrombert_prepare_env`")
            print(msg)
            raise FileNotFoundError(msg)

def overlap_region(region_bed, chrombert_region_file, odir):
    # ---------- overlapping focus regions ----------
    cmd_overlap = f"""
    cut -f 1-3 {region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools intersect -F 0.5 -wa -wb -a {chrombert_region_file} -b - \
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
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)
    overlap_idx = overlap_bed["build_region_index"].to_numpy()

    # ---------- non-overlapping focus regions ----------
    cmd_no = f"""
    cut -f 1-3 {region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools intersect -f 0.5 -v -a - -b {chrombert_region_file} \
    > {odir}/no_overlap_focus.bed
    """
    sp.run(cmd_no, shell=True, check=True, executable="/bin/bash")
    
    total_focus = sum(1 for _ in open(region_bed))
    no_overlap_region_len = sum(1 for _ in open(f"{odir}/no_overlap_focus.bed"))
    print(
        f"Focus region summary - total: {total_focus}, "
        f"overlapping with ChromBERT: {overlap_bed.shape[0]}, It is possible for a single focus region to overlap multiple ChromBERT regions,"
        f"non-overlapping: {no_overlap_region_len}"
    )
    print("Overlapping focus regions BED file:", f"{odir}/overlap_focus.bed")
    print("Overlapping focus regions will be used for ChromBERT embedding.")
    print("Non-overlapping focus regions BED file:", f"{odir}/no_overlap_focus.bed")
    return overlap_bed, overlap_idx

def overlap_regulator_func(regulator, chrombert_regulator_file):
    # load regulator list from ChromBERT
    chrombert_regulator = pd.read_csv(
        chrombert_regulator_file,
        sep="\t",
        header=None,
        names=["regulator"],
    )["regulator"].tolist()
    chrombert_regulator = [i.lower() for i in chrombert_regulator]

    # parse user-provided regulators
    focus_regulator_list = [r.strip().lower() for r in regulator.split(";") if r.strip()]

    overlap_regulator = list(set(chrombert_regulator) & set(focus_regulator_list))
    not_overlap_regulator = list(set(focus_regulator_list) - set(chrombert_regulator))
    
    regulator_dict = {tmp_regulator:chrombert_regulator.index(tmp_regulator) for tmp_regulator in overlap_regulator}
    
    print("Note: All regulator names were converted to lowercase for matching.")
    print(
        f"Regulator count summary - requested: {len(focus_regulator_list)}, "
        f"matched in ChromBERT: {len(overlap_regulator)}, "
        f"not found: {len(not_overlap_regulator)}"
    )
    print(f"ChromBERT regulator list loaded from: {chrombert_regulator_file}")
    return overlap_regulator, not_overlap_regulator, regulator_dict

def plot_regulator_subnetwork(G, target_reg, odir,k_hop=1, threshold=None, quantile = None):
    """
    """
    if target_reg not in G:
        print(f"[WARN] {target_reg} not found in graph (degree == 0)")
        return

    # 1. pick k-hop ego graph
    subG = nx.ego_graph(G, target_reg, radius=k_hop)  # target_reg + its k-hop neighbors

    print(f"Subnetwork for {target_reg}:")
    print("  nodes:", subG.number_of_nodes())
    print("  edges:", subG.number_of_edges())

    # 2. plot
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(subG, seed=42)

    # node color/size: distinguish center node and neighbors (optional)
    node_colors = []
    node_sizes = []
    for n in subG.nodes():
        if n == target_reg:
            node_colors.append("red")
            node_sizes.append(500)
        else:
            node_colors.append("lightgray")
            node_sizes.append(500)

    edges = subG.edges(data=True)
    weights = [d.get("weight", 1.0) for (_, _, d) in edges]

    edge_widths = [1 + 3 * (w - min(weights)) / (max(weights) - min(weights) + 1e-8) for w in weights]

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(subG, pos, font_size=10)

    plt.axis("off")
    plt.title(f"Subnetwork of {target_reg} (k:{k_hop}, threshold:{threshold:.3f}, quantile:{quantile:.3f})")
    plt.tight_layout()
    plt.savefig(f"{odir}/subnetwork_{target_reg}_k{k_hop}.pdf")

def plot_trn(embs, regulators, focus_regulator, odir, quantile=0.99):
    cos_sim = cosine_similarity(embs)
    cos_sim_df = pd.DataFrame(cos_sim, index=regulators, columns=regulators)
    cos_sim_df.to_csv(f"{odir}/regulator_cosine_similarity.tsv", sep="\t", index=True)
    N = embs.shape[0]
    i_upper = np.triu_indices(N, k=1)
    threshold = np.quantile(cos_sim[i_upper], quantile)

    
    # only keep edges with cosine similarity >= threshold
    G = nx.Graph()
    edge_rows = []
    for i in range(N):
        for j in range(i + 1, N):
            w = cos_sim[i, j]
            if w >= threshold:
                n1 = regulators[i]
                n2 = regulators[j]
                G.add_edge(n1, n2, weight=w)
                edge_rows.append((n1, n2, w))
    df_edges = pd.DataFrame(edge_rows,
                            columns=["node1", "node2", "cosine_similarity"])
    df_edges.to_csv(f"{odir}/total_graph_edge_threshold{threshold:.2f}_quantile{quantile:.2f}.tsv", sep="\t", index=False)
    
    print("Number of nodes of total graph:", G.number_of_nodes())
    print(f"Number of edges of total graph (threshold={threshold:.3f}):", G.number_of_edges())
    if focus_regulator is not None:
        for reg in focus_regulator:
            plot_regulator_subnetwork(G, reg, odir, k_hop=1, threshold=threshold, quantile = quantile)
    
def main():
    args = parse_args()
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    check_files(files_dict, args)

    # overlap chrombert region and user-provided region
    overlap_region(args.region_bed, files_dict["chrombert_region_file"], odir)
    
    # overlap chrombert regulator and user-provided regulator
    focus_regulator = args.regulator
    if focus_regulator is not None:
        overlap_regulator, _, _ = overlap_regulator_func(focus_regulator, files_dict["chrombert_regulator_file"])
    else:
        overlap_regulator = None
    
    # init datamodule
    data_config = DatasetConfig(
        kind = "GeneralDataset",
        supervised_file = f"{odir}/model_input.tsv",
        hdf5_file = files_dict["hdf5_file"],
        batch_size = args.batch_size,
        num_workers = 8,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()
    
    # init chrombert
    model_config = chrombert.get_preset_model_config(
        basedir = args.chrombert_cache_dir,
        genome = "hg38",
        dropout = 0,
        preset = "general",
        pretrain_ckpt = files_dict["pretrain_ckpt"],
        mtx_mask = files_dict["mtx_mask"],
    )
    model = model_config.init_model().get_embedding_manager().cuda().bfloat16()
    regulators = model.list_regulator
    embs = np.zeros((len(regulators), 768), dtype=np.float64)
    total_counts = 0
    with torch.no_grad():
        for batch in tqdm(dl, total=len(dl)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            emb = model(batch) # initialize the cache 
            bs = batch["region"].shape[0]
            total_counts += bs
            emb_np = emb.float().cpu().numpy()            
            embs += emb_np.sum(axis=0)
    embs /= total_counts
    
    plot_trn(embs, regulators, overlap_regulator, odir, quantile=args.quantile)



    print("Finished!")  
    print("Saved regulator cosine similarity and co-association subnetwork graph to:", odir)

if __name__ == "__main__":
    main()
