import os
import re
import click
from types import SimpleNamespace
import subprocess as sp

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

from chrombert import ChromBERTFTConfig, DatasetConfig

from .utils import resolve_paths, check_files
from .utils import overlap_region, overlap_regulator_func, chrom_to_int_series


def plot_regulator_subnetwork(G: nx.Graph, target_reg: str, odir: str, k_hop: int, threshold: float, quantile: float):
    """Plot k-hop ego subnetwork for a given regulator."""
    if target_reg not in G:
        print(f"[WARN] {target_reg} not found in graph (degree == 0)")
        return

    subG = nx.ego_graph(G, target_reg, radius=k_hop)

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(subG, seed=42)

    node_colors = ["red" if n == target_reg else "lightgray" for n in subG.nodes()]
    node_sizes = [600 if n == target_reg else 400 for n in subG.nodes()]

    edges = list(subG.edges(data=True))
    weights = [d.get("weight", 1.0) for (_, _, d) in edges]
    if len(weights) == 0:
        weights = [1.0]
    w_min, w_max = min(weights), max(weights)
    edge_widths = [1 + 3 * (w - w_min) / (w_max - w_min + 1e-8) for w in weights]

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(subG, pos, font_size=9)

    plt.axis("off")
    plt.title(f"{target_reg} subnetwork (k={k_hop}, thr={threshold:.3f}, q={quantile:.3f})")
    plt.tight_layout()
    plt.savefig(f"{odir}/subnetwork_{target_reg}_k{k_hop}.pdf")
    plt.close()


def build_trn_from_embeddings(embs: np.ndarray, regulators: list[str], odir: str, quantile: float):
    """
    Build a co-association graph based on cosine similarity thresholded by quantile.
    Save:
      - regulator_cosine_similarity.tsv
      - total_graph_edge_*.tsv
    Returns: (G, threshold, cos_sim_df)
    """
    cos_sim = cosine_similarity(embs)
    cos_sim_df = pd.DataFrame(cos_sim, index=regulators, columns=regulators)
    cos_sim_df.to_csv(f"{odir}/regulator_cosine_similarity.tsv", sep="\t", index=True)

    N = embs.shape[0]
    i_upper = np.triu_indices(N, k=1)
    threshold = np.quantile(cos_sim[i_upper], quantile)

    G = nx.Graph()
    edge_rows = []
    for i in range(N):
        for j in range(i + 1, N):
            w = float(cos_sim[i, j])
            if w >= threshold:
                n1, n2 = regulators[i], regulators[j]
                G.add_edge(n1, n2, weight=w)
                edge_rows.append((n1, n2, w))

    df_edges = pd.DataFrame(edge_rows, columns=["node1", "node2", "cosine_similarity"])
    df_edges.to_csv(
        f"{odir}/total_graph_edge_threshold{threshold:.3f}_quantile{quantile:.3f}.tsv",
        sep="\t",
        index=False,
    )

    print("Total graph nodes:", G.number_of_nodes())
    print(f"Total graph edges (threshold={threshold:.3f}):", G.number_of_edges())
    return G, threshold, cos_sim_df


def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    if args.genome.lower() == "mm10" and args.resolution != "1kb":
        raise ValueError("mm10 currently only supports 1kb in this cache layout (adjust if you have more).")

    files_dict = resolve_paths(args)
    required_keys=[
        "chrombert_region_file",
        "chrombert_regulator_file",
        "hdf5_file",
        "pretrain_ckpt",
        "mtx_mask"
    ]
    check_files(files_dict, required_keys=required_keys)

    # Intersect user regions with ChromBERT regions
    overlap_bed = overlap_region(args.region_bed, files_dict["chrombert_region_file"], odir)

    # Convert chrom names to integer codes
    overlap_bed["chrom"] = chrom_to_int_series(overlap_bed["chrom"], args.genome)
    overlap_bed = overlap_bed.dropna(subset=["chrom"]).copy()
    overlap_bed["chrom"] = overlap_bed["chrom"].astype(int)
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)

    # Optional: filter regulators for subnetwork plotting
    focus_regs = None
    if args.regulator is not None:
        focus_regs, _, _= overlap_regulator_func(args.regulator, files_dict["chrombert_regulator_file"])
        if len(focus_regs) == 0:
            print("[WARN] None of the requested regulators were found in ChromBERT regulator list. Will still build full TRN.")
            focus_regs = None

    # Data loader
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=f"{odir}/model_input.tsv",
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dl = data_config.init_dataloader()

    # Model embedding manager
    model_config = ChromBERTFTConfig(
        genome=args.genome,
        dropout=0,
        task="general",
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        mtx_mask=files_dict["mtx_mask"],
    )
    model = model_config.init_model().get_embedding_manager().cuda().bfloat16()

    regulators = model.list_regulator
    # We will compute mean embedding of each regulator across all overlapping regions
    embs_sum = np.zeros((len(regulators), 768), dtype=np.float64)
    total_counts = 0

    with torch.no_grad():
        for batch in tqdm(dl, total=len(dl)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            emb = model(batch)
            emb_np = emb.float().cpu().numpy()

            bs = emb_np.shape[0]
            total_counts += bs
            embs_sum += emb_np.sum(axis=0)

    embs_mean = embs_sum / max(total_counts, 1)

    # Build TRN graph by cosine similarity quantile threshold
    G, threshold, _ = build_trn_from_embeddings(embs_mean, regulators, odir, quantile=args.quantile)

    # Optional: plot subnetworks for user-specified regulators
    if focus_regs is not None:
        for reg in focus_regs:
            plot_regulator_subnetwork(
                G, reg, odir, k_hop=args.k_hop, threshold=threshold, quantile=args.quantile
            )

    print("Finished!")
    print("Saved outputs to:", odir)


@click.command(name="infer_trn", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region-bed", "region_bed",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region BED file (focus regions).")
@click.option("--regulator", default=None,
              help="Optional. Regulators to plot subnetworks, e.g. EZH2;BRD4;CTCF. Use ';' to separate.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")

@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False), help="Resolution.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default=os.path.expanduser("~/.cache/chrombert/data"),
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache dir (contains config/ and checkpoint/ etc).")
@click.option("--batch-size", default=64, show_default=True, type=int,
              help="Batch size for region dataloader.")
@click.option("--num-workers", default=8, show_default=True, type=int,
              help="Number of dataloader workers.")
@click.option("--quantile", default=0.99, show_default=True, type=float,
              help="Quantile threshold for cosine similarity edges.")
@click.option("--k-hop", default=1, show_default=True, type=int,
              help="k-hop radius for subnetwork plotting.")

def infer_trn(region_bed, regulator, odir, genome, resolution,chrombert_cache_dir,
              batch_size, num_workers, quantile, k_hop):

    args = SimpleNamespace(
        region_bed=region_bed,
        regulator=regulator,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        quantile=quantile,
        k_hop=k_hop, 
    )
    run(args)


if __name__ == "__main__":
    infer_trn()

