import os
import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

from .utils import resolve_paths, check_files
from .utils import check_region_file, overlap_regulator_func
from .utils_interpret import (
    build_interpret_config,
    load_interpret_model,
)
from .embed_regulator import embed_regulator_processed_mean


def plot_regulator_subnetwork(G: nx.Graph, target_reg: str, odir: str, k_hop: int, threshold: float, quantile: float, return_fig=False):
    """Plot k-hop ego subnetwork for a given regulator."""
    if target_reg not in G:
        print(f"[WARN] {target_reg} not found in graph (degree == 0)")
        return None

    subG = nx.ego_graph(G, target_reg, radius=k_hop)

    fig = plt.figure(figsize=(6, 6))
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
    plt.savefig(f"{odir}/subnetwork_{target_reg}_k{k_hop}_q{quantile:.3f}_thr{threshold:.3f}.pdf")
    print(f"Regulator subnetwork saved to: {odir}/subnetwork_{target_reg}_k{k_hop}_q{quantile:.3f}_thr{threshold:.3f}.pdf")
    
    if not return_fig:
        plt.close()
    return None


def build_regulator_subnetwork(embs: np.ndarray, regulators: list[str], odir: str, quantile: float):
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
    return G, threshold, cos_sim_df, df_edges


def run(args, return_data=False):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    if args.genome.lower() == "mm10" and args.resolution != "1kb":
        raise ValueError("mm10 currently only supports 1kb in this cache layout (adjust if you have more).")

    files_dict = resolve_paths(args)
    required_keys=[
        "chrombert_region_file",
        "chrombert_regulator_file",
        "hdf5_file",
        # "pretrain_ckpt",
        # "mtx_mask"
    ]
    check_files(files_dict, required_keys=required_keys)

    # Intersect user regions with ChromBERT regions
    overlap_bed = check_region_file(args.region, files_dict, odir)
    sup_file = f"{odir}/model_input.tsv"

    # Optional: filter regulators for subnetwork plotting
    focus_regs = None
    if args.regulator is not None:
        focus_regs, _, _ = overlap_regulator_func(
            args.regulator, files_dict["chrombert_regulator_file"]
        )
        if len(focus_regs) == 0:
            print(
                "[WARN] None of the requested regulators were found in ChromBERT regulator list. "
                "Will still build full TRN."
            )
            focus_regs = None

    data_config, model_config = build_interpret_config(
        args, files_dict, sup_file
    )
    _, model_emb = load_interpret_model(model_config)
    dl = data_config.init_dataloader(supervised_file=sup_file)
    emb_odir = os.path.join(odir, "emb")
    os.makedirs(emb_odir, exist_ok=True)
    embs_pool, regulators = embed_regulator_processed_mean(
        dl, model_emb, emb_odir, "region"
    )

    # Build TRN graph by cosine similarity quantile threshold
    G, threshold, cos_sim_df, df_edges = build_regulator_subnetwork(embs_pool, regulators, odir, quantile=args.quantile)

    # Optional: plot subnetworks for user-specified regulators
    if focus_regs is not None:
        for reg in focus_regs:
            plot_regulator_subnetwork(
                G, reg, odir, k_hop=args.k_hop, threshold=threshold, quantile=args.quantile, 
                return_fig=return_data
            )

    print("Finished!")
    print("Saved outputs to:", odir)
    print(f"Regulator cosine similarity saved to: {odir}/regulator_cosine_similarity.tsv")
    print(f"Total graph edges saved to: {odir}/total_graph_edge_threshold{threshold:.3f}_quantile{args.quantile:.3f}.tsv")
    if return_data:
        return cos_sim_df,df_edges


@click.command(name="interpret_regulator_regulator_interactions", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region BED file (focus regions).")
@click.option("--regulator", default=None,
              help="Optional. Regulators to plot subnetworks, e.g. EZH2;BRD4;CTCF. Use ';' to separate.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--ft-ckpt", "ft_ckpt",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, default=None, show_default=True,
              help="Fine-tuned ChromBERT checkpoint. If provided, using this ckpt to generate embeddings.")
@click.option("--ignore-regulator", "ignore_regulator",
              type=str,
              required=False, default=None, show_default=True,
              help="Ignore regulator(s) Use ';' to separate multiple names. If provided, will ignore these regulators.")
@click.option("--gep", "gep", is_flag=True, default=False, show_default=True,
              help="Use GEP model (multi-flank-window). Default: False.")
@click.option("--flank-window", "flank_window",
              type=int,
              required=False, default=4, show_default=True,
              help="Flank window size for GEP model.")

@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False), help="Resolution.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default="~/.cache/chrombert/data",
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache dir (contains config/ and checkpoint/ etc).")
@click.option("--batch-size", default=64, show_default=True, type=int,
              help="Batch size for region dataloader.")
@click.option("--quantile", default=0.98, show_default=True, type=float,
              help="Quantile threshold for cosine similarity edges.")
@click.option("--k-hop", default=1, show_default=True, type=int,
              help="k-hop radius for subnetwork plotting.")

def interpret_regulator_regulator_interactions(
    region,
    regulator,
    odir,
    genome,
    resolution,
    chrombert_cache_dir,
    batch_size,
    quantile,
    k_hop,
    ft_ckpt,
    ignore_regulator,
    gep,
    flank_window,
):
    '''
    Interpret regulator-regulator interactions
    '''
    args = SimpleNamespace(
        region=region,
        regulator=regulator,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        batch_size=batch_size,
        quantile=quantile,
        k_hop=k_hop,
        ft_ckpt=ft_ckpt,
        ignore_regulator=ignore_regulator,
        gep=gep,
        flank_window=flank_window,
    )
    run(args)


if __name__ == "__main__":
    interpret_regulator_regulator_interactions()

