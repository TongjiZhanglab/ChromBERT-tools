import os
import pickle
import itertools
from collections import defaultdict

import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import networkx as nx
import nxviz as nv
from nxviz import annotate
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from chrombert.scripts.chrombert_make_dataset import get_overlap
from chrombert import ChromBERTFTConfig, DatasetConfig

from .utils import resolve_paths, check_files, overlap_regulator_func
from .utils import split_data, cal_metrics_binary, factor_rank
from .utils_train_cell import retry_train

def merge_regions_by_mode(dfs, mode, function_name):
    """
    Merge multiple region DataFrames based on specified mode.

    Args:
        dfs: List of DataFrames containing region information
        mode: 'and' requires all conditions; 'or' requires any condition
        function_name: Name for error messages (e.g., 'function1', 'function2')

    Returns:
        Merged DataFrame with label column
    """
    if mode not in {"and", "or"}:
        raise ValueError(f"{function_name}_mode must be 'and' or 'or', got '{mode}'")
    if not dfs:
        raise ValueError(f"{function_name}: no overlapping regions found (empty set).")

    if mode == "or":
        out = pd.concat(dfs, ignore_index=True)
        out = out.drop_duplicates(subset=["build_region_index"]).reset_index(drop=True)
        out["label"] = 1
        return out

    # mode == "and": intersection on build_region_index
    keep = set(dfs[0]["build_region_index"])
    for df in dfs[1:]:
        keep &= set(df["build_region_index"])
    out = dfs[0][dfs[0]["build_region_index"].isin(keep)].copy()
    out = out.drop_duplicates(subset=["build_region_index"]).reset_index(drop=True)
    out["label"] = 1

    return out


def make_dataset(args, files_dict, d_odir):
    """
    Prepare training dataset by defining function1 (positive) and function2 (negative) regions.
    """
    # Parse input BED files
    func1_bed_files_list = [x for x in args.function1_bed.split(";") if x.strip()]
    func2_bed_files_list = [x for x in args.function2_bed.split(";") if x.strip()]

    # Load ChromBERT reference regions
    ref_regions = files_dict["chrombert_region_file"]

    func1_dfs = []
    func2_dfs = []
    for bed_file in func1_bed_files_list:
        df = get_overlap(
            supervised=bed_file,
            regions=ref_regions,
            no_filter=False,
        ).assign(label=lambda df: df["label"] > 0)
        func1_dfs.append(df)

    for bed_file in func2_bed_files_list:
        df = get_overlap(
            supervised=bed_file,
            regions=ref_regions,
            no_filter=False,
        ).assign(label=lambda df: df["label"] > 0)
        func2_dfs.append(df)

    # Merge regions for function1 (positive class)
    func1_regions = merge_regions_by_mode(func1_dfs, args.function1_mode, "function1")

    # Merge regions for function2
    func2_regions = merge_regions_by_mode(func2_dfs, args.function2_mode, "function2")

    # Remove function1 regions from function2 to avoid overlap
    func2_only = func2_regions.loc[
        ~func2_regions["build_region_index"].isin(func1_regions["build_region_index"])
    ].reset_index(drop=True)
    func2_only["label"] = 0  # Negative class

    # Combine positive and negative samples
    combined_dataset = pd.concat([func1_regions, func2_only], ignore_index=True)
    combined_dataset.to_csv(os.path.join(d_odir, "total.csv"), index=False)

    print(f"  Function1 regions (positive): {len(func1_regions)}")
    print(f"  Function2 regions (negative): {len(func2_only)}")
    print(f"  Total dataset size: {len(combined_dataset)}")

    # Downsample if dataset is large and fast mode is enabled
    if args.mode == "fast":
        print("  Fast mode: downsampling to 20k regions (10k per class)")
        downsampled_dataset = (
            combined_dataset.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=min(10000, len(g)), random_state=55))
            .reset_index(drop=True)
        )
        downsampled_dataset.to_csv(os.path.join(d_odir, "total_sampled.csv"), index=False)
        split_data(downsampled_dataset, "_sampled", d_odir)
    else:
        print("  Using all regions for training")
        split_data(combined_dataset, "", d_odir)
        args.mode = "full"

    return args


def generate_emb(emb_odir, data_config, model_tuned):
    model_tuned = model_tuned.eval()
    model_emb = model_tuned.get_embedding_manager()
    regulators = model_emb.list_regulator
    regulator_idx_dict = {regulator: idx for idx, regulator in enumerate(regulators)}

    dl_test = data_config.init_dataloader(batch_size=1)

    total_counts_func1 = 0
    total_counts_func2 = 0
    embs_pool_func1 = np.zeros((len(regulators), 768), dtype=np.float64)
    embs_pool_func2 = np.zeros((len(regulators), 768), dtype=np.float64)

    for batch in tqdm(dl_test):
        with torch.no_grad():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            emb = model_emb(batch)

        if batch["label"].item() == 1:
            total_counts_func1 += batch["region"].shape[0]
            emb_np = emb.float().cpu().numpy()
            embs_pool_func1 += emb_np.sum(axis=0)
        else:
            total_counts_func2 += batch["region"].shape[0]
            emb_np = emb.float().cpu().numpy()
            embs_pool_func2 += emb_np.sum(axis=0)

    embs_pool_func1 /= total_counts_func1
    embs_pool_func2 /= total_counts_func2
    embs_pool_func1_dict = {
        regulator: embs_pool_func1[regulator_idx_dict[regulator]] for regulator in regulators
    }
    embs_pool_func2_dict = {
        regulator: embs_pool_func2[regulator_idx_dict[regulator]] for regulator in regulators
    }
    out_pkl_func1 = os.path.join(emb_odir, "func1_regulator_embs_dict.pkl")
    out_pkl_func2 = os.path.join(emb_odir, "func2_regulator_embs_dict.pkl")
    with open(out_pkl_func1, "wb") as f:
        pickle.dump(embs_pool_func1_dict, f)
    with open(out_pkl_func2, "wb") as f:
        pickle.dump(embs_pool_func2_dict, f)

    return embs_pool_func1, embs_pool_func2, model_emb, regulators


def get_node_order(G, group_by, sort_by):
    groups = defaultdict(list)
    for node in G.nodes(data=True):
        group_key = node[1].get(group_by)
        sort_value = node[1].get(sort_by)
        groups[group_key].append((node[0], sort_value))

    sorted_nodes = []
    for group in sorted(groups.keys()):
        sorted_nodes.extend(sorted(groups[group], key=lambda x: x[1]))

    node_labels_in_order = [node[0] for node in sorted_nodes]
    return node_labels_in_order


def plot_dual_trn(
    results_odir,
    top_pairs,
    dual_regulator,
    df_results,
    df_cos_func1,
    df_cos_func2,
    thre_func1,
    thre_func2,
    differential_threshold=0.1,
):
    cccol = ["#CE0013", "#16557A"]
    G = nx.Graph()
    graph_regulators = np.concatenate([[dual_regulator], top_pairs])
    for i,j in itertools.combinations(np.arange(len(graph_regulators)), 2):
        if i == 0:
            G.add_nodes_from([(i, {'group':0, 'factor':graph_regulators[i], 'value': 1})])
            if df_results.loc[graph_regulators[j], 'diff'] > differential_threshold:
                G.add_nodes_from([(j, {'group':1, 'factor':graph_regulators[j], 'value':df_results.loc[graph_regulators[j], 'function1']})])
                G.add_edges_from([(i, j, {'color': cccol[0], 'edge_value': df_results.loc[graph_regulators[j], 'function1']})])
            elif df_results.loc[graph_regulators[j], 'diff'] < (-1 * differential_threshold):
                G.add_nodes_from([(j, {'group':2, 'factor':graph_regulators[j], 'value':df_results.loc[graph_regulators[j], 'function2']})])
                G.add_edges_from([(i, j, {'color': cccol[1], 'edge_value': df_results.loc[graph_regulators[j], 'function2']})])
        else:
            # print(graph_regulators[i], graph_regulators[j])
            if df_cos_func1.loc[graph_regulators[i], graph_regulators[j]] > thre_func1 or df_cos_func2.loc[graph_regulators[i], graph_regulators[j]] > thre_func2:
                G.add_edges_from([(i, j, {'color': 'lightgrey', 'edge_value': max(df_cos_func1.loc[graph_regulators[i], graph_regulators[j]], df_cos_func2.loc[graph_regulators[i], graph_regulators[j]])})])
    mapping = {i: str(graph_regulators[i]).upper() for i in range(len(graph_regulators))}
    G = nx.relabel_nodes(G, mapping)
    _ax = nv.circos(
        G,
        group_by="group",
        sort_by="value",
        node_color_by="group",
        edge_alpha_by="edge_value",
    )
    _node_order = get_node_order(G, "group", "value")
    annotate.circos_labels(G, group_by="group", sort_by="value", layout="rotate")
    plt.tight_layout()
    plt.savefig(f"{results_odir}/dual_regulator_{dual_regulator}_subnetwork.pdf")
    print(f"Dual regulator subnetwork plot saved: {results_odir}/dual_regulator_{dual_regulator}_subnetwork.pdf")
    print("Yellow color represents function1 subnetwork; blue color represents function2 subnetwork.")


def infer_driver_factor_trn(
    args, dual_regulators, data_config, model_tuned, differential_threshold=0.1
):
    emb_odir = f"{args.odir}/emb";os.makedirs(emb_odir, exist_ok=True)
    results_odir = f"{args.odir}/results";os.makedirs(results_odir, exist_ok=True)
    
    embs_pool_func1, embs_pool_func2, _model_emb, regulators = generate_emb(
        emb_odir, data_config, model_tuned
    )
    dual_regulator_sim_df = factor_rank(embs_pool_func1, embs_pool_func2, regulators, results_odir)
    print("Finished stage 4a: infer driver factors in different regions (top 25):")
    print(dual_regulator_sim_df.head(n=25))

    cos_func1 = cosine_similarity(embs_pool_func1)
    cos_func2 = cosine_similarity(embs_pool_func2)
    df_cos_func1 = pd.DataFrame(cos_func1, columns=regulators, index=regulators)
    df_cos_func2 = pd.DataFrame(cos_func2, columns=regulators, index=regulators)

    df_cos_func1.to_csv(
        os.path.join(results_odir, "regulator_cosine_similarity_on_function1_regions.csv")
    )
    df_cos_func2.to_csv(
        os.path.join(results_odir, "regulator_cosine_similarity_on_function2_regions.csv"))

    if dual_regulators is not None:
        thre_func1 = np.percentile(cos_func1.flatten(), 95)
        thre_func2 = np.percentile(cos_func2.flatten(), 95)
        for _idx, dual_regulator in enumerate(dual_regulators):
            df_cos_reg = pd.DataFrame(
                index=regulators,
                data={
                    "function1": df_cos_func1.loc[dual_regulator, :],
                    "function2": df_cos_func2.loc[dual_regulator, :],
                },
            )
            df_cos_reg["diff"] = df_cos_reg["function1"] - df_cos_reg["function2"]
            df_candidate = df_cos_reg[df_cos_reg["diff"].abs() > differential_threshold]
            topN_pos = df_candidate.query("function1 > @thre_func1").index.values
            topN_neg = df_candidate.query("function2 > @thre_func2").index.values
            top_pairs = np.union1d(topN_pos, topN_neg)

            plot_dual_trn(
                results_odir,
                top_pairs,
                dual_regulator,
                df_cos_reg,
                df_cos_func1,
                df_cos_func2,
                thre_func1,
                thre_func2,
                differential_threshold=differential_threshold,
            )

        print("Finished stage 4b: infer dual-functional regulator subnetworks")


def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    files_dict = resolve_paths(args)
    required_keys=[
        "chrombert_region_file",
        "hdf5_file",
        "pretrain_ckpt",
        "mtx_mask",
        "meta_file"
    ]
    check_files(files_dict, required_keys=required_keys)

    dual_regulator = args.dual_regulator
    overlap_dual_regulator = None  # avoid undefined
    if dual_regulator is not None:
        overlap_dual_regulator, _, _ = overlap_regulator_func(
            dual_regulator, files_dict["chrombert_regulator_file"]
        )

    ignore_regulator = args.ignore_regulator
    ignore_object = None
    if ignore_regulator is not None:
        overlap_ignore, _, _ = overlap_regulator_func(
            ignore_regulator, files_dict["chrombert_regulator_file"]
        )
        ignore_object = ";".join(overlap_ignore) if len(overlap_ignore) > 0 else None
    ignore = True if ignore_object is not None else False
    ignore_index = None
    
    # 1. prepare dataset
    print("Stage 1: Praparing the dataset")
    d_odir = f"{odir}/dataset"
    os.makedirs(d_odir, exist_ok=True)
    args = make_dataset(args, files_dict, d_odir)
    print("Finished stage 1")

    # 2. train chrombert
    if args.ft_ckpt is not None:
        print(f"Use fine-tuned ChromBERT checkpoint file: {args.ft_ckpt} to find driver factors in different regions")
        data_config = DatasetConfig(
            kind="GeneralDataset",
            supervised_file=None,
            hdf5_file=files_dict["hdf5_file"],
            batch_size=args.batch_size,
            num_workers=8,
            meta_file=files_dict["meta_file"],
        )
        if ignore:
            data_config.ignore = ignore
            data_config.ignore_object = ignore_object

            ds = data_config.init_dataset(supervised_file=os.path.join(d_odir, "total.csv"))
            ignore_index = ds[0]["ignore_index"]
        
        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task="general",
            dropout=0,
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            finetune_ckpt=args.ft_ckpt,
            ignore = ignore,
            ignore_index = ignore_index,
        )
        model_tuned = model_config.init_model().cuda()
        print("Finished stage 2")
    else:
        print("Stage 2: Fine-tuning the model")
        model_tuned, train_odir, model_config, data_config = retry_train(args, files_dict, cal_metrics_binary, metcic='auprc', min_threshold=0.2, train_kind = 'classification', ignore_object=ignore_object)
        print("Finished stage 2")

    # 4. infer driver factor in different regions
    data_config.supervised_file = os.path.join(d_odir, "test_sampled.csv") if args.mode == "fast" else os.path.join(d_odir, "test.csv")
    print("Stage 4: infer driver factors in different regions")
    infer_driver_factor_trn(args,overlap_dual_regulator, data_config, model_tuned)
    print("Finished stage 4")
    print("Finished all stages!")
    if args.ft_ckpt is not None:
        print(f"Used fine-tuned ChromBERT checkpoint: {args.ft_ckpt}")
    else:
        print(f"Fine-tuned ChromBERT saved in {train_odir}")
    print(f"Key regulators for classifying function 1 and function 2 regions: {args.odir}/results/factor_importance_rank.csv")
    print(f"Dual-functional regulator subnetwork: {args.odir}/results/dual_regulator_*_subnetwork.pdf (if --dual-regulator was provided)")


@click.command(
    name="find_driver_in_dual_region",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--function1-bed",
    "function1_bed",
    required=True,
    help="Different genomic regions for function1. Use ';' to separate multiple BED files.",
)
@click.option(
    "--function1-mode",
    "function1_mode",
    default="and",
    show_default=True,
    type=click.Choice(["and", "or"], case_sensitive=False),
    help="Logic mode for function1 regions: 'and' requires all conditions; 'or' requires any condition.",
)
@click.option(
    "--function2-bed",
    "function2_bed",
    required=True,
    help="Different genomic regions for function2. Use ';' to separate multiple BED files.",
)
@click.option(
    "--function2-mode",
    "function2_mode",
    default="and",
    show_default=True,
    type=click.Choice(["and", "or"], case_sensitive=False),
    help="Logic mode for function2 regions: 'and' requires all conditions; 'or' requires any condition.",
)
@click.option(
    "--dual-regulator",
    "dual_regulator",
    default=None,
    help="Dual-functional regulator(s). Use ';' to separate multiple regulators.",
)
@click.option(
    "--ignore-regulator",
    "ignore_regulator",
    default=None,
    help="Regulators to ignore. Use ';' to separate multiple regulators.",
)
@click.option(
    "--odir",
    default="./output",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory.",
)
@click.option(
    "--genome",
    default="hg38",
    show_default=True,
    type=click.Choice(["hg38", "mm10"], case_sensitive=False),
    help="Genome version.",
)

@click.option(
    "--resolution", 
    default="1kb", 
    show_default=True,
    type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
    help="ChromBERT resolution."
)

@click.option(
    "--ft-ckpt",
    "ft_ckpt",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Fine-tuned ChromBERT checkpoint file.",
)

@click.option(
    "--batch-size",
    "batch_size",
    default=4,
    show_default=True,
    type=int,
    help="Batch size. Increase this value if you have sufficient GPU memory.",
)
@click.option(
    "--mode",
    default="fast",
    show_default=True,
    type=click.Choice(["fast", "full"], case_sensitive=False),
    help="Training mode: 'fast' downsamples to 20k regions; 'normal' uses all regions.",
)
@click.option(
    "--chrombert-cache-dir",
    "chrombert_cache_dir",
    default=os.path.expanduser("~/.cache/chrombert/data"),
    show_default=True,
    type=click.Path(file_okay=False),
    help="ChromBERT cache directory (containing config/ and anno/ subfolders).",
)
def find_driver_in_dual_region(
    function1_bed,
    function1_mode,
    function2_bed,
    function2_mode,
    dual_regulator,
    ignore_regulator,
    odir,
    genome,
    resolution,
    ft_ckpt,
    batch_size,
    mode,
    chrombert_cache_dir,
):
    """
    Find driver factors in dual functional regions.
    
    """
    args = SimpleNamespace(
        function1_bed=function1_bed,
        function1_mode=str(function1_mode).lower(),
        function2_bed=function2_bed,
        function2_mode=str(function2_mode).lower(),
        dual_regulator=dual_regulator,
        ignore_regulator=ignore_regulator,
        odir=odir,
        genome=genome,
        resolution=resolution,
        ft_ckpt=ft_ckpt,
        batch_size=batch_size,
        mode=str(mode).lower(),
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    find_driver_in_dual_region()
