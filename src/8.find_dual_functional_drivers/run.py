'''
This script finds driver factors in different functional regions.
If you have identified a dual-functional regulator, you can use this script 
to find its dual networks by specifying the --dual_regulator parameter.

Usage example:

Positive regions represent function1, and negative regions represent function2.

python 11.find_driver_factor_in_different_functional_regions.py \
    --function1_bed "${datadir}/hESC_GSM1003524_EZH2.bed;${datadir}/hESC_GSM1003525_H3K27me3.bed" \
    --function1_mode "and" \
    --function2_bed "${datadir}/hESC_GSM1003524_EZH2.bed" \
    --function2_mode "and" \
    --dual_regulator ezh2 \
    --odir output \
    --ignore_regulator "h3k27me3"
'''

import argparse
import os
import pickle
import glob
import json
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchmetrics as tm
import networkx as nx
import nxviz as nv
from nxviz import annotate
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import chrombert
from chrombert.scripts.utils import HDF5Manager
from chrombert.scripts.chrombert_make_dataset import get_overlap
from chrombert import ChromBERTFTConfig, DatasetConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Find driver factors in different functional regions and infer dual networks for dual-functional regulators.")
    parser.add_argument(
        "--function1_bed",
        type=str,
        required=True,
        help="Different genomic regions for function1. Use ';' to separate multiple BED files.",
    )
    parser.add_argument(
        "--function1_mode",
        type=str,
        required=False,
        default = "and",
        help="Logic mode for function1 regions: 'and' requires all conditions to be satisfied, 'or' requires at least one condition.",
    )
    parser.add_argument(
        "--function2_bed",
        type=str,
        required=True,
        help="Different genomic regions for function2. Use ';' to separate multiple BED files.",
    )
    parser.add_argument(
        "--function2_mode",
        type=str,
        required=False,
        default = "and",
        help="Logic mode for function2 regions: 'and' requires all conditions to be satisfied, 'or' requires at least one condition.",
    )
    parser.add_argument(
        "--dual_regulator",
        type=str,
        required=False,
        default=None,
        help="Dual-functional regulator(s) for which to find dual networks. Use ';' to separate multiple regulators.",
    )
    parser.add_argument(
        "--ignore_regulator",
        type=str,
        required=False,
        default=None,
        help="Regulators to ignore as input features when distinguishing functional regions. Use ';' to separate multiple regulators. Example: if function1 has 'ezh2 and h3k27me3' while function2 has only 'ezh2', you may want to ignore 'h3k27me3'.",
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
        help="Batch size. Increase this value if you have sufficient GPU memory.",
        default=4,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        help="Training mode: 'fast' downsamples to 20k regions for quicker training; 'normal' uses all regions.",
        default="fast",
    )

    parser.add_argument(
        "--chrombert_cache_dir",
        type=str,
        required=False,
        default=os.path.expanduser('~/.cache/chrombert/data'),
        help="ChromBERT cache directory (containing config/ and anno/ subfolders).",
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
    
    # 6) ChromBERT meta file:
    meta_file = os.path.join(args.chrombert_cache_dir, "config", "hg38_6k_meta.json")
    
    return {
        "chrombert_regulator_file": chrombert_regulator_file,
        "hdf5_file": hdf5_file,
        "pretrain_ckpt": pretrain_ckpt,
        "mtx_mask": mtx_mask,
        "chrombert_region_file": chrombert_region_file,
        "meta_file": meta_file,
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
    print(f"not found factors: {not_overlap_regulator}")
    print(f"ChromBERT regulator list loaded from: {chrombert_regulator_file}")
    return overlap_regulator, not_overlap_regulator, regulator_dict
        
def split_data(df, name, odir):
    """Split data into train/validation/test sets (80%/10%/10%)."""
    columns = ['chrom', 'start', 'end', 'build_region_index', 'label']
    train = df.sample(frac=0.8, random_state=55)
    test = df.drop(train.index).sample(frac=0.5, random_state=55)
    valid = df.drop(train.index).drop(test.index)
    
    train[columns].to_csv(f"{odir}/train{name}.csv", index=False)
    test[columns].to_csv(f"{odir}/test{name}.csv", index=False)
    valid[columns].to_csv(f"{odir}/valid{name}.csv", index=False)


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
    
    Args:
        args: Command-line arguments
        files_dict: Dictionary containing ChromBERT file paths
        d_odir: Output directory for dataset files
    
    Returns:
        Updated args with mode information
    """
    # Parse input BED files
    func1_bed_files_list = [x for x in args.function1_bed.split(";") if x.strip()]
    func2_bed_files_list = [x for x in args.function2_bed.split(";") if x.strip()]
    
    # Load ChromBERT reference regions (index 0 in dfs)
    ref_regions = files_dict["chrombert_region_file"]    
    # Build list of dataframes: [bed_file1, bed_file2, ...]
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
    func1_regions = merge_regions_by_mode(
        func1_dfs, 
        args.function1_mode, 
        "function1"
    )   
    # Merge regions for function2
    func2_regions = merge_regions_by_mode(
        func2_dfs, 
        args.function2_mode, 
        "function2"
    )
    
    # Remove function1 regions from function2 to avoid overlap (function2 becomes negative class)
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
    if len(combined_dataset) > 20000 and args.mode == "fast":
        print(f"  Fast mode: downsampling to 20k regions (10k per class)")
        downsampled_dataset = (
            combined_dataset
            .groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=min(10000, len(g)), random_state=55))
            .reset_index(drop=True)
        )
        downsampled_dataset.to_csv(os.path.join(d_odir, "total_sampled.csv"), index=False)
        split_data(downsampled_dataset, "_sampled", d_odir)
    else:
        print(f"  Using all regions for training")
        split_data(combined_dataset, "", d_odir)
        args.mode = "normal"
    
    return args

def model_train(args,files_dict,d_odir,train_odir, ignore_object):
    # 1. init train datamodule
    ignore = True if ignore_object is not None else False
    dc = chrombert.DatasetConfig(
            kind = "GeneralDataset",
            supervised_file = None,
            hdf5_file = files_dict["hdf5_file"],
            batch_size = args.batch_size,
            num_workers = 8,
            ignore = ignore, 
            ignore_object = ignore_object,
            meta_file = files_dict["meta_file"],
    )
    if args.mode == "fast":
        ds = dc.init_dataset(supervised_file = os.path.join(d_odir, "train_sampled.csv"))
        data_module = chrombert.LitChromBERTFTDataModule(
            config = dc, 
            train_params = {'supervised_file': f'{d_odir}/train_sampled.csv'}, 
            val_params = {'supervised_file':f'{d_odir}/valid_sampled.csv'}, 
            test_params = {'supervised_file':f'{d_odir}/test_sampled.csv'}
        )
    else:
        ds = dc.init_dataset(supervised_file = os.path.join(d_odir, "train.csv"))
        data_module = chrombert.LitChromBERTFTDataModule(
            config = dc, 
            train_params = {'supervised_file': f'{d_odir}/train.csv'}, 
            val_params = {'supervised_file':f'{d_odir}/valid.csv'}, 
            test_params = {'supervised_file':f'{d_odir}/test.csv'}
        )
    ignore_index = ds[0]["ignore_index"] if ignore else None
    data_module.setup()
    
    # 2. init chrombert 
    model_config = chrombert.get_preset_model_config(
        basedir = args.chrombert_cache_dir,
        genome = "hg38",
        preset = "general",
        pretrain_ckpt = files_dict["pretrain_ckpt"],
        mtx_mask = files_dict["mtx_mask"],
        ignore = ignore,
        ignore_index = ignore_index,
    )
    
    model = model_config.init_model()
    model.freeze_pretrain(2) ### freeze chrombert 6 transformer blocks during fine-tuning
    
    
    # 3. init trainer
    train_config = chrombert.finetune.TrainConfig(kind='classification',        
                                              loss='bce',
                                              max_epochs=5,
                                              accumulate_grad_batches=8,
                                              val_check_interval=0.2,
                                              limit_val_batches=0.5,
                                              tag='dual_functional_regions')
    train_module = train_config.init_pl_module(model) # wrap model with PyTorch Lightning module
    callback_ckpt = pl.callbacks.ModelCheckpoint(monitor = f"{train_config.tag}_validation/auprc", mode = "max")
    early_stop = pl.callbacks.EarlyStopping(
        monitor=f"{train_config.tag}_validation/auprc",
        mode="max",
        patience=5, # early stop if no improvement for 5 validation steps    
        min_delta=0.01,
        verbose=True,
    )
    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        log_every_n_steps=1, 
        limit_val_batches = train_config.limit_val_batches,
        val_check_interval = train_config.val_check_interval,
        accelerator="gpu", 
        accumulate_grad_batches= train_config.accumulate_grad_batches, 
        fast_dev_run=False, 
        precision="bf16-mixed",
        strategy="auto",
        enable_progress_bar=False, # 添加：禁用 tqdm 进度条
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            callback_ckpt,
            early_stop,
        ],
        logger=pl.loggers.TensorBoardLogger(f"./{train_odir}/lightning_logs", name='dual_functional_regions'),
        )
    trainer.fit(train_module,data_module)
    return data_module, model_config

def cal_metrics(preds,labels):
    metrics_auroc = tm.AUROC(task="binary", ignore_index=-1)
    metrics_auprc = tm.AveragePrecision(task="binary", ignore_index=-1)
    metrics_mcc = tm.MatthewsCorrCoef(task="binary", ignore_index=-1)
    metrics_f1 = tm.F1Score(task="binary", ignore_index=-1)
    metrics_precision = tm.Precision(task="binary", ignore_index=-1)
    metrics_recall = tm.Recall(task="binary", ignore_index=-1)


    score_auroc = metrics_auroc(preds, labels).item()
    score_auprc = metrics_auprc(preds, labels).item()
    score_mcc = metrics_mcc(preds, labels).item()
    score_f1 = metrics_f1(preds, labels).item()
    score_precision = metrics_precision(preds, labels).item()
    score_recall = metrics_recall(preds, labels).item()

    metrics_auroc.reset()
    metrics_auprc.reset()
    metrics_mcc.reset()
    metrics_f1.reset()
    metrics_precision.reset()
    metrics_recall.reset()

    metrics = {
        "auroc": score_auroc,
        "auprc": score_auprc,
        "mcc": score_mcc,
        "f1": score_f1,
        "precision": score_precision,
        "recall": score_recall,
    }
    return metrics

def eval_model(args,files_dict,d_odir,train_odir, data_module, model_config):
    ckpts = glob.glob(f"{train_odir}/**/checkpoints/*.ckpt", recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {train_odir}. Please verify that training completed successfully.")
    ft_ckpt = os.path.abspath(max(ckpts, key=os.path.getmtime))

    # init test dataset
    dc_test = data_module.test_config
    dl_test = dc_test.init_dataloader(batch_size = args.batch_size)
    
    # init fine-tuened chrombert model
    model_tuned = model_config.init_model(finetune_ckpt = ft_ckpt,dropout = 0).eval().cuda()
    
    # forward
    test_preds = []
    test_labels = []
    for batch in dl_test:
        with torch.no_grad():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            logits = model_tuned(batch)
            test_labels.append(batch["label"].cpu())
            preds = torch.sigmoid(logits)
            test_preds.append(preds)

    test_preds = torch.cat(test_preds, dim = 0).cpu()
    test_labels = torch.cat(test_labels, axis = 0)
    test_preds = test_preds.reshape(-1)
    test_labels = test_labels.reshape(-1)
    test_metrics = cal_metrics(test_preds, test_labels)
    print(f"ft_ckpt: {ft_ckpt}, test_metrics: {test_metrics}")
    with open(os.path.join(train_odir, "eval_performance.json"), "w") as f:
        json.dump(test_metrics, f)
    return model_tuned

        
def generate_emb(emb_odir,data_module,model_tuned):
    model_emb = model_tuned.get_embedding_manager()
    regulators = model_emb.list_regulator
    regulator_idx_dict = {regulator:idx for idx,regulator in enumerate(regulators)}
    
    dc_test = data_module.test_config
    dl_test = dc_test.init_dataloader(batch_size = 1)
    
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
    embs_pool_func1_dict = {regulator:embs_pool_func1[regulator_idx_dict[regulator]] for regulator in regulators}
    embs_pool_func2_dict = {regulator:embs_pool_func2[regulator_idx_dict[regulator]] for regulator in regulators}
    out_pkl_func1 = os.path.join(emb_odir, f"func1_regulator_embs_dict.pkl")
    out_pkl_func2 = os.path.join(emb_odir, f"func2_regulator_embs_dict.pkl")
    with open(out_pkl_func1, "wb") as f:
        pickle.dump(embs_pool_func1_dict, f)
    with open(out_pkl_func2, "wb") as f:
        pickle.dump(embs_pool_func2_dict, f)
    
    return embs_pool_func1, embs_pool_func2, model_emb, regulators

def get_node_order(G, group_by, sort_by):
    # Group nodes by the specified attribute
    groups = defaultdict(list)
    for node in G.nodes(data=True):
        group_key = node[1].get(group_by)
        sort_value = node[1].get(sort_by)
        groups[group_key].append((node[0], sort_value))

    # Sort nodes within each group based on the specified attribute
    sorted_nodes = []
    for group in sorted(groups.keys()):
        sorted_nodes.extend(sorted(groups[group], key=lambda x: x[1]))

    # Extract node labels in order
    node_labels_in_order = [node[0] for node in sorted_nodes]
    return node_labels_in_order

def plot_dual_trn(results_odir, top_pairs, dual_regulator, df_results, df_cos_func1, df_cos_func2, thre_func1, thre_func2,differential_threshold=0.1):
    cccol = ["#CE0013","#16557A"]
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
    ax = nv.circos(
        G,
        group_by="group",
        sort_by="value",
        node_color_by="group",
        edge_alpha_by="edge_value"
    )
    node_order = get_node_order(G, 'group', 'value')
    annotate.circos_labels(G, group_by="group", sort_by="value", layout="rotate")
    plt.tight_layout()
    plt.savefig(f'{results_odir}/dual_regulator_subnetwork.pdf')
    print(f"Dual regulator subnetwork plot saved: {results_odir}/dual_regulator_subnetwork.pdf")
    print("Yellow color represents function1 subnetwork; blue color represents function2 subnetwork.")

def infer_driver_factor_trn(dual_regulators, emb_odir,results_odir,data_module, model_tuned, differential_threshold = 0.1):
    embs_pool_func1, embs_pool_func2, model_emb, regulators = generate_emb(emb_odir, data_module, model_tuned)
    
    dual_regulator_sim = [cosine_similarity(embs_pool_func1[i].reshape(1, -1), embs_pool_func2[i].reshape(1, -1))[0, 0] for i in range(embs_pool_func1.shape[0])]
    dual_regulator_sim_df = pd.DataFrame({'factors':regulators,'similarity':dual_regulator_sim}).sort_values(by='similarity').reset_index(drop=True)
    dual_regulator_sim_df = dual_regulator_sim_df[dual_regulator_sim_df["factors"] != "input"].reset_index(drop=True)
    dual_regulator_sim_df['rank']=dual_regulator_sim_df.index + 1
    dual_regulator_sim_df.to_csv(os.path.join(results_odir, "factor_importance_rank.csv"), index=False)
    print("Finished stage 4.1: infer driver factors in different regions (top 25):")
    print(dual_regulator_sim_df.head(n=25))
    
    
    cos_func1 = cosine_similarity(embs_pool_func1)
    cos_func2 = cosine_similarity(embs_pool_func2)
    df_cos_func1 = pd.DataFrame(cos_func1, columns = regulators, index = regulators)
    df_cos_func2 = pd.DataFrame(cos_func2, columns = regulators, index = regulators)
    
    df_cos_func1.to_csv(os.path.join(results_odir, "regulator_cosine_similarity_on_function1_regions.csv"), index=False)
    df_cos_func2.to_csv(os.path.join(results_odir, "regulator_cosine_similarity_on_function2_regions.csv"), index=False)
    
    if dual_regulators is not None:
        thre_func1 = np.percentile(cos_func1.flatten(), 95)
        thre_func2 = np.percentile(cos_func2.flatten(), 95)
        for idx, dual_regulator in enumerate(dual_regulators):
            df_cos_reg = pd.DataFrame(index =regulators, data = {"function1":df_cos_func1.loc[dual_regulator,:],"function2":df_cos_func2.loc[dual_regulator,:]})
            df_cos_reg["diff"] = df_cos_reg["function1"] - df_cos_reg["function2"]
            df_candidate = df_cos_reg[df_cos_reg["diff"].abs() > differential_threshold]
            topN_pos = df_candidate.query("function1 > @thre_func1").index.values
            topN_neg = df_candidate.query("function2 > @thre_func2").index.values
            top_pairs = np.union1d(topN_pos, topN_neg)
            
            plot_dual_trn(results_odir, top_pairs, dual_regulator, df_cos_reg, df_cos_func1, df_cos_func2, thre_func1, thre_func2,differential_threshold=differential_threshold)

        print("Finished stage 4.2: infer dual-functional regulator subnetworks")
    
def main():
    args = parse_args()
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    files_dict = resolve_paths(args)
    check_files(files_dict, args)
    
    dual_regulator = args.dual_regulator
    overlap_dual_regulator = None  # Initialize to avoid undefined variable error    
    if dual_regulator is not None:
        overlap_dual_regulator, not_overlap_dual_regulator, dual_regulator_idx_dict = overlap_regulator_func(dual_regulator, files_dict["chrombert_regulator_file"])
    
    ignore_regulator = args.ignore_regulator
    ignore_object = None
    if ignore_regulator is not None:
        overlap_ignore, not_overlap_ignore, _ = overlap_regulator_func(ignore_regulator, files_dict["chrombert_regulator_file"])
        ignore_object =";".join(overlap_ignore) if len(overlap_ignore) > 0 else None
    
    # 1. prepare dataset
    print("Processing stage 1: prepare dataset")
    d_odir = f"{odir}/dataset";  os.makedirs(d_odir, exist_ok=True)
    args = make_dataset(args, files_dict, d_odir)
    print("Finished stage 1: prepare dataset")
    
    # 2. train chrombert
    train_odir = f"{odir}/train"; os.makedirs(train_odir, exist_ok=True)
    print("Processing stage 2: train ChromBERT to identify different regions")
    data_module,model_config = model_train(args, files_dict, d_odir, train_odir, ignore_object)
    print("Finished stage 2: train ChromBERT to identify different regions")
    
    # 3. eval finetuned chrombert performance
    print("Processing stage 3: evaluate fine-tuned ChromBERT performance")
    model_tuned = eval_model(args, files_dict, d_odir, train_odir, data_module, model_config)
    print("Finished stage 3: evaluate fine-tuned ChromBERT performance")

    # 4. infer driver factor in different regions
    print("Processing stage 4: infer driver factors in different regions")
    emb_odir = f"{odir}/emb"; os.makedirs(emb_odir, exist_ok=True)
    results_odir = f"{odir}/results"; os.makedirs(results_odir, exist_ok=True)
    infer_driver_factor_trn(overlap_dual_regulator, emb_odir, results_odir, data_module, model_tuned)
    print("Finished stage 4: infer driver factors in different regions")

if __name__ == "__main__":
    main()