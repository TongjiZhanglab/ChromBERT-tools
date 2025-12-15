'''
This script is used to infer trn for 1073 regulators for specific cell types.

Two typical usage patterns:

1) Use a prepared ChromBERT cache directory (recommended):


python run.py \
    --cell_type_peak /path/to/cell_type.bed \
    --cell_type_bw /path/to/cell_type.bw \
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
from chrombert.scripts.chrombert_make_dataset import process
import bbi  # pip install pybbi
import lightning.pytorch as pl 
import glob
from chrombert.scripts.utils import HDF5Manager
from tqdm import tqdm
import pickle
from chrombert import ChromBERTFTConfig, DatasetConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Infer TRN for regulators on specific regions.")
    parser.add_argument("--cell_type_bw",
        type=str,
        required=True,
        help="Cell type BigWig file.",
    )
    parser.add_argument(
        "--cell_type_peak",
        type=str,
        required=True,
        help="Cell type Peak file.",
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
        default=4,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        help="whether use fast mode, if fast mode, downsample region numbers to 20k for training",
        default="fast",
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
    
    # 6) ChromBERT base chromatin accessibility signal file:
    base_ca_signal = os.path.join(args.chrombert_cache_dir, "anno", "hm_1kb_accessibility_signal_mean.npy")
    
    return {
        "chrombert_regulator_file": chrombert_regulator_file,
        "hdf5_file": hdf5_file,
        "pretrain_ckpt": pretrain_ckpt,
        "mtx_mask": mtx_mask,
        "chrombert_region_file": chrombert_region_file,
        "base_ca_signal": base_ca_signal,
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


def bw_getSignal_bins(
    bw, regions:pd.DataFrame
    ):
    regions = regions.copy()
    with bbi.open(str(bw)) as bwf:
        mtx = bwf.stackup(regions["chrom"],regions["start"],regions["end"], bins=1, missing=0)
        # mean= bwf.info["summary"]["mean"]
        # mtx = mtx/mean
    df_signal = pd.DataFrame(data = mtx, columns = ['signal'])
    return df_signal

def split_data(df,name,odir):
    columns=['chrom','start','end','build_region_index','label']
    train = df.sample(frac=0.8,random_state=55)
    test = df.drop(train.index).sample(frac=0.5,random_state=55)
    valid = df.drop(train.index).drop(test.index)
    train[columns].to_csv(f"{odir}/train{name}.csv",index=False)
    test[columns].to_csv(f"{odir}/test{name}.csv",index=False)
    valid[columns].to_csv(f"{odir}/valid{name}.csv",index=False)

def make_dataset(peak,bw,d_odir, files_dict):
    
    # 1.prepare_dataset
    total_peak_process = process(peak,files_dict['chrombert_region_file'],mode='region')[['chrom','start','end','build_region_index']].drop_duplicates().reset_index(drop=True)
    total_peak_process.to_csv(f'{d_odir}/chrombert_region_overlap_peak.csv',index=False)
    
    total_region_processed_sampled = total_peak_process.sample(n=20000,random_state=55).reset_index(drop=True)
    total_region_processed_sampled.to_csv(f'{d_odir}/total_region_processed_sampled.csv',index=False)
    
    # 2. scan signal
    signal = bw_getSignal_bins(bw=bw,regions=total_peak_process)
    total_region_signal_processed = pd.concat([total_peak_process,signal],axis=1)
    base_ca_signal_array = np.load(files_dict['base_ca_signal']) # baseline chrom accessibility
    total_region_signal_processed['baseline'] = base_ca_signal_array[total_region_signal_processed['build_region_index'].values]
    
    # 3. log2 signal and fold change
    total_region_signal_processed['log2_signal'] = np.log2(total_region_signal_processed['signal'] + 1)
    total_region_signal_processed['log2_baseline'] = np.log2(total_region_signal_processed['baseline'] + 1)
    total_region_signal_processed['label'] = total_region_signal_processed['log2_signal'] - total_region_signal_processed['log2_baseline']
    total_region_signal_processed = total_region_signal_processed[['chrom','start','end','build_region_index','label','log2_signal','log2_baseline','signal','baseline']]
    
    
    total_region_signal_processed.to_csv(f'{d_odir}/total_region_signal_processed.csv',index=False)
    total_region_signal_processed_sampled = total_region_signal_processed[total_region_signal_processed.build_region_index.isin(total_region_processed_sampled.build_region_index)]
    total_region_signal_processed_sampled
    total_region_signal_processed_sampled.to_csv(f'{d_odir}/total_region_signal_processed_sampled.csv',index=False)
    
    
    # 4. train/test/valid split
    split_data(total_region_signal_processed,'',d_odir)
    split_data(total_region_signal_processed_sampled,'_sampled',d_odir)
    
    # 5.up region and nochange region
    up_region = total_region_signal_processed[total_region_signal_processed["label"] > 1].sort_values("label", ascending=False).head(1000).reset_index(drop=True)


    total_region_signal_processed['abs_label'] = np.abs(total_region_signal_processed['label'])
    nochange_region = total_region_signal_processed.query("signal > 0 or baseline > 0").query("label <1 and label > -1").sort_values('abs_label').reset_index(drop=True).iloc[0:1000]
    nochange_region
    
    up_region.to_csv(f'{d_odir}/up_region.csv',index=False)
    nochange_region.to_csv(f'{d_odir}/nochange_region.csv',index=False)
    

def model_train(d_odir,train_odir,args,files_dict):
    # 1. init train datamodule
    # init dataconfig
    data_config = DatasetConfig(
        kind = "GeneralDataset",
        supervised_file = None,
        hdf5_file = files_dict["hdf5_file"],
        batch_size = args.batch_size,
        num_workers = 8,
    )
    # init datamodule
    if args.mode == 'fast':
        data_module = chrombert.LitChromBERTFTDataModule(
            config = data_config, 
            train_params = {'supervised_file': f'{d_odir}/train_sampled.csv'}, 
            val_params = {'supervised_file':f'{d_odir}/valid_sampled.csv'}, 
            test_params = {'supervised_file':f'{d_odir}/test_sampled.csv'}
        )
    else:
        data_module = chrombert.LitChromBERTFTDataModule(
            config = data_config, 
            train_params = {'supervised_file': f'{d_odir}/train.csv'}, 
            val_params = {'supervised_file':f'{d_odir}/valid.csv'}, 
            test_params = {'supervised_file':f'{d_odir}/test.csv'}
        )
    data_module.setup()
    
    # 2. init chrombert 
    model_config = chrombert.get_preset_model_config(
        basedir = args.chrombert_cache_dir,
        genome = "hg38",
        preset = "general",
        pretrain_ckpt = files_dict["pretrain_ckpt"],
        mtx_mask = files_dict["mtx_mask"],
    )
    model = model_config.init_model()
    model.freeze_pretrain(2) ### freeze chrombert 6 transformer blocks during fine-tuning 

    # 3. init trainer
    train_config = chrombert.finetune.TrainConfig(kind='regression',        
                                              loss='rmse',
                                              max_epochs=5,
                                              accumulate_grad_batches=8,
                                              val_check_interval=0.2,
                                              limit_val_batches=0.5,
                                              tag='cell_specific')
    train_module = train_config.init_pl_module(model)
    callback_ckpt = pl.callbacks.ModelCheckpoint(monitor = f"{train_config.tag}_validation/pcc", mode = "max")
    early_stop = pl.callbacks.EarlyStopping(
        monitor=f"{train_config.tag}_validation/pcc",
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
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            callback_ckpt,
            early_stop,
        ],
        logger=pl.loggers.TensorBoardLogger(f"./{train_odir}/lightning_logs", name='cell_specific'),
        )
    trainer.fit(train_module,data_module)
    
    
def generate_emb(model_emb, files_dict, sup_file, odir, name, args):

    
    data_config = DatasetConfig(
        kind = "GeneralDataset",
        supervised_file = sup_file,
        hdf5_file = files_dict["hdf5_file"],
        batch_size = args.batch_size,
        num_workers = 8,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()
    regulators = model_emb.list_regulator
    regulator_idx_dict = {regulator:idx for idx,regulator in enumerate(regulators)}

    
    total_counts = 0
    embs_pool = np.zeros((len(regulators), 768), dtype=np.float64)

    with torch.no_grad():
        for batch in tqdm(dl, total = len(dl)):
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            total_counts += batch["region"].shape[0]
            emb = model_emb(batch) # initialize the cache
            emb_np = emb.float().cpu().numpy()            
            embs_pool += emb_np.sum(axis=0)

        embs_pool /= total_counts

    embs_pool_dict = {}
    for reg_name, reg_idx in regulator_idx_dict.items():
        embs_pool_dict[reg_name] = embs_pool[reg_idx]
    out_pkl = os.path.join(odir, f"{name}_region_pool_regulator_embs_dict.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(embs_pool_dict, f)
    return embs_pool


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
    
def model_embedding(train_odir, args, files_dict): 
    ckpts = glob.glob(f"{train_odir}/lightning_logs/**/checkpoints/*.ckpt", recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No ckpt found under {train_odir}/lightning_logs, Please check you train finished")

    ft_ckpt = os.path.abspath(max(ckpts, key=os.path.getmtime))
    
    
    # load cell-specific chrombert
    model_tuned = chrombert.get_preset_model_config(
        basedir = args.chrombert_cache_dir,
        genome = "hg38",
        preset = "general",
        mtx_mask = files_dict["mtx_mask"],
        dropout = 0,
        finetune_ckpt = f'{ft_ckpt}').init_model()
    model_emb = model_tuned.get_embedding_manager().cuda()

    return model_emb
    
    
def main():
    args = parse_args()
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    files_dict = resolve_paths(args)
    check_files(files_dict, args)
    
    d_odir = f"{odir}/dataset";  os.makedirs(d_odir, exist_ok=True)
    train_odir = f"{odir}/train"; os.makedirs(train_odir, exist_ok=True)
    results_odir = f"{odir}/results"; os.makedirs(results_odir, exist_ok=True)
    emb_odir = f"{odir}/emb"; os.makedirs(emb_odir, exist_ok=True)

    
    # 1. preapre dataset for cell specific training
    make_dataset(args.cell_type_peak,args.cell_type_bw, d_odir, files_dict)
    print("finished stage1: prepare dataset")
    
    # 2. train chrombert
    model_train(d_odir,train_odir,args,files_dict)
    print('finished stage 2: the important stage,  Congratudate you get a cell-specific chrombert')
    
    # 3. infer cell specific regulator and trn
    model_emb = model_embedding(train_odir, args, files_dict)
    # 3.1 get up/nochange region embedding
    up_emb = generate_emb(model_emb, files_dict, f"{d_odir}/up_region.csv", emb_odir, "up", args)
    nochange_emb = generate_emb(model_emb, files_dict, f"{d_odir}/nochange_region.csv", emb_odir, "nochange", args)
    
    ## 3.2 find key regulator
    chrom_acc_similarity = [cosine_similarity(up_emb[i].reshape(1, -1), nochange_emb[i].reshape(1, -1))[0, 0] for i in range(up_emb.shape[0])]
    chrom_acc_similarity_df = pd.DataFrame({'factors':model_emb.list_regulator,'similarity':chrom_acc_similarity}).sort_values(by='similarity').reset_index(drop=True)
    chrom_acc_similarity_df = chrom_acc_similarity_df[chrom_acc_similarity_df["factors"] != "input"].reset_index(drop=True)
    chrom_acc_similarity_df['rank']=chrom_acc_similarity_df.index + 1
    chrom_acc_similarity_df.to_csv(f'{results_odir}/factor_importance_rank.csv',index=False)
    print("finished stage 3: infer cell specific regulator (top25):")
    print(chrom_acc_similarity_df.head(n=25))
    
    ## 3.3 plot trn
    focus_regulator = chrom_acc_similarity_df.head(n=25).factors.tolist()
    plot_trn(up_emb, model_emb.list_regulator, focus_regulator, results_odir, quantile=args.quantile)
    print("finished stage 4: plot trn")
    
    print(f"Finished all stages!")
    print(f"You get a cell-specific chrombert in {train_odir}")
    print(f"You can get the most important regulators for this cell type in {results_odir}/factor_importance_rank.csv")
    print(f"You can get the trn for this cell type in {results_odir}/total_graph_edge_threshold{args.quantile:.2f}_quantile{args.quantile:.2f}.tsv")
    print(f"You can get the subnetwork for each top25 regulator in {results_odir}/subnetwork_*.pdf")
    
    
if __name__ == "__main__":
    main()
