import os
import pickle
import glob
import json
import subprocess

import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torchmetrics as tm
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import chrombert
from chrombert.scripts.chrombert_make_dataset import process
from chrombert import ChromBERTFTConfig, DatasetConfig
from lightning.pytorch.callbacks import TQDMProgressBar

from .utils import resolve_paths, check_files
from .utils import split_data, bw_getSignal_bins
from .utils import cal_metrics_regression
from .utils import factor_rank
from .utils_train_cell import retry_train
        
def make_exp_dataset(args, files_dict,exp_d_odir):
    gene_meta = pd.read_csv(files_dict['gene_meta_tsv'],sep='\t').query("gene_biotype == 'protein_coding'")[['chrom','start','end','build_region_index','gene_id','tss']]
    # print(f"protein code genes have {len(gene_meta)}")
    
    if args.exp_tpm1 is not None and args.exp_tpm2 is not None:
        if os.path.exists(f'{exp_d_odir}/total.csv'):
            print(f"Expression dataset already exists in {exp_d_odir}")
            return True
        else:
            print(f"Processing stage 1: prepare expression dataset")
            cell1_exp = pd.read_csv(args.exp_tpm1)
            cell2_exp = pd.read_csv(args.exp_tpm2)
            cell1_columns = [i.lower() for i in cell1_exp.columns]
            cell2_columns = [i.lower() for i in cell2_exp.columns]
            cell1_exp.columns = cell1_columns
            cell2_exp.columns = cell2_columns
            if 'gene_id' not in cell1_columns or 'gene_id' not in cell2_columns:
                raise ValueError("expression files needed `gene_id` column, which is ensembl gene_id")
            if 'tpm' not in cell1_columns or 'tpm' not in cell2_columns:
                raise ValueError("expression files needed `tpm` column, which is gene expression values")
            cell1_exp['log1p_tpm'] = np.log1p(cell1_exp['tpm'])
            cell2_exp['log1p_tpm'] = np.log1p(cell2_exp['tpm'])
            merge_exp = pd.merge(cell1_exp[['gene_id','log1p_tpm']],cell2_exp[['gene_id','log1p_tpm']],on=['gene_id'],how='inner',suffixes=['_1','_2']).reset_index(drop=True)
            merge_exp['label'] = merge_exp['log1p_tpm_2'] - merge_exp['log1p_tpm_1']
            if args.direction != '2-1':
                merge_exp['label'] = -merge_exp['label']
                
            merge_exp_anno = pd.merge(gene_meta,merge_exp[['gene_id','label','log1p_tpm_1','log1p_tpm_2']],on='gene_id',how = 'inner')[['chrom','start','end','build_region_index','label','gene_id','tss','log1p_tpm_1','log1p_tpm_2']]
            merge_exp_anno.to_csv(f'{exp_d_odir}/total.csv',index=False)
            
            train_data = merge_exp_anno.sample(frac=0.8,random_state=55)
            test_data = merge_exp_anno.drop(train_data.index).sample(frac=0.5,random_state=55)
            valid_data = merge_exp_anno.drop(train_data.index).drop(test_data.index)
            train_data.to_csv(f'{exp_d_odir}/train.csv',index=False)
            test_data.to_csv(f'{exp_d_odir}/test.csv',index=False)
            valid_data.to_csv(f'{exp_d_odir}/valid.csv',index=False)
            
            up_data = merge_exp_anno[merge_exp_anno['label']>1].sort_values("label", ascending=False).head(1000).reset_index(drop=True)
            merge_exp_anno['abs_label'] = np.abs(merge_exp_anno['label'])
            nochange_data = merge_exp_anno[(merge_exp_anno['label']>-0.5) & (merge_exp_anno['label']<0.5)].sort_values('abs_label').reset_index(drop=True).iloc[0:1000]
            
            up_data.to_csv(f'{exp_d_odir}/up.csv',index=False)
            nochange_data.to_csv(f'{exp_d_odir}/nochange.csv',index=False)
                   
            return True
        
    else:
        print("No expression files provided for cells in cell state transition")
        return False
    
        
def make_acc_dataset(args,files_dict,acc_d_odir):
    if args.acc_peak1 is not None and args.acc_peak2 is not None and args.acc_signal1 is not None and args.acc_signal2 is not None:
        if os.path.exists(f'{acc_d_odir}/total.csv'):
            print(f"Chromatin accessibility dataset already exists in {acc_d_odir}")
            if os.path.exists(f'{acc_d_odir}/total_sampled.csv'):
                args.mode = 'fast'
            else:
                args.mode = 'full'
            return args, True
        else:
            print(f"Processing stage 1: prepare chromatin accessibility dataset")
            # 1.prepare peak bed files
            out_bed = f"{acc_d_odir}/total_peak.bed"
            cmd = f"""
            cat {args.acc_peak1} {args.acc_peak2} \
            | sort -k1,1 -k2,2n \
            | bedtools merge -i - \
            > {out_bed}
            """
            subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
            
            # 2. prepare overlap peak and chrombert regions
            total_peak_process = process(out_bed,files_dict['chrombert_region_file'],mode='region')[['chrom','start','end','build_region_index']].drop_duplicates().reset_index(drop=True)

            # 3. prepare background region for gene tss
            gene_meta = pd.read_csv(files_dict['gene_meta_tsv'],sep='\t').query("gene_biotype == 'protein_coding'")
            gene_tss_10kb = pd.DataFrame({'chrom':gene_meta['chrom'],'start':gene_meta['tss']-10000,'end':gene_meta['tss']+10000})
            gene_tss_10kb.to_csv(f'{acc_d_odir}/gene_tss_10kb.bed',sep='\t',index=False,header=None)
            
            gene_tss_10kb_process = process(f'{acc_d_odir}/gene_tss_10kb.bed',files_dict['chrombert_region_file'],mode='region').drop_duplicates(subset='build_region_index')[['chrom','start','end','build_region_index']]
            
            # 4.concat peak and background region (negtive peak gene tss region)
            total_region_processed = pd.concat([total_peak_process,gene_tss_10kb_process],axis=0).drop_duplicates().reset_index(drop=True)
            total_region_processed.to_csv(f'{acc_d_odir}/total_region_processed.csv',index=False)
            
            # 5. prepare chromatin accessibility signal
            cell1_signal = bw_getSignal_bins(args.acc_signal1,total_region_processed,scale=True,name='cell1_signal')
            cell2_signal = bw_getSignal_bins(args.acc_signal2,total_region_processed,scale=True,name='cell2_signal')
            
            total_region_signal = pd.concat([total_region_processed, cell1_signal,cell2_signal],axis=1)
            total_region_signal['log2_cell1_signal'] = np.log2(1+total_region_signal['cell1_signal'])
            total_region_signal['log2_cell2_signal'] = np.log2(1+total_region_signal['cell2_signal'])
            total_region_signal['label'] = total_region_signal['log2_cell2_signal'] - total_region_signal['log2_cell1_signal']
            if args.direction != '2-1':
                total_region_signal['label'] = -total_region_signal['label']
            
            total_region_signal.to_csv(f'{acc_d_odir}/total.csv',index=False)
            
            if args.mode == 'fast' and len(total_region_signal) > 20000:
                total_region_signal_sampled = total_region_signal.sample(n=20000,random_state=55).reset_index(drop=True)
                total_region_signal_sampled.to_csv(f'{acc_d_odir}/total_sampled.csv',index=False)
                split_data(total_region_signal_sampled, "_sampled", acc_d_odir)
            else:
                args.mode = 'normal'
                split_data(total_region_signal, "", acc_d_odir)
            
            # 6.up region and nochange region
            up_region = total_region_signal[total_region_signal["label"] > 1].sort_values("label", ascending=False).head(1000).reset_index(drop=True)
            total_region_signal['abs_label'] = np.abs(total_region_signal['label'])
            nochange_region = total_region_signal.query("cell1_signal > 0 or cell2_signal > 0").query("label <1 and label > -1").sort_values('abs_label').reset_index(drop=True).iloc[0:1000]
            up_region.to_csv(f'{acc_d_odir}/up.csv',index=False)
            nochange_region.to_csv(f'{acc_d_odir}/nochange.csv',index=False)   
                
            return args, True
    else:
        return args, False
        
def make_dataset(args,files_dict,exp_data_odir,acc_data_odir):
    exp_bool = make_exp_dataset(args,files_dict,exp_data_odir)
    args, acc_bool = make_acc_dataset(args,files_dict,acc_data_odir)
    if not acc_bool and not exp_bool:
        raise ValueError("you need to provide valid about chromatin accessibility files or gene expression files")
    return args, exp_bool, acc_bool


def generate_emb(model_tuned, data_config, sup_file, odir, name, files_dict):
    model_tuned = model_tuned.eval()
    model_emb = model_tuned.get_embedding_manager()
    data_config.supervised_file = sup_file
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
            total_counts += batch["input_ids"].shape[0]
            emb = model_emb(batch) # initialize the cache
            emb_np = emb.float().cpu().numpy()            
            embs_pool += emb_np.sum(axis=0)

        embs_pool /= total_counts

    embs_pool_dict = {regulator:embs_pool[regulator_idx_dict[regulator]] for regulator in regulators}
    out_pkl = os.path.join(odir, f"{name}_regulator_embs_dict.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(embs_pool_dict, f)
    
    return embs_pool, regulators


def infer_driver_factor(emb_odir, results_odir, data_config, model_tuned,data_odir,files_dict, source):
    up_emb, up_factors = generate_emb(model_tuned, data_config, f"{data_odir}/up.csv", emb_odir, 'up', files_dict)
    nochange_emb, nochange_factors = generate_emb(model_tuned, data_config, f"{data_odir}/nochange.csv", emb_odir, 'nochange', files_dict)
    assert up_factors == nochange_factors, "up and nochange factors are not the same"
    regulator_sim_df = factor_rank(up_emb, nochange_emb, up_factors, results_odir)
    if os.path.exists(files_dict["chrombert_factor_file"]):
        with open(files_dict["chrombert_factor_file"],"r") as f:
            factors = f.read().strip().split("\n")
            factors = [f.strip().lower() for f in factors]
            regulator_sim_df = regulator_sim_df.query("factors in @factors").sort_values(by='similarity').reset_index(drop=True)
    regulator_sim_df['rank']=regulator_sim_df.index + 1
    regulator_sim_df.to_csv(os.path.join(results_odir, "factor_importance_rank.csv"), index=False)
    print(f"Finished stage 4 {source}: infer driver factors in cell state transition (top 25):")
    print(regulator_sim_df.head(n=25))    
    
      
def run(args):
    """Main execution function for finding driver factors in cell state transitions."""
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    
    exp_odir = f'{odir}/exp'
    os.makedirs(exp_odir, exist_ok=True)
    acc_odir = f'{odir}/acc'
    os.makedirs(acc_odir, exist_ok=True)
    
    files_dict = resolve_paths(args)
    required_keys = [
        "chrombert_region_file",
        "hdf5_file",
        "pretrain_ckpt",
        "mtx_mask",
        "gene_meta_tsv"
    ]
    check_files(files_dict, required_keys=required_keys)
    
    # 1. prepare dataset
    print("Stage 1: prepare dataset")
    exp_data_odir = f'{exp_odir}/dataset'; os.makedirs(exp_data_odir, exist_ok=True)
    acc_data_odir = f'{acc_odir}/dataset'; os.makedirs(acc_data_odir, exist_ok=True)
    args, exp_bool, acc_bool = make_dataset(args, files_dict, exp_data_odir, acc_data_odir)
    print("Finished Stage 1")
    print(f"Whether to train ChromBERT to predict expression changes in cell state transition: {exp_bool}")
    print(f"Whether to train ChromBERT to predict chromatin accessibility changes in cell state transition: {acc_bool}")
    
    if exp_bool:
        # 2. train chrombert (exp)
        train_exp_odir = f"{exp_odir}/train"; os.makedirs(train_exp_odir, exist_ok=True)
        print("Processing stage 2 (exp): train ChromBERT to predict expression changes in cell state transition")

        if args.ft_ckpt_exp is not None:
            print(f"Use fine-tuned ChromBERT checkpoint file: {args.ft_ckpt_exp} to find driver factors in different expression activity genes")
            data_config = DatasetConfig(
                kind="MultiFlankwindowDataset",
                supervised_file=None,
                hdf5_file=files_dict["hdf5_file"],
                batch_size=args.batch_size,
                num_workers=2,
                meta_file=files_dict["meta_file"],
                flank_window=4,
            )
            model_config = ChromBERTFTConfig(
                genome=args.genome,
                task="gep",
                dropout=0,
                pretrain_ckpt=files_dict["pretrain_ckpt"],
                finetune_ckpt=args.ft_ckpt_exp,
                mtx_mask=files_dict["mtx_mask"],
                gep_flank_window=4
            )
            model_tuned = model_config.init_model().cuda()
            print("Finished Stage 2 (exp): use fine-tuned ChromBERT to find driver factors in different expression activity genes")
        else:
            print("Stage 2 (exp): train ChromBERT to predict expression changes in cell state transition")
            model_tuned, exp_train_odir, model_config, data_config = retry_train(args, files_dict, cal_metrics_regression, metcic='pearsonr', min_threshold=0.2, train_kind = 'regression', task="gep",odir=exp_odir)
            print("Finished stage 2 (exp): train ChromBERT to predict expression changes in cell state transition")

        # 4. infer driver factor in different expression activity genes
        print("Stage 3 (exp): infer driver factors in different expression activity genes")
        exp_emb_odir = f"{exp_odir}/emb"; os.makedirs(exp_emb_odir, exist_ok=True)
        exp_results_odir = f"{exp_odir}/results"; os.makedirs(exp_results_odir, exist_ok=True)
        infer_driver_factor(exp_emb_odir, exp_results_odir, data_config, model_tuned, exp_data_odir, files_dict, "exp")
        print("Finished stage 3 (exp)")
        
    
    if acc_bool:
        train_acc_odir = f"{acc_odir}/train"; os.makedirs(train_acc_odir, exist_ok=True)
        if args.ft_ckpt_acc is not None:
            print(f"Use fine-tuned ChromBERT checkpoint file: {args.ft_ckpt_acc} to find driver factors in different chromatin accessibility activity genes")
            data_config = DatasetConfig(
                kind="GeneralDataset",
                supervised_file=None,
                hdf5_file=files_dict["hdf5_file"],
                batch_size=args.batch_size,
                num_workers=8,
                meta_file=files_dict["meta_file"],
            )
            model_config = ChromBERTFTConfig(
                genome=args.genome,
                task="general",
                dropout=0,
                pretrain_ckpt=files_dict["pretrain_ckpt"],
                mtx_mask=files_dict["mtx_mask"],
                finetune_ckpt=args.ft_ckpt_acc,
            )
            model_tuned = model_config.init_model().cuda()
            print("Finished Stage 2 (acc): use fine-tuned ChromBERT to find driver factors in different chromatin accessibility activity genes")
        else:
            print("Stage 2 (acc): train ChromBERT to predict chromatin accessibility changes in cell state transition")
            model_tuned, acc_train_odir, model_config, data_config = retry_train(args, files_dict, cal_metrics_regression, metcic='pearsonr', min_threshold=0.2, train_kind = 'regression', task="general",odir=acc_odir)
            print("Finished stage 2 (acc): train ChromBERT to predict chromatin accessibility changes in cell state transition")

        # 4. infer driver factor in different chromatin accessibility activity genes
        print("Stage 3 (acc): infer driver factors in different chromatin accessibility activity genes")
        acc_emb_odir = f"{acc_odir}/emb"; os.makedirs(acc_emb_odir, exist_ok=True)
        acc_results_odir = f"{acc_odir}/results"; os.makedirs(acc_results_odir, exist_ok=True)
        infer_driver_factor(acc_emb_odir, acc_results_odir, data_config, model_tuned, acc_data_odir, files_dict, "acc")
        print("Finished stage 3 (acc)")
        
    if acc_bool and exp_bool:
        merge_odir = os.path.join(odir, "merge"); os.makedirs(merge_odir, exist_ok=True)
        print("Merging factor ranks from expression and chromatin accessibility")
        exp_rank_df = pd.read_csv(os.path.join(exp_results_odir, "factor_importance_rank.csv"))
        acc_rank_df = pd.read_csv(os.path.join(acc_results_odir, "factor_importance_rank.csv"))
        merge_df = pd.merge(exp_rank_df,acc_rank_df,on='factors',how='inner',suffixes=['_exp','_acc'])
        merge_df['total_rank']=((merge_df['rank_exp']+merge_df['rank_acc'])/2).rank().astype(int)
        merge_df = merge_df.sort_values('total_rank').reset_index(drop=True)
        merge_df.to_csv(os.path.join(merge_odir, "factor_importance_rank.csv"), index=False)
        print("Finished merging factor ranks from expression and chromatin accessibility")
        print("Top 25 driver factors:")
        print(merge_df.head(n=25))
    
    print("Finished all stages!")
    if exp_bool:
        if hasattr(args, 'ft_ckpt_exp') and args.ft_ckpt_exp:
            print(f"Used fine-tuned ChromBERT checkpoint: {args.ft_ckpt_exp}")
        else:
            print(f"Fine-tuned ChromBERT model (for expression changes) saved in: {exp_train_odir}")
        print(f"Driver factors for expression changes: {exp_results_odir}/factor_importance_rank.csv")
        
    if acc_bool:
        if hasattr(args, 'ft_ckpt_acc') and args.ft_ckpt_acc:
            print(f"Used fine-tuned ChromBERT checkpoint: {args.ft_ckpt_acc}")
        else:
            print(f"Fine-tuned ChromBERT model (for accessibility changes) saved in: {acc_train_odir}")
        print(f"Driver factors for chromatin accessibility changes: {acc_results_odir}/factor_importance_rank.csv")
        
    if acc_bool and exp_bool:
        print(f"Integrated driver factor rankings: {merge_odir}/factor_importance_rank.csv")

@click.command(
    name="find_driver_in_transition",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--exp-tpm1",
    "exp_tpm1",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Expression (TPM) file for cell state 1. CSV format with 'gene_id' and 'tpm' columns.",
)
@click.option(
    "--exp-tpm2",
    "exp_tpm2",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Expression (TPM) file for cell state 2. CSV format with 'gene_id' and 'tpm' columns.",
)
@click.option(
    "--acc-peak1",
    "acc_peak1",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Chromatin accessibility peak BED file for cell state 1.",
)
@click.option(
    "--acc-peak2",
    "acc_peak2",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Chromatin accessibility peak BED file for cell state 2.",
)
@click.option(
    "--acc-signal1",
    "acc_signal1",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Chromatin accessibility signal BigWig file for cell state 1.",
)
@click.option(
    "--acc-signal2",
    "acc_signal2",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Chromatin accessibility signal BigWig file for cell state 2.",
)
@click.option(
    "--direction",
    default="2-1",
    show_default=True,
    type=click.Choice(["2-1", "1-2"], case_sensitive=False),
    help="Direction of cell state transition: '2-1' means from state 1 to state 2; '1-2' means from state 2 to state 1.",
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
    help="Reference genome.",
)
@click.option(
    "--resolution",
    default="1kb",
    show_default=True,
    type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
    help="ChromBERT resolution.",
)
@click.option(
    "--mode",
    default="fast",
    show_default=True,
    type=click.Choice(["fast", "normal"], case_sensitive=False),
    help="Training mode: 'fast' downsamples to 20k regions for quicker training; 'normal' uses all regions.",
)
@click.option(
    "--ft-ckpt-exp",
    "ft_ckpt_exp",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Fine-tuned ChromBERT checkpoint file for expression changes.",
)
@click.option(
    "--ft-ckpt-acc",
    "ft_ckpt_acc",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Fine-tuned ChromBERT checkpoint file for chromatin accessibility changes.",
)
@click.option(
    "--chrombert-cache-dir",
    "chrombert_cache_dir",
    default="~/.cache/chrombert/data",
    show_default=True,
    type=click.Path(file_okay=False),
    help="ChromBERT cache directory (containing config/ and anno/ subfolders).",
)
@click.option(
    "--batch-size",
    "batch_size",
    default=4,
    show_default=True,
    type=int,
    help="Batch size. Increase this value if you have sufficient GPU memory.",
)

def find_driver_in_transition(
    exp_tpm1,
    exp_tpm2,
    acc_peak1,
    acc_peak2,
    acc_signal1,
    acc_signal2,
    direction,
    odir,
    genome,
    resolution,
    mode,
    ft_ckpt_exp,
    ft_ckpt_acc,
    chrombert_cache_dir,
    batch_size
):
    """
    Find driver factors in cell state transitions.
    
    This tool identifies key transcription factors that drive cell state transitions
    by analyzing changes in gene expression and/or chromatin accessibility between
    two cell states.
    
    You must provide at least one of the following:
    - Expression data (--exp-tpm1 and --exp-tpm2)
    - Accessibility data (--acc-peak1, --acc-peak2, --acc-signal1, --acc-signal2)
    
    Providing both expression and accessibility data yields more confident results.
    """
    args = SimpleNamespace(
        exp_tpm1=exp_tpm1,
        exp_tpm2=exp_tpm2,
        acc_peak1=acc_peak1,
        acc_peak2=acc_peak2,
        acc_signal1=acc_signal1,
        acc_signal2=acc_signal2,
        direction=direction,
        odir=odir,
        genome=genome,
        resolution=resolution,
        mode=mode,
        ft_ckpt_exp=ft_ckpt_exp,
        ft_ckpt_acc=ft_ckpt_acc,
        chrombert_cache_dir=chrombert_cache_dir,
        batch_size = batch_size,
    )
    run(args)


if __name__ == "__main__":
    find_driver_in_transition()