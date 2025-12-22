import os
import click
from types import SimpleNamespace
import numpy as np
import pandas as pd
import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
import torch
from tqdm import tqdm
from .utils import resolve_paths, check_files, check_region_file

    
def model_emb_func(args,files_dict,odir):
    # init datamodule
    data_config = DatasetConfig(
        kind = "GeneralDataset",
        supervised_file = f"{odir}/model_input.tsv",
        hdf5_file = files_dict["hdf5_file"],
        batch_size = 4,
        num_workers = 8,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()

    # init chrombert
    model_config = ChromBERTFTConfig(
        genome = args.genome,
        dropout = 0,
        task = "general",
        pretrain_ckpt = files_dict["pretrain_ckpt"],
        mtx_mask = files_dict["mtx_mask"],
    )
    model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
    return ds, dl, model_emb
    
def run(args, return_data=False):

    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    focus_region = args.region
    files_dict = resolve_paths(args)
    check_files(files_dict, required_keys=[
        "chrombert_region_file",
        "hdf5_file",
        "pretrain_ckpt"])

    # ---------- overlapping focus regions ----------
    overlap_bed = check_region_file(focus_region,files_dict,odir)
    overlap_idx = overlap_bed["build_region_index"].to_numpy()

    # ---------- focus region embeddings ----------
    emb_npy_path = files_dict["region_emb_npy"]
    if os.path.exists(emb_npy_path):
        all_emb = np.load(emb_npy_path)
        overlap_emb = all_emb[overlap_idx]
        np.save(f"{odir}/overlap_region_emb.npy", overlap_emb)
    else:
        print(f"ChromBERT region embedding file not found: {emb_npy_path}, and not directly pick region embedding from cache dir.")
        print("Load model ChromBERT to embed focus regions.")
        ds, dl, model_emb = model_emb_func(args,files_dict,odir)
    
        region_embs = []
        with torch.no_grad():
            for batch in tqdm(dl, total = len(dl)):
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                model_emb(batch) # initialize the cache 
                region_embs.append(model_emb.get_region_embedding().float().cpu().detach())
        region_embs = torch.cat(region_embs,axis=0).numpy()
        np.save(f"{odir}/overlap_region_emb.npy", region_embs)
        

    # ---------- report ----------
    total_focus = sum(1 for _ in open(focus_region))
    no_overlap_region_len = sum(1 for _ in open(f"{odir}/no_overlap_region.bed"))

    print("Finished!")
    print(
        f"Focus region summary - total: {total_focus}, "
        f"overlapping with ChromBERT: {overlap_bed.shape[0]}, It is possible for a single region to overlap multiple ChromBERT regions,"
        f"non-overlapping: {no_overlap_region_len}"
    )
    print("Overlapping focus regions BED file:", f"{odir}/overlap_region.bed")
    print("Non-overlapping focus regions BED file:", f"{odir}/no_overlap_region.bed")
    print("Overlapping focus region embeddings saved to:", f"{odir}/overlap_region_emb.npy")
    
    if return_data:
        return overlap_emb, overlap_bed


@click.command(name="embed_region",
               context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region file.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False), help="Resolution. Mouse only supports 1kb resolution.")

@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default=os.path.expanduser("~/.cache/chrombert/data"),
              show_default=True,
              type=click.Path(file_okay=False),
              help="ChromBERT cache dir. Default is ~/.cache/chrombert/data. if you use command `chrombert_prepare_env` to download the data, you don't need to provide this.")

@click.option("--chrombert-region-file", "chrombert_region_file",
              default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="ChromBERT region BED file. If not provided, use the default hg38_6k_1kb_region.bed in the cache dir.")
@click.option("--chrombert-region-emb-file", "chrombert_region_emb_file",
              default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="ChromBERT region embedding file. If not provided, use the default hm_1kb_all_region_emb.npy in the cache dir.")



def embed_region(region, odir, genome, resolution, chrombert_cache_dir,chrombert_region_file, chrombert_region_emb_file):      
    '''
    Extract general region embeddings
    '''
    args = SimpleNamespace(
        region=region,
        odir=odir,
        genome=genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file
    )
    run(args)