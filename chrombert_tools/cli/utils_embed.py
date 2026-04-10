import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from chrombert_hf import ChromBERTFTConfig, DatasetConfig, ChromBERTConfig
from transformers import AutoModel
from .utils import (
    get_model_name,
    cal_metrics_regression,
    HDF5Manager,
    overlap_regulator_func
)
from .utils_train_cell import make_dataset, retry_train
from typing import Any, Optional, Tuple


def is_cell_specific(args):
    '''
    Check if the embedding is cell-specific
    '''
    return (
        args.ft_ckpt is not None
        or (args.cell_type_bw is not None and args.cell_type_peak is not None)
    )

def build_dataloader(supervised_file, hdf5_file, batch_size, num_workers=8):
    '''
    Build dataloader
    '''
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=supervised_file,
        hdf5_file=hdf5_file,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()
    return ds, dl

def get_required_keys(args):
    """
    Get required cache keys for current mode.
    """
    # required = [
    #     "chrombert_region_file",
    #     "chrombert_regulator_file",
    #     "hdf5_file",
    #     "pretrain_ckpt",
    # ]
    # if is_cell_specific(args):
    #     required.append("mtx_mask")

    required = [
        "chrombert_region_file",
        "chrombert_regulator_file",
        "hdf5_file",
    ]
    return required


def resolve_ignore_object(
    ignore_regulator: Optional[str], chrombert_regulator_file: str
) -> Tuple[bool, Optional[str]]:
    """
    Map CLI ignore string to dataset/model ignore_object.

    Returns:
        (ignore, ignore_object) where ignore_object is ';'-joined ChromBERT names or None.
    """
    if ignore_regulator is None:
        return False, None
    overlap_ignore, _, _ = overlap_regulator_func(
        ignore_regulator, chrombert_regulator_file
    )
    ignore_object = ";".join(overlap_ignore) if overlap_ignore else None
    return bool(ignore_object), ignore_object

def build_model_dataset_config(
    args: Any,
    files_dict: dict,
    supervised_file_for_ignore_idx: str,
    gep: bool = False,
    flank_window: int = 4,
    ignore_regulator: Optional[str] = None,
) -> Tuple[DatasetConfig, ChromBERTFTConfig]:
    """
    Build DatasetConfig + ChromBERTFTConfig like interpret_regulators_across_regions.

    args must provide: genome, resolution, batch_size, ft_ckpt.
    gep / flank_window: taken from args if present, else use the function parameters
    (defaults: gep=False, flank_window=4). Optional: ignore_regulator on args.

    supervised_file_for_ignore_idx: TSV used only to init_dataset when computing ignore_index.
    """
    ignore_regulator = getattr(args, "ignore_regulator", ignore_regulator)
    ignore, ignore_object = resolve_ignore_object(
        ignore_regulator,
        files_dict["chrombert_regulator_file"],
    )
    ignore_index = None
    gep = getattr(args, "gep", gep)
    flank_window = getattr(args, "flank_window", flank_window)
    if not gep:
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
            ds0 = data_config.init_dataset(supervised_file=supervised_file_for_ignore_idx)
            ignore_index = ds0[0]["ignore_index"]

        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task="general",
            dropout=0,
            pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            finetune_ckpt=args.ft_ckpt,
            ignore=ignore,
            ignore_index=ignore_index,
        )
    else:
        data_config = DatasetConfig(
            kind="MultiFlankwindowDataset",
            supervised_file=None,
            hdf5_file=files_dict["hdf5_file"],
            batch_size=args.batch_size,
            num_workers=2,
            meta_file=files_dict["meta_file"],
            flank_window=flank_window,
        )
        if ignore:
            data_config.ignore = ignore
            data_config.ignore_object = ignore_object
            ds0 = data_config.init_dataset(supervised_file=supervised_file_for_ignore_idx)
            ignore_index = ds0[0]["ignore_index"]

        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task="gep",
            dropout=0,
            pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            finetune_ckpt=args.ft_ckpt,
            gep_flank_window=flank_window,
            ignore=ignore,
            ignore_index=ignore_index,
        )

    return data_config, model_config


def build_cell_model_emb(args, files_dict,odir):
    '''
    Build cell-specific model and embedding manager
    '''
    
    if args.cell_type_peak is not None and args.cell_type_bw is not None:
        d_odir = f"{odir}/dataset"
        os.makedirs(d_odir, exist_ok=True)
        print("Preparing dataset ...")
        make_dataset(args.cell_type_peak, args.cell_type_bw, d_odir, files_dict, args.mode)
        sup_file = f"{d_odir}/background_region.csv"
    
    # 1) load ft ckpt if provided
    if args.ft_ckpt is not None:
        print(f"Using provided fine-tuned checkpoint: {args.ft_ckpt}")
        if sup_file is None:
            sup_file = f"{odir}/model_input.tsv" if args.region is not None else f"{odir}/model_input_gene.tsv"
        data_config, model_config = build_model_dataset_config(args, files_dict, supervised_file_for_ignore_idx=sup_file) # this file only used for ignore_index (init, dataset not used)
        model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
        return model_emb, data_config

    # 2) no ft ckpt, train cell-specific model on the fly
    train_odir = f"{odir}/train"
    os.makedirs(train_odir, exist_ok=True)

    print("Fine-tuning cell-specific model...")
    model_tuned, train_odir, model_config, data_config = retry_train(
        args,
        files_dict,
        cal_metrics_regression,
        metcic="pearsonr",
        min_threshold=0.4,
    )
    model_emb = model_tuned.get_embedding_manager().cuda().bfloat16()
    return model_emb, data_config


def batch_num_regions(batch: dict) -> int:
    """Batch size key for GeneralDataset ('region') vs GEP ('center_region')."""
    if "region" in batch:
        return batch["region"].shape[0]
    return batch["center_region"].shape[0]


def generate_embeddings(dl, model_emb):
    '''
    Generate embeddings from chrombert (forward pass)
    '''
    region_embs = []
    with torch.no_grad():
        for batch in tqdm(dl, total=len(dl)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            model_emb(batch)
            region_embs.append(model_emb.get_region_embedding().float().cpu().detach())

    return torch.cat(region_embs, dim=0).numpy()


def generate_regulator_embeddings(ds, dl, model_emb, overlap_bed, regulator_idx_dict, odir, oname, return_data):
    '''
    Generate regulator embeddings from chrombert (forward pass)
    '''
    shapes = {f"emb/{k}": [(len(ds), 768), np.float16] for k in regulator_idx_dict}
    reg_sums = {name: np.zeros(768, dtype=np.float64) for name in regulator_idx_dict}
    reg_emb_dict = {}
    total_counts = 0

    out_h5 = f"{odir}/region_aware_{oname}.hdf5"
    with HDF5Manager(out_h5, region=[(len(ds), 4), np.int64], **shapes) as h5:
        with torch.no_grad():
            for batch in tqdm(dl, total=len(dl), desc="Computing regulator embeddings"):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()

                model_emb(batch)

                bs = batch_num_regions(batch)
                start_idx = total_counts
                total_counts += bs
                end_idx = total_counts

                idx_key = (
                    "build_region_index"
                    if "build_region_index" in batch
                    else "center_build_region_index"
                )
                batch_index = batch[idx_key].long().cpu().numpy().reshape(-1)
                region = overlap_bed[["chrom", "start", "end", "build_region_index"]].iloc[start_idx:end_idx].values # the region of users' interest
                assert (batch_index == region[:, -1].reshape(-1)).all(), "Batch index and region index do not match"

                embs = {
                    f"emb/{k}": model_emb.get_regulator_embedding(k).float().cpu().numpy()
                    for k in regulator_idx_dict
                }
                h5.insert(region=region, **embs)

                for reg_name in regulator_idx_dict:
                    emb_np = model_emb.get_regulator_embedding(reg_name).float().cpu().numpy()
                    reg_sums[reg_name] += emb_np.sum(axis=0)
                    if return_data:
                        if reg_name not in reg_emb_dict:
                            reg_emb_dict[reg_name] = []
                        reg_emb_dict[reg_name].append(emb_np)

    if return_data:
        for key in reg_emb_dict:
            reg_emb_dict[key] = np.concatenate(reg_emb_dict[key], axis=0)
    else:
        reg_emb_dict = None

    reg_means = {
        reg_name: (sum_vec / total_counts)
        for reg_name, sum_vec in reg_sums.items()
    }
    out_pkl = os.path.join(odir, f"mean_{oname}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(reg_means, f)
    return reg_means, reg_emb_dict


def generate_regulator_embeddings_only_mean(
    dl,
    model_emb: torch.nn.Module,
    odir: str,
    oname: str,
):
    """
    Sum regulator embeddings over all samples in dataloader(supervised_file=sup_file).

    region_name: optional name for error messages (e.g. 'region1').
    """
    regulators = model_emb.regulator_names
    embs_sum = np.zeros((len(regulators), 768), dtype=np.float64)
    total_counts = 0

    iterator = tqdm(dl)
    with torch.no_grad():
        for batch in iterator:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            emb = model_emb(batch)
            emb_np = emb.float().cpu().numpy()
            total_counts += batch_num_regions(batch)
            embs_sum += emb_np.sum(axis=0)

    if total_counts == 0:
        loc = f"region {oname!r} " if oname else ""
        raise ValueError(
            f"No batches for {loc}). "
            "Check region overlap and model_input.tsv."
        )
    regulator_idx_dict = {regulator: idx for idx, regulator in enumerate(regulators)}
    embs_pool = embs_sum / total_counts
    embs_pool_dict = {
        regulator: embs_pool[regulator_idx_dict[regulator]] for regulator in regulators
    }
    out_pkl = os.path.join(odir, f"mean_regulator_emb_{oname}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(embs_pool_dict, f)
    return embs_pool, regulators