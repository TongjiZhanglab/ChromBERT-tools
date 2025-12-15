#!/bin/bash
# This script finds driver factors in different functional regions.
# If you have identified a dual-functional regulator, you can use this script to find its dual networks by specifying the --dual_regulator parameter.


export CUDA_VISIBLE_DEVICES=0 
datadir="/mnt/Storage2/home/chenqianqian/projects/chrombert/review_cg/2.ft_wider/meta_data_1kb/data/demo/ezh2"
cache_dir="/mnt/Storage2/home/chenqianqian/projects/chrombert/meta_data/meta_data_hm_1kb/data"
odir="output2"

nohup singularity exec --nv \
  --bind /mnt/Storage2:/mnt/Storage2 \
  /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/Singularity/chrombert_20251215.sif \
  python run.py \
    --function1_bed "${datadir}/hESC_GSM1003524_EZH2.bed;${datadir}/hESC_GSM1498900_H3K27me3.bed" \
    --function2_bed "${datadir}/hESC_GSM1003524_EZH2.bed" \
    --function1_mode and \
    --function2_mode and \
    --odir ${odir} \
    --dual_regulator ezh2 \
    --ignore_regulator "h3k27me3" \
    --chrombert_cache_dir ${cache_dir} > "run.stdout.log" \
  2> "run.stderr.log" &
