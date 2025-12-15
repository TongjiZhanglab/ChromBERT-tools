# singularity exec --nv \
#   --bind /mnt/Storage2:/mnt/Storage2 \
#   /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/Singularity/chrombert_bedtools.sif \
#   python 8.infer_trn.py \
#     --region_bed /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/test/0.tmp_region/CTCF_ENCFF664UGR_sample10000.bed \
#     --odir output \
#     --regulator "EZH2;BRD4;CTCF" \
#     --chrombert_cache_dir /mnt/Storage2/home/chenqianqian/projects/chrombert/meta_data/meta_data_hm_1kb/data

singularity exec --nv \
  --bind /mnt/Storage2:/mnt/Storage2 \
  /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/Singularity/chrombert_bedtools.sif \
  python run.py \
    --region_bed /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/test/0.tmp_region/CTCF_ENCFF664UGR_sample10000.bed \
    --odir output2 \
    --chrombert_cache_dir /mnt/Storage2/home/chenqianqian/projects/chrombert/meta_data/meta_data_hm_1kb/data