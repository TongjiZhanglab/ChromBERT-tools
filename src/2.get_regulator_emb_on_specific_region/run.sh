singularity exec --nv \
  --bind /mnt/Storage2:/mnt/Storage2 \
  /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/Singularity/chrombert_bedtools.sif \
  python run.py \
    --region_bed /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/test/0.tmp_region/CTCF_ENCFF664UGR_sample10000.bed \
    --regulator "EZH2;BRD4;CTCF;FOXA3" \
    --odir output \
    --chrombert_cache_dir /mnt/Storage2/home/chenqianqian/projects/chrombert/meta_data/meta_data_hm_1kb/data