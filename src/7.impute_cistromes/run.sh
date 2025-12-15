export CUDA_VISIBLE_DEVICES=0
nohup singularity exec --nv \
  --bind /mnt/Storage2:/mnt/Storage2 \
  /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/Singularity/chrombert_20251212.sif \
  python run.py \
    --region_bed /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/test/0.tmp_region/CTCF_ENCFF664UGR_sample10000.bed \
    --odir output2 \
    --cistrome "BCL11A:GM12878;BRD4:MCF7;CTCF:HepG2" \
    --chrombert_cache_dir /mnt/Storage2/home/chenqianqian/projects/chrombert/meta_data/meta_data_hm_1kb/data > run.log 2>&1 &