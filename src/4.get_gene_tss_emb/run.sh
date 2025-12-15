singularity exec --nv \
  --bind /mnt/Storage2:/mnt/Storage2 \
  /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/Singularity/chrombert_bedtools.sif \
  python run.py \
    --gene "ENSG00000170921;TANC2;ENSG00000200997;DPYD;SNORA70" \
    --odir output \
    --chrombert_cache_dir /mnt/Storage2/home/chenqianqian/projects/chrombert/meta_data/meta_data_hm_1kb/data