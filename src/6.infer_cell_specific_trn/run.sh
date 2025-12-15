

export CUDA_VISIBLE_DEVICES=0
nohup singularity exec --nv \
  --bind /mnt/Storage2:/mnt/Storage2 \
  /mnt/Storage2/home/chenqianqian/projects/chrombert/chrombert_tools/Singularity/chrombert_20251212.sif \
  python run.py \
    --cell_type_bw /mnt/Storage2/home/chenqianqian/projects/chrombert/meta_data/meta_data_hm_1kb/data/demo/transdifferentiation/chrom_accessibility/myoblast_ENCFF149ERN_signal.bigwig \
    --cell_type_peak /mnt/Storage2/home/chenqianqian/projects/chrombert/meta_data/meta_data_hm_1kb/data/demo/transdifferentiation/chrom_accessibility/myoblast_ENCFF647RNC_peak.bed \
    --odir output \
    --chrombert_cache_dir /mnt/Storage2/home/chenqianqian/projects/chrombert/meta_data/meta_data_hm_1kb/data > 9.infer_cell_specific_trn.log 2>&1 &