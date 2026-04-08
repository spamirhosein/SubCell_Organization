# Stage 2 MIBI data prep (directory-based stacks)

This guide adapts Stage 2 preprocessing to MIBI’s per-FOV channel folders while keeping the existing SOLAR Stage 2 pipeline intact.

## Required folder layout
- Intensity images: `image_data/{fov}/{channel}.tiff`
- Cell masks: `segmentation/cleaned_segmasks/{fov}_cleaned_mask.tiff`
- Nuclear masks: `segmentation/cellpose_output/{fov}_nuclear.tiff`

## 1) Build Stage 2 tables
```bash
python -m solar.datasets.build_stage2_tables_mibi \
  --cell_table manifests/cells.parquet \
  --image_root image_data \
  --cell_mask_root segmentation/cleaned_segmasks \
  --nuc_mask_root segmentation/cellpose_output \
  --out_cell_table manifests/cell_table_stage2.parquet \
  --out_fov_table manifests/fov_table_stage2.parquet
```

## 2) Compute per-channel stats (masked by cell pixels)
```bash
python -m solar.datasets.compute_channel_stats \
  --fov_table manifests/fov_table_stage2.parquet \
  --channel_names CH1 CH2 CH3 \
  --mask_threshold 0.0 \
  --out stage_stats/channel_stats.json
```

## 3) Export canonicalized crops + Stage 2 manifest
```bash
python -m solar.datasets.export_stage2_crops \
  --cell_table manifests/cell_table_stage2.parquet \
  --channel_names CH1 CH2 CH3 \
  --use_centroids --x_column X --y_column Y \
  --save_masks \
  --out_manifest manifests/stage2_manifest.parquet
```

## 4) Train Stage 2
```bash
python -m solar.train.train_solar_map_vae \
  --manifest manifests/stage2_manifest.parquet \
  --channel_stats stage_stats/channel_stats.json
```

Notes:
- `channel_names` is required when `stack_path` is a directory.
- If any channel TIFF is missing, the pipeline will raise an error with the missing filename.
- If centroid-based crops miss the target cell, the exporter will fall back to the mask-based centroid.
