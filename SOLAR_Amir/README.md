# SOLAR Stage 1 (SolarShapeVAE) Quickstart

This repo currently implements Stage 1 (SolarShapeVAE) with mask-only training, combined-mask support, manifest tooling, and train/test logging.

## Data Preparation
1) **Generate combined masks (0 background, 1 cytoplasm, 2 nucleus) per cell**
   ```bash
   python -m solar.datasets.make_combined_masks \
     --cell_dir /path/to/segmentation/cleaned_mask_dir \
     --nuclear_dir /path/to/segmentation/nuclear_mask_dir \
     --out_dir /path/to/SCALER/SCALER_masks \
     --framesize 256 --downsample 2
   ```

2) **Build a manifest for Stage 1** (single label-map PNG/TIFF per row):
   ```bash
   python -m solar.datasets.build_mask_manifest \
     --nucleus_dir /path/to/SCALER/SCALER_masks \
     --combined_mask \
     --infer_sample_id \  # derive sample_id from filename stem before `_cell_`
     --relative_to /path/to/base (optional) \
     --out manifests/masks_label.tsv
   ```
   - Format with `--combined_mask`: `labelmap_path<TAB>sample_id`
   - If you prefer a constant sample_id, omit `--infer_sample_id` and set `--sample_id <id>`.
   - To create train/val/test splits, make separate manifest files (e.g., shuffle/split via a small script).

## Training SolarShapeVAE
Example run (train/test split done inside the script, 80/20, seeded):
```bash
python -m solar.train.train_solar_shape_vae \
  --mask_manifest manifests/masks_label.tsv \
  --mask_root /path/to/base \  # set if manifest paths are relative
  --combined_mask \
  --val_background 0 --val_cytoplasm 1 --val_nucleus 2 \
  --checkpoint checkpoints/solar_shape_vae.pt \
  --batch_size 32 --epochs 100 --high_res_size 256 --low_res_size 128 \
  --latent_dim 16 --group_N 8 --seed 0
```
Notes:
- The script splits the provided manifest into train/test (80/20) with a seeded `random_split`.
- Train loader uses `BalancedBatchSampler` on the train subset; test loader is plain (no oversampling).
- Recon grids, test metrics, and optional embeddings are taken from the **test** subset.

## Logged Information
- **TensorBoard** (default log dir `runs/solarshape_vae`, or override `--log_dir`):
  - Train: loss_total, loss_bce, loss_kl, warmup_beta (per step)
  - Test: loss_total, loss_bce, loss_kl (per epoch, beta=1.0)
  - Images: recon grids for nucleus/cell channels from the test subset
  - Weights: histograms per epoch
  ```bash
  tensorboard --logdir runs/solarshape_vae --port 6006 --host 0.0.0.0
  ```

- **Checkpoint**: saved to `--checkpoint` path (default `checkpoints/solar_shape_vae.pt`).
- **Embeddings export** (optional): `--embeddings_out path.parquet` dumps mu from the test loader (up to `--embed_limit`).

## Key Scripts
- `solar/datasets/make_combined_masks.py`: crops per-cell combined masks from segmentation outputs.
- `solar/datasets/build_mask_manifest.py`: builds manifest TSVs; supports `--combined_mask` and `--infer_sample_id`.
- `solar/train/train_solar_shape_vae.py`: training loop with seeded train/test split, TensorBoard logging, checkpoints, and embedding export.

---

# SOLAR: Pixel-Wise Positivity Probability Maps for MIBI Images

## Overview
This tool generates **pixel-wise positivity probability maps** for organelle markers in multiplexed MIBI images. It is designed to handle:
- Inter-sample variability.
- Free-form background/haze.
- Salt-and-pepper/hot-pixel artifacts.

The output is a **float32 probability map** \(P(x) \in [0, 1]\) representing the likelihood that a pixel belongs to a true organelle signal.

## Installation
Ensure the following dependencies are installed:
- `numpy`
- `scipy`
- `tifffile`
- `nellie` (for thresholding utilities)
- `scikit-image` (for connected component analysis)

## Usage
### Command-Line Interface
The CLI processes TIFF or NumPy files and outputs the probability map as a TIFF file.

```bash
python solar/cli/positivity_map_cli.py input.tif output.tif \
    --asinh_cofactor 1.0 \
    --gaussian_sigmas_px 1.5 2.3 3.0 \
    --z_windows_px 31 101 \
    --nbins 256 \
    --low_percentile 1.0 \
    --sigmoid_slope 0.7 \
    --tile_size 256 \
    --tile_overlap 64 \
    --min_component_area_px 30 \
    --debug
```

### Parameters
| Parameter               | Default       | Description |
|-------------------------|---------------|-------------|
| `asinh_cofactor`        | `1.0`         | Cofactor for asinh transformation. |
| `gaussian_sigmas_px`    | `[1.5, 2.3, 3.0]` | Gaussian smoothing sigmas in pixels. |
| `z_windows_px`          | `[31, 101]`   | Window sizes for spatial z-scoring. |
| `nbins`                 | `256`         | Number of bins for histogram computation. |
| `low_percentile`        | `1.0`         | Percentile for masking low-intensity pixels. |
| `sigmoid_slope`         | `0.7`         | Slope for sigmoid mapping. |
| `tile_size`             | `256`         | Size of tiles for gating. |
| `tile_overlap`          | `64`          | Overlap between tiles. |
| `min_component_area_px` | `30`          | Minimum area for connected components. |

### Outputs
- **Probability Map**: Saved as a float32 TIFF file.
- **Debug Information** (optional): Saved as a NumPy `.npy` file.

## Example
Input: `example_input.tif` (2046×2046 MIBI image)

```bash
python solar/cli/positivity_map_cli.py example_input.tif example_output.tif --debug
```

Output:
- `example_output.tif`: Probability map.
- `example_output_debug.npy`: Debug information.

## Testing
Run unit tests to validate the implementation:
```bash
pytest tests/test_positivity_probability_map.py
```

## Notes
- The tool is GPU-compatible if `nellie` is configured with CuPy.
- Default parameters are optimized for 2046×2046 MIBI images with 90 nm pixel size.
