# Stage 2 plan (stacked 128×128) — with per-cell canonicalization + masked global normalization + saving 256×256 for Stage 3

## Goal
Train SolarMapVAE (Stage 2) to learn “where” each marker localizes from stacked 128×128 single-cell crops, conditioned on:
- Stage 1 morphology posterior (mu/logvar → resampled `z_shape` each step)
- `cond_cell` (lineage marker means per cell)
- `cond_sample` (subcellular marker medians per FOV)

During export, also save aligned canonicalized 256×256 stacks per cell for Stage 3 (texture), and normalize with global per-channel mean/std computed from full FOV images using only pixels inside cell masks.

---

## A. Data preparation

### A1. Inputs
- FOV multi-TIFF (multi-plane image; planes correspond to markers).
- Cleaned cell mask per FOV (integer ID per cell; 0 = background).
- Nuclear mask per FOV (for nucleus-aware canonicalization; if missing, canonicalization becomes cell-only).
- Cell table (one row per cell) with:
  - `fov_name`
  - `cell_mask_id`
  - lineage marker intensities
  - subcellular marker intensities
  - other marker intensities

### A2. Canonical channel order
- Define a fixed ordered list:
  - `channel_names = [marker_1, marker_2, ..., marker_C]`
- Ensure it matches the multi-TIFF plane reading order.
- This order is a hard contract for export, dataset loading, model I/O, and interpretation.

### A3. Stable identifiers
- `cell_id = f"{fov_name}__{cell_mask_id}"`
- `sample_id = int` per `fov_name` (map `fov_name -> 0..N-1`)

---

## A4. Canonicalization reference (Stage 1) + how to reuse it
- Canonicalization must be applied **per-cell crop level**, not at whole-FOV level.
- Reuse the Stage 1 logic in `solar/datasets/make_combined_masks.py`, especially:
  - `_crop_and_label(...)` semantics (0 background, 1 cytoplasm, 2 nucleus)
  - `_align_label(...)` canonicalization behavior (translation + rotation + reflection)

**Implementation requirement:**  
Do not just canonicalize the labelmap and ignore intensity. The same transforms must be applied to both:
- the per-cell label crop, and
- the stacked intensity crop (all channels).

**Recommended approach:**  
Implement a paired canonicalization helper that follows the same steps as `_align_label` but returns/apply the transform to both label + intensity stack.

---

## A5. Per-cell export pipeline (256×256 canonicalized → save → downsample to 128×128)
For each row in the cell table:

1) Load FOV intensity stack `stack_full` in `channel_names` order (shape `(C, H, W)`).

2) Load cleaned integer-ID cell mask `id_mask` (shape `(H, W)`).

3) Load nuclear mask `nuc_mask` if available (shape `(H, W)`).

4) Compute crop center from `id_mask == cell_mask_id` (centroid).

5) Crop 256×256 window (pad with 0 if needed):
- `x256_raw`: `(C,256,256)` from `stack_full`
- `id256`: `(256,256)` from `id_mask`
- `nuc256`: `(256,256)` from `nuc_mask` (optional)

6) Build combined labelmap `label256_raw` with Stage 1 semantics:
- 1 where `id256 == cell_mask_id`
- 2 where `(id256 == cell_mask_id) & (nuc256 > 0)`
- 0 elsewhere

7) Canonicalize **per cell** (Stage 1 behavior), applying identical transforms to label + intensity:
- `label256 = canonicalize(label256_raw)`  (Stage 1 logic)
- `x256 = apply_same_transform(x256_raw)`  (channel-wise, same transform ops)

8) Define canonical cell mask:
- `cellmask256 = (label256 > 0)`

9) Downsample for Stage 2:
- `x128_raw = downsample_256_to_128(x256)` using area/average pooling → `(C,128,128)`
- `cellmask128 = downsample_256_to_128(cellmask256.astype(float)) > 0.5` → `(128,128)`

10) Save raw stacks (float32):
- Stage 3 (raw):
  - `stage_crops_256/<fov_name>/<cell_id>.pt` containing `x256` `(C,256,256)`
- Stage 2 (raw):
  - `stage_crops_128/<fov_name>/<cell_id>.pt` containing `x128_raw` `(C,128,128)`

11) (Recommended) Save per-cell canonical masks:
- `stage_masks_256/<fov_name>/<cell_id>.pt` containing `cellmask256`
- `stage_masks_128/<fov_name>/<cell_id>.pt` containing `cellmask128`

12) (Optional) Save canonicalization metadata for debugging:
- long-axis rotation angle
- flips/rotations used for nucleus quadrant rule

---

## A6. Global per-channel normalization stats (from FOV images; only pixels in cell masks)
**Policy:** Compute mean/std per channel from the raw FOV images, using only pixels inside cell masks (`id_mask > 0`). Padded pixels are excluded automatically.

Two-pass approach (recommended):

**Pass 1 (mean)**
- For each FOV:
  - load `stack_full (C,H,W)` and `id_mask (H,W)`
  - `mask_all_cells = (id_mask > 0)`
  - for each channel `c`:
    - `sum[c] += stack_full[c][mask_all_cells].sum()`
    - `count[c] += mask_all_cells.sum()`
- `mean[c] = sum[c] / count[c]`

**Pass 2 (std)**
- For each FOV:
  - load `stack_full` and `mask_all_cells`
  - for each channel `c`:
    - `sumsq[c] += ((stack_full[c][mask_all_cells] - mean[c])**2).sum()`
- `std[c] = sqrt(sumsq[c] / count[c])`, clamp `std >= 1e-6`

Persist stats:
- `stage_stats/channel_stats.(json|pt)` containing:
  - `channel_names` (ordered)
  - `mean: float[C]`
  - `std: float[C]`

---

## A7. Conditioning vectors (from cell table)
- `cond_cell` (per cell):
  - lineage marker columns → global z-score across all cells
- `cond_sample` (per FOV):
  - group by `fov_name`
  - take medians of subcellular marker columns
  - global z-score across FOVs
  - join back to each cell

---

## A8. Stage 1 morphology stats (for Stage 2 conditioning; used later in Stage 3 too)
- Export per `cell_id`:
  - `mu_shape (D_shape)`
  - `logvar_shape (D_shape)`
- Join to Stage 2 manifest by `cell_id`.

---

## A9. Unified manifest table (Parquet recommended)
Per cell row include:
- `cell_id`, `fov_name`, `sample_id`, `cell_mask_id`
- `stack128_path`, `stack256_path`
- (recommended) `mask128_path`, `mask256_path`
- `cond_cell` (vector columns or sidecar pointer)
- `cond_sample` (vector columns or sidecar pointer)
- `mu_shape`, `logvar_shape` (or sidecar pointer)
- (optional) canonicalization metadata
- (optional) split info

---

## A10. Data QC (must do before training)
- Export QC:
  - sample random cells across FOVs
  - visualize `x256` vs `x128` for a few channels
  - overlay `cellmask256/cellmask128` to confirm centering + background
  - check canonical orientation consistency (as in Stage 1)
- Stats QC:
  - after applying `(x - mean[c]) / std[c]`, per-channel distributions look centered/scaled on a sample
- Reproducibility QC:
  - re-export a small subset and confirm identical tensors

---

## B. Dataset implementation (Stage 2; stacked `.pt` per cell)

### B1. New dataset module
- `solar/datasets/solar_stacked_dataset.py`
- `SolarStackedDatasetStage2`

### B2. What it loads
- `stack128_path -> x128_raw (C,128,128)`
- `channel_stats -> mean/std (C)`

### B3. Output dict from `__getitem__`
- `"low_res"`: `(C,128,128)` normalized:
  - `x[c] = (x_raw[c] - mean[c]) / std[c]`
- `"masks"`: optional `(2,128,128)` for debugging/conditioning, but not used for recon loss masking
- `"sample_id"`, `"cell_id"`
- `"cond_cell" (D_cell,)`, `"cond_sample" (D_sample,)`
- `"mu_shape" (D_shape,)`, `"logvar_shape" (D_shape,)`

---

## C. Stage 2 model development (SolarMapVAE)

### C1. Inputs / outputs
Inputs:
- `x: (B,C,128,128)`
- conditioning:
  - `z_shape` sampled each step from `mu_shape/logvar_shape`
  - `cond_cell`, `cond_sample`

Outputs:
- `x_hat: (B,C,128,128)`
- `z_coarse: (B,C,d_coarse)`
- `mu/logvar` for `z_coarse`

### C2. Losses
- Reconstruction loss: **unmasked** (model learns to place background pixels).
- KL loss on `z_coarse` (warmup/beta schedule optional).

---

## D. Training

### D1. Training script
- `solar/train/train_solar_map_vae.py`
- Loads manifest + channel_stats
- Splits by FOV (`sample_id`) for validation
- TensorBoard logging: recon grids + scalars
- Saves checkpoints every N epochs

### D2. Validation
- Qualitative:
  - recon per channel, especially background placement and perinuclear/peripheral patterns
- Quantitative:
  - recon loss curves, KL curves, stability on held-out FOVs
