# Pre-processing for SOLAR (MIBI organelle positivity maps)

## Goal
Generate robust **pixel-wise probability maps** \(P_m(x)\in[0,1]\) for each marker \(m\) from multiplexed MIBI images, accounting for inter-FOV variability, organelle spatial scale (~500 nm), and background/haze, while avoiding dotty “enhanced background” artifacts.

These probability maps are used as inputs/targets for SOLAR’s three-stage decomposition: shape → map → texture.

## Scope
- Operates on **whole FOV images** (no per-cell masks) for now.
- Typical image size ~2046×2046.
- Typical pixel size ~90 nm (organelle diameter ≥500 nm ⇒ clustered signal ~5–6 px).

## Outputs
For each marker \(m\), the `positivity_map_cli` writes:
- `P_m`: float32 probability map (same H×W as input) to `output_root/positivity_map/<fov>/<marker>.tiff`.

Optional debug outputs (when CLI `--debug` is enabled):
- Threshold diagnostics: `t_minotri`, `t_triangle`, `t_otsu`, `otsu_score`.
- Masking diagnostics: `t_low`, `mask_fraction`.
- Z-score diagnostics: `z_sigma_floor_fraction` (fraction of pixels where local std was floored, per window size).
- Background normalization diagnostics (if enabled): `z_bg`, `z_den`, `t_use`.
- Tile gating diagnostics: `tile_evidence`.

## CLI (source of truth)
Use:
- `python -m solar.cli.positivity_map_cli <fov_root=image_data> <output_root> <markers...>`

Key options (current implementation):
- `--asinh_cofactor` (float): asinh cofactor.
- `--despeckle_median_size` (int): early median filter size (0 disables).
- `--gaussian_sigmas_px` (floats): Gaussian sigmas used to build structure-supported intensity.
- `--z_windows_px` (ints): window sizes used for local z-scoring.
- `--z_sigma_floor` (float): floor local std to prevent z-score blow-ups (0 disables).
- `--normalize_z_to_bg` (flag): normalize Z to a per-FOV background→threshold scale before sigmoid mapping.
- `--low_percentile` (float): percentile to mask low-intensity pixels for stable thresholding.
- `--sigmoid_slope` (float): slope in probability mapping.
- `--tile_free` (flag): disable tile gating; otherwise uses `--tile_size` / `--tile_overlap`.
- `--min_component_area_px` (int): minimum connected-component size applied on `P > 0.9` (use cautiously; see notes below).

## Pipeline overview (per marker channel)

### Step 1 — Asinh transform
Transform raw marker intensity \(I\):
- \(I_{asinh} = \mathrm{asinh}(I / c)\)
where \(c\) is `asinh_cofactor`.

### Step 1b — Early despeckle (median filter) **(new)**
To suppress salt-and-pepper / hot-pixel-like speckles before they are amplified by local z-scoring:
- If `despeckle_median_size > 0`, apply `median_filter(I_asinh, size=despeckle_median_size)`.

### Step 2 — Structure-supported intensity map `S`
Compute a multi-scale smoothed map and take the max over sigmas:
- `S = max_sigma gaussian_filter(I_asinh, sigma)`

Rationale: true organelle signal is spatially clustered at ~500 nm; smoothing reduces isolated pixel noise.

### Step 3 — Masking for stable thresholding
Exclude empty/near-empty background so thresholds are not dominated by zeros:
1) `v = S[S > 0]`.
2) If `v.size < 10000`, treat as near-zero signal and return an all-zero probability map.
3) `t_low = percentile(v, low_percentile)`.
4) `M = (S >= t_low)`.

### Step 4 — Spatial z-scoring (local contrast)
Compute local z-scores over each window `w` in `z_windows_px`:
- `Z_w = (S - mu_w) / (sigma_w + eps)`.

Then combine conservatively:
- `Z = min_w Z_w`.

### Step 4b — Z-score sigma floor **(new)**
Local variance collapse can create artificially huge z-scores from tiny speckles.
If `z_sigma_floor > 0`:
- floor `sigma_w = max(sigma_w, z_sigma_floor)` before computing `Z_w`.

### Step 5 — Adaptive cutoff with Minotri
Compute Minotri on masked z-scores:
- `values = Z[M]`
- `t_minotri = min(t_triangle, t_otsu)` (computed by `minotri_threshold`).

### Step 6 — Probability mapping (sigmoid)
Default mapping (legacy behavior):
- `Z_norm = Z`
- `t_use = t_minotri`
- `P = 1 / (1 + exp(-(Z_norm - t_use) / sigmoid_slope))`.

### Step 6b — Background-normalized mapping **(new, optional)**
To align background probability levels across FOVs (A/B comparable because it’s a flag), enable `--normalize_z_to_bg`.

Implementation:
- `z_bg = median(Z[M])`
- `den = max(t_minotri - z_bg, 1e-5)`
- `Z_norm = (Z - z_bg) / den`
- `t_use = 1.0`
- Then apply the same sigmoid mapping using `Z_norm` and `t_use`.

Interpretation: this normalizes the per-FOV background-to-threshold distance, so the meaning of probability is more consistent across FOVs.

### Step 7 — Tile gating (optional)
If tile gating is enabled (i.e. not `--tile_free`), split into overlapping tiles and compute an evidence score:
- `evidence = percentile(Z_tile, 99.9) - median(Z_tile)`.

If evidence is below a fixed threshold (currently 0.1), suppress probabilities in that tile.

### Step 8 — Minimum size enforcement (use cautiously)
Connected-component filtering is applied on a hard threshold `B = (P > 0.9)` and removes components smaller than `min_component_area_px`.

Important caveat:
- Because it measures only the *saturated core* (`P>0.9`), large biological structures can fragment into many small high-P islands and be unintentionally removed when increasing `min_component_area_px`.
- Prefer addressing speckles earlier (Steps 1b and 4b) and use Step 8 only as a weak prior.

## Practical guidance / troubleshooting
- Too many tiny enhanced dots: increase `despeckle_median_size` and/or enable `z_sigma_floor` (these reduce amplification by z-scoring).
- Background probability differs across FOVs: enable `normalize_z_to_bg` to improve cross-FOV comparability.
- Nuclear markers (e.g., HH3) are not “puncta-like”: the local-contrast z-score tends to highlight heterochromatin peaks rather than full nuclei; use this as a known control limitation, or add a separate mode for broad structures.

## Next steps in SOLAR
- Use `P_m` maps as training targets / pseudo-labels for Stage 2 (SolarMapVAE).
- Use per-FOV debug stats (`t_minotri`, `z_bg`, `z_den`, `z_sigma_floor_fraction`) to detect dataset drift and parameter sensitivity.
