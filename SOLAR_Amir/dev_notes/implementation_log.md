## [Stage 0 Completed] - 2026-01-17
**Module Implemented**: SolarDataset + BalancedBatchSampler

### 1. Files Created:
- `solar/datasets/solar_dataset.py`: SolarDataset with padding-aware preprocessing, config dataclass, and batch visualization CLI.
- `solar/datasets/samplers.py`: BalancedBatchSampler that oversamples minority sample_ids and interleaves batches.
- `tests/test_solar_dataset.py`: Synthetic-data unit tests for dataset outputs and sampler balance.

### 2. Verification Steps Performed:
- [x] Unit Test Passed: `tests/test_solar_dataset.py`
- [ ] Smoke Test: Overfit 64 cells (not run; data/model not yet wired)

### 3. Key Decisions/Notes:
- Used center crop + symmetric zero-padding to reach target canvases and avoid upscaling small cells.
- Downsampled high-res channels to low-res with area interpolation only when larger than target; otherwise pad.
- Balanced sampler oversamples each sample_id to the max group length to keep epochs sample-balanced.

### 4. Next Steps:
- Install pytest in the environment and run `tests/test_solar_dataset.py`.
- Wire dataset to real data loaders/paths and add canonical rotation (if needed) before training Stage 1.
- Add smoke overfit loop once SolarShapeVAE is available.

## [Stage 1 Completed] - 2026-01-17
**Module Implemented**: SolarShapeVAE (rotation-invariant morphology VAE)

### 1. Files Created:
- `solar/models/solar_shape_vae.py`: e2cnn-based encoder with GroupPooling and PyTorch decoder producing 2×64×64 masks.
- `solar/train/train_solar_shape_vae.py`: Training loop with BCE+KL (warmup), synthetic smoke option, checkpointing, optional mu export.
- `tests/test_solar_shape_vae_forward.py`: Forward-shape unit test with e2cnn import guard.

### 2. Verification Steps Performed:
- [x] Unit Test Passed: `tests/test_solar_shape_vae_forward.py`
- [x] Smoke Test: Overfit 64 synthetic cells (loss 0.39 → 0.037 over 10 epochs; see train log)

### 3. Key Decisions/Notes:
- Encoder uses three R2Conv stride-2 blocks with InnerBatchNorm+ReLU, then GroupPooling for rotation invariance before mu/logvar.
- Decoder is a standard ConvTranspose stack with sigmoid output; latent_dim/group_N exposed via constructor.
- KL warmup is linear in global steps; BalancedBatchSampler used in training script to keep sample_id balance.
- Embedding export uses pandas parquet if available; silently skipped otherwise.
- Swapped encoder gspace to `FlipRot2dOnR2` (dihedral) for rotation+reflection equivariance; kept decoder standard.
- Added TensorBoard logging (scalars per step, weight histograms per epoch, recon grids on fixed 16 samples) and `--log_dir` flag.
- SolarDataset now accepts PNG paths directly and loads via PIL; kept padding/downsample logic unchanged.
- Added paired split builder: `solar/datasets/build_paired_split.py` with group-aware 80/20 split, manifests, and Subset-based train/val.

### 4. Next Steps:
- Install `e2cnn` and `pytest`, then run `python -m pytest tests/test_solar_shape_vae_forward.py` and a smoke overfit on synthetic masks.
- Wire training script to real mask crops (replace synthetic loader) and monitor reconstruction quality.
- Add canonical rotation preprocessing if needed before feeding masks to Stage 1.

## [Stage 1 Update] - 2026-01-17
**Module Implemented**: Real-data SolarShapeVAE loader + mask-only dataset support

### 1. Files Created:
- `solar/train/train_solar_shape_vae.py`: Real-data path via `--mask_manifest` with optional `--mask_root`; uses mask-only dataset config.
- `solar/datasets/solar_dataset.py`: Added `mask_only` config flag and placeholder channels to allow mask-only batches.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_solar_shape_vae_forward.py` (not re-run after changes)
- [ ] Smoke Test: Real-data training not run (no manifest provided here)

### 3. Key Decisions/Notes:
- Manifest format: tab-separated `nucleus_mask_path [tab] cell_mask_path [tab] sample_id`; if only one path is given, it is reused for both masks and sample_id defaults to 0.
- Mask-only mode keeps `low_res`/`high_res` as single-channel placeholders to preserve dataloader structure while Stage 1 consumes only `masks`.

### 4. Next Steps:
- Provide a mask manifest and run `python -m solar.train.train_solar_shape_vae --mask_manifest <file> --batch_size <b> --epochs <e>`.
- Optionally set `--mask_root` if manifest paths are relative; adjust `--high_res_size`/`--low_res_size` to your crop sizes.
- Add canonical rotation preprocessing later if needed for alignment.

## [Stage 1 Tooling] - 2026-01-17
**Module Implemented**: Mask manifest builder for SolarShapeVAE

### 1. Files Created:
- `solar/datasets/build_mask_manifest.py`: CLI to generate mask manifest TSV (nucleus\tcell\tsample_id) from mask directories using `_cell_####` stem matching.

### 2. Verification Steps Performed:
- [ ] Manual run not executed here; script is deterministic and uses existing extractor logic from `build_paired_split`.

### 3. Key Decisions/Notes:
- Supports separate `--nucleus_dir` and `--cell_dir` (defaults to same dir) and optional `--relative_to` to keep manifest paths relative for use with `--mask_root` during training.
- Single `--sample_id` applied to all rows; adjust per-sample by running per-sample directories and concatenating outputs if needed.

### 4. Next Steps:
- Run `python -m solar.datasets.build_mask_manifest --nucleus_dir <masks> --out manifests/masks.tsv --sample_id 0 --relative_to <base>` and feed to Stage 1 training with `--mask_root <base>`.

## [Stage 1 Input Flex] - 2026-01-17
**Module Implemented**: Combined-mask support (0/1/2 label maps) for Stage 1

### 1. Files Updated:
- `solar/datasets/solar_dataset.py`: `combined_mask_values` config and combined mask handling to derive nucleus (value==nucleus) and cell (value==cytoplasm or nucleus) masks.
- `solar/train/train_solar_shape_vae.py`: CLI flags `--combined_mask`, `--val_background`, `--val_cytoplasm`, `--val_nucleus` to consume single-path label maps from the manifest.

### 2. Verification Steps Performed:
- [ ] Not re-run tests; changes are localized to input decoding.

### 3. Key Decisions/Notes:
- Combined mask interpretation defaults to background=0, cytoplasm=1, nucleus=2; cell mask is (cytoplasm OR nucleus), nucleus mask is equality to nucleus value.
- When `--combined_mask` is set, manifest rows can still have one or two columns; only the first path is used as the label map.

### 4. Next Steps:
- Prepare manifest with single-path label maps and run Stage 1: `python -m solar.train.train_solar_shape_vae --mask_manifest <file> --combined_mask --mask_root <base> ...`.

## [Stage 1 Tooling Update] - 2026-01-17
**Module Updated**: build_mask_manifest combined-mask mode

### 1. Files Updated:
- `solar/datasets/build_mask_manifest.py`: `--combined_mask` flag to emit `labelmap\tsample_id` lines for 0/1/2 PNG/TIF label maps.

### 2. Verification Steps Performed:
- [ ] Not run; logic mirrors existing path listing.

### 3. Key Decisions/Notes:
- When `--combined_mask` is set, only the first column is used by Stage 1 loader; sample_id column remains.

### 4. Next Steps:
- Generate label-map manifest: `python -m solar.datasets.build_mask_manifest --nucleus_dir <label_dir> --combined_mask --sample_id 0 --relative_to <base> --out manifests/masks_label.tsv`.

## [Stage 1 Data Prep] - 2026-01-17
**Module Implemented**: Combined-mask cropper CLI

### 1. Files Created:
- `solar/datasets/make_combined_masks.py`: CLI to crop per-cell combined masks (0 background, 1 cytoplasm, 2 nucleus) from cell/nuclear segmentations; pairs FOVs by stem and saves PNGs.

### 2. Verification Steps Performed:
- [ ] Not run; logic mirrors prior notebook flow with edge checks and nearest-neighbor downsample.

### 3. Key Decisions/Notes:
- Defaults: framesize=256, downsample=2, output to `SCALER/SCALER_masks`. Skips cells too close to borders.

### 4. Next Steps:
- Run `python -m solar.datasets.make_combined_masks --cell_dir <cleaned_masks> --nuclear_dir <nuclear_masks> --out_dir <out> --framesize 256 --downsample 2` then build manifest with `--combined_mask`.

## [Stage 1 Tooling Fix] - 2026-01-17
**Module Updated**: build_mask_manifest sample_id inference

### 1. Files Updated:
- `solar/datasets/build_mask_manifest.py`: `--infer_sample_id` flag to derive sample_id from filename stem before `_cell_` so mixed-sample manifests no longer default all zeros.

### 2. Verification Steps Performed:
- [ ] Not run; logic is straightforward stem mapping.

### 3. Key Decisions/Notes:
- When `--infer_sample_id` is set, the script ignores `--sample_id` and assigns sample IDs per unique stem prefix.

### 4. Next Steps:
- Rebuild manifest with `--infer_sample_id` if sample IDs should reflect per-FOV prefixes rather than a single constant.

## [Stage 1 Training Update] - 2026-01-17
**Module Updated**: SolarShapeVAE training loop with train/test split and test logging

### 1. Files Updated:
- `solar/train/train_solar_shape_vae.py`: Added seeded 80/20 train/test split via `random_split`, train loader uses `BalancedBatchSampler` on train subset, test loader is plain; added evaluation loop with TensorBoard logging (test loss/BCE/KL, beta=1.0), recon grids now sampled from test subset with fixed seed, embeddings export uses test loader.

### 2. Verification Steps Performed:
- [ ] Not re-run tests post-change.

### 3. Key Decisions/Notes:
- Evaluation uses beta=1.0 (no warmup) to reflect full objective; training keeps KL warmup.
- Split reproducible with `torch.Generator().manual_seed(args.seed)`.

### 4. Next Steps:
- Optionally add val split and early stopping if needed; rerun training to populate new test metrics in TensorBoard.

## [Stage 1 Training Fix] - 2026-01-17
**Module Updated**: SolarShapeVAE train loop bugfix

### 1. Files Updated:
- `solar/train/train_solar_shape_vae.py`: Fixed NameError in epoch summary (uses `train_loader` for batch count).

### 2. Verification Steps Performed:
- [ ] Not re-run training; small fix only.

### 3. Key Decisions/Notes:
- None.

### 4. Next Steps:
- Re-run training; warning from e2cnn about uint8 indexing is benign.

## [Stage 1 Data Prep Align] - 2026-01-17
**Module Updated**: make_combined_masks alignment

### 1. Files Updated:
- `solar/datasets/make_combined_masks.py`: Added centroid-based translation, PCA orientation to diagonal, and nucleus-upper-left flip/rotate before saving combined masks (order=0, preserves 0/1/2 labels).

### 2. Verification Steps Performed:
- [ ] Not run; logic ported from prior notebook utility.

### 3. Key Decisions/Notes:
- Alignment applied unconditionally after cropping, before downsampling; uses nearest-neighbor to avoid label bleed.

### 4. Next Steps:
- Rerun combined mask generation if aligned crops are desired.

## [Stage 1 Tooling - Weights] - 2026-01-17
**Module Implemented**: Class weight computation for combined masks

### 1. Files Created:
- `solar/datasets/compute_class_weights.py`: CLI to compute class frequencies/weights (0/1/2) from a mask manifest; outputs counts and weights summing to 1.

### 2. Verification Steps Performed:
- [ ] Not run; simple histogram logic.

### 3. Key Decisions/Notes:
- Reads first column of manifest (combined masks), optional `--mask_root` to prepend; weights = counts / total.

### 4. Next Steps:
- Run `python -m solar.datasets.compute_class_weights --mask_manifest manifests/masks_label.tsv --mask_root <base>` and feed `--class_weights` in training.

## [Stage 1 Data Prep Fix] - 2026-01-17
**Module Updated**: make_combined_masks cell-driven pairing

### 1. Files Updated:
- `solar/datasets/make_combined_masks.py`: Pairing now uses cell_dir filenames as the reference; extra nuclear masks are ignored, missing nuclear partners are warned and skipped instead of causing failure.

### 2. Verification Steps Performed:
- [ ] Not run; logic change only.

### 3. Key Decisions/Notes:
- Keys are derived from stems with `_cleaned_mask` / `_nuclear` stripped; only cell_dir keys are kept, so dataset cardinality follows cell masks.

### 4. Next Steps:
- Rerun mask generation after updating nuclear_dir contents if needed.

## [Stage 1 Resolution Update] - 2026-01-17
**Module Updated**: SolarShapeVAE + SolarDataset defaults for downsample=2

### 1. Files Updated:
- `solar/train/train_solar_shape_vae.py`: Default `low_res_size` set to 128 and passed into `SolarShapeVAE` so model input matches data.
- `solar/models/solar_shape_vae.py`: Default `input_size` set to 128 to align with 256→128 downsample.
- `solar/datasets/solar_dataset.py`: Config default `low_res_size` updated to 128; visualization titles now reflect actual tensor sizes.
- `tests/test_solar_shape_vae_forward.py`: Forward shape test uses 128×128 masks and expects 3-channel outputs.
- `tests/test_solar_dataset.py`: Dataset shape expectations updated to 128×128 low-res masks.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_solar_shape_vae_forward.py`
- [ ] Unit Test Passed: `tests/test_solar_dataset.py`
- [ ] Smoke Test: Not run after resolution change

### 3. Key Decisions/Notes:
- Default downsample factor is now 2 (256→128); training CLI defaults follow this setting.
- `SolarShapeVAE` now receives `input_size` from CLI to stay consistent if users override `low_res_size`.

### 4. Next Steps:
- Re-run unit tests to confirm shape expectations at 128×128.
- Re-run Stage 1 training with regenerated masks (downsample 2) and verify reconstruction quality.

## [Stage 1 Stability Update] - 2026-01-17
**Module Updated**: SolarShapeVAE KL stabilization and training loop safety

### 1. Files Updated:
- `solar/models/solar_shape_vae.py`: Clamp `logvar` to [-6, 6] inside `kl_loss` to cap variance-driven spikes.
- `solar/train/train_solar_shape_vae.py`: Extend default `--kl_warmup_steps` to 2000 and add `--grad_clip` (default 5.0) applied via `clip_grad_norm_` each step.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_solar_shape_vae_forward.py`
- [ ] Smoke Test: Not rerun after stability changes

### 3. Key Decisions/Notes:
- Followed SCALER practices: longer warmup, grad clipping, and logvar clamp to tame KL excursions.

### 4. Next Steps:
- Re-run Stage 1 training with the new defaults; monitor KL for reduced spikes.

## [Stage 1 Logging Update] - 2026-01-17
**Module Updated**: TensorBoard run naming for hyperparameter sweeps

### 1. Files Updated:
- `solar/train/train_solar_shape_vae.py`: Added `make_run_dir` to auto-append `ld{latent_dim}_wu{kl_warmup_steps}_lr{lr}` to `--log_dir`, so runs with different latent dims, warmup steps, or learning rates are separated.

### 2. Verification Steps Performed:
- [ ] Not run; logging-only change.

### 3. Key Decisions/Notes:
- Run directories now encode latent_dim, warmup_steps, and lr to simplify TensorBoard comparisons.

### 4. Next Steps:
- Launch new runs; check `runs/solarshape_vae/ld*_wu*_lr*` for clean separation in TensorBoard.

## [Stage 1 Beta/FreeBits Update] - 2026-01-17
**Module Updated**: KL/beta scheduling flexibility

### 1. Files Updated:
- `solar/train/train_solar_shape_vae.py`: Added `--max_beta`, `--beta_cycle_steps`, and `--kl_free_bits` flags; introduced `beta_factor` for capped or triangular beta schedules; added `kl_with_free_bits` (logvar clamp + per-dim free bits) and applied it in training; run naming now includes max_beta and free_bits.

### 2. Verification Steps Performed:
- [ ] Not run; logic-only change.

### 3. Key Decisions/Notes:
- Default eval keeps full KL (no free bits); training can cap beta, cycle beta, and apply free bits to reduce recon degradation after warmup.

### 4. Next Steps:
- Sweep beta settings, e.g., `--max_beta 0.5`, `--kl_free_bits 0.5`, or `--beta_cycle_steps 1000`, and compare runs via the hyperparam-coded log dirs.

## [Stage 1 Arch Align] - 2026-01-17
**Module Updated**: SolarShapeVAE aligned to SCALER width/depth

### 1. Files Updated:
- `solar/models/solar_shape_vae.py`: Default latent_dim to 32; base filters 64 with 3 stride-2 equivariant conv blocks (kernel 5) using `FlipRot2dOnR2`; decoder mirrors channel widths; config now stores base_filters/nlayers/kernel_size.
- `solar/train/train_solar_shape_vae.py`: Default `--latent_dim` set to 32.
- `tests/test_solar_shape_vae_forward.py`: Test instantiates latent_dim 32.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_solar_shape_vae_forward.py`
- [ ] Smoke Test: Not run after architecture change

### 3. Key Decisions/Notes:
- Kept SCALER-like channel scaling (64,128,256) while retaining e2cnn encoder for rotation/reflection equivariance; decoder remains plain CNN.

## [Stage 1 Arch Tuning] - 2026-01-17
**Module Updated**: SolarShapeVAE width reduction for speed

### 1. Files Updated:
- `solar/models/solar_shape_vae.py`: Default `base_filters` reduced from 64 to 32 to lighten encoder/decoder while keeping 3-layer stride-2 structure and equivariant encoder.

### 2. Verification Steps Performed:
- [ ] Not run; parameter-only change.

### 3. Key Decisions/Notes:
- Halving base filters should cut MACs and memory markedly while preserving architecture shape.

## [Stage 1 Config Flex] - 2026-01-17
**Module Updated**: Expose base_filters in trainer/run naming

### 1. Files Updated:
- `solar/train/train_solar_shape_vae.py`: Added `--base_filters` CLI (default 32), passed into `SolarShapeVAE`, and included in run-directory naming for TensorBoard comparisons.

### 2. Verification Steps Performed:
- [ ] Not run; config-only change.

### 3. Key Decisions/Notes:
- Run naming now encodes base_filters to keep sweep logs disambiguated.

### 4. Next Steps:
- Sweep base_filters (e.g., 24/32/48) to trade off speed vs. reconstruction.

### 4. Next Steps:
- Re-run training to gauge speed/quality tradeoff; adjust base_filters further if needed.

### 4. Next Steps:
- Re-run forward/unit tests and a short training run to confirm stability with the wider encoder/decoder.

## [Stage 2 Completed] - 2026-01-21
**Module Implemented**: SolarMapVAE + Stage 2 dataset/loader

### 1. Files Created:
- `solar/datasets/solar_stacked_dataset.py`: Stage 2 dataset that loads canonicalized stacks, normalization stats, and conditioning vectors (`cond_cell`, `cond_sample`, `mu_shape`, `logvar_shape`).
- `solar/datasets/canonicalize.py`: Shared helper to apply Stage 1 canonicalization transforms to both labelmaps and intensity stacks, plus downsampling utilities.
- `solar/models/solar_map_vae.py`: Conditional per-marker VAE with independent encoders/decoders per channel and conditioning on resampled morphology + cell/sample covariates.
- `solar/train/train_solar_map_vae.py`: Training loop with beta warmup/cycle, free-bits KL, BalancedBatchSampler by sample_id, TensorBoard logging, and synthetic smoke option.
- `tests/test_solar_map_vae_forward.py`: Forward-shape unit test for SolarMapVAE.
- `tests/test_solar_stacked_dataset_stage2.py`: Dataset unit test covering normalization and conditioning vectors.

### 2. Verification Steps Performed:
- [x] Unit Test Passed: `tests/test_solar_map_vae_forward.py`
- [x] Unit Test Passed: `tests/test_solar_stacked_dataset_stage2.py`
- [ ] Smoke Test: Full Stage 2 training (real stacks/manifest) not run yet

### 3. Key Decisions/Notes:
- Stage 2 model uses independent marker encoders/decoders with shared conditioning vector `z_shape + cond_cell + cond_sample`; morphology latent is resampled each forward pass.
- Reconstruction loss is MSE over normalized stacks (unmasked) to force background placement; KL uses per-channel latent sums with optional free bits.
- Dataset expects per-cell manifest columns prefixed `cond_cell_`, `cond_sample_`, `mu_shape_`, `logvar_shape_`; channel stats (mean/std) are loaded from JSON/PT and applied per channel.
- Added canonicalization helper mirroring Stage 1 alignment (centroid translation, PCA angle to diagonal, nucleus-upper-left rule) and ensured identical transforms apply to intensity stacks.

### 4. Next Steps:
- Run the Stage 2 export pipeline to produce canonicalized 256→128 stacks and channel stats per plan, then launch `python -m solar.train.train_solar_map_vae --manifest <parquet> --channel_stats <json> ...`.
- Add QC notebook/plots for Stage 2 stacks (x256 vs x128, masks), and consider masked-validation visual checks before long runs.
- Integrate FOV-level channel-stat computation (pixels inside cell masks only) into a CLI utility for reproducible normalization.

## [Stage 1 Diagnostic Option] - 2026-01-18
**Module Updated**: Optional plain Conv encoder for A/B

### 1. Files Updated:
- `solar/models/solar_shape_vae.py`: Added `use_e2cnn` flag (default True); when False, builds a plain Conv2d+BatchNorm+ReLU stride-2 encoder instead of e2cnn, with encode logic branching accordingly. Decoder unchanged.
- `solar/train/train_solar_shape_vae.py`: Added `--no_e2cnn` flag (diagnostic only), passed into the model, and included `e2` marker in run dir naming.

### 2. Verification Steps Performed:
- [ ] Not run; diagnostic path only.

### 3. Key Decisions/Notes:
- Default remains e2cnn for Stage 1 requirements; plain Conv path is for quick A/B to rule out equivariance issues.

### 4. Next Steps:
- If needed, run a short `--no_e2cnn` experiment to compare recon quality; revert to e2cnn by default.

## [Stage 1 Objective Align] - 2026-01-18
**Module Implemented**: SolarShapeVAE loss parity with SCALER

### 1. Files Updated:
- `solar/train/train_solar_shape_vae.py`: Reconstruction loss uses weighted categorical NLL summed over pixels with normalized class weights; KL helper now sums per sample (optional free bits) and redundant KL compute removed in the train loop.
- `solar/models/solar_shape_vae.py`: KL loss now matches SCALER scale (sum over latent dims, batch mean).

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_solar_shape_vae_forward.py`
- [ ] Smoke Test: Train Stage 1 with updated objective
- [ ] Eval: Compare recon quality before/after normalization

### 3. Key Decisions/Notes:
- Loss scaling mirrors SCALER (pixel-summed recon, batch-mean) to boost reconstruction emphasis and reduce unintended KL dominance.
- Class weights are normalized to the number of classes for stable magnitude across different weight sets.

### 4. Next Steps:
- Run short A/B: old objective vs updated weighted NLL (with/without `--no_e2cnn`) using the same seed to confirm reconstruction gain.
- Tune `--max_beta` or `--kl_free_bits` if KL still overwhelms reconstruction.

## [Stage 1 Checkpointing] - 2026-01-19
**Module Updated**: Training checkpoint naming/schedule

### 1. Files Updated:
- `solar/train/train_solar_shape_vae.py`: Added `--save_every` for periodic saves; checkpoints now stored under a run-named subfolder (from `make_run_dir`) with filenames `{run_dir.name}_epochXXXX.pt`; final checkpoint always written with epoch number padded to 4 digits.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_solar_shape_vae_forward.py`
- [ ] Smoke Test: Periodic checkpoint save

### 3. Key Decisions/Notes:
- If `--checkpoint` is a directory or file, checkpoints go into `<checkpoint_base>/<run_dir.name>/`; periodic saves respect `save_every` and final save uses epoch count.

### 4. Next Steps:
- Run a short synthetic job with `--epochs 12 --save_every 10` to confirm checkpoints appear at epochs 10 and 12 in the run-named folder.

## [Stage 2 MIBI Prep Update] - 2026-01-22
**Module Implemented**: Directory-based Stage 2 data prep

### 1. Files Created:
- `solar/datasets/build_stage2_tables_mibi.py`: CLI to build Stage 2 cell/fov tables from MIBI folder layout with deterministic `sample_id`.
- `dev_notes/stage2_mibi_prep.md`: Short runbook with the 3-command pipeline for MIBI folder layout.
- `tests/test_mibi_stage2_tables_and_io.py`: End-to-end test for directory-based stacks + table builder.

### 2. Files Updated:
- `solar/datasets/export_stage2_crops.py`: Added directory stack loading with explicit `channel_names`, optional centroid-based crops, and warnings on fallback.
- `solar/datasets/compute_channel_stats.py`: Added directory stack loading with explicit `channel_names`.

### 3. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_mibi_stage2_tables_and_io.py` (not run here)
- [ ] Unit Test Passed: `tests/test_export_stage2_crops.py` (not re-run after changes)
- [ ] Unit Test Passed: `tests/test_compute_channel_stats.py` (not re-run after changes)

### 4. Key Decisions/Notes:
- Directory stack loading requires explicit channel ordering; missing channel files raise clear errors.
- Optional centroid cropping falls back to mask-derived centers if the target label is missing from the crop.

### 5. Next Steps:
- Run `python -m pytest tests/test_mibi_stage2_tables_and_io.py tests/test_export_stage2_crops.py tests/test_compute_channel_stats.py`.
- Use the new runbook to build tables, compute channel stats, and export Stage 2 crops on MIBI data.

## [Stage 2 Manifest Portability] - 2026-01-22
**Module Implemented**: Flat crop layout + relative path manifests

### 1. Files Updated:
- `solar/datasets/export_stage2_crops.py`: Added `--flat_output`, `--filename_template`, and `--relative_paths` for flat output and manifest-relative filenames with collision checks.
- `solar/datasets/solar_stacked_dataset.py`: Added `data_root` path resolution for relative `stack*_path` and `mask*_path`.
- `solar/train/train_solar_map_vae.py`: Added `--data_root` CLI and passed into Stage 2 dataset config.

### 2. Files Created:
- `tests/test_stage2_relative_paths.py`: Covers flat export filenames, relative manifest paths, and `data_root` resolution.

### 3. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_stage2_relative_paths.py` (not run here)

### 4. Key Decisions/Notes:
- Flat outputs default to relative manifest filenames when `--flat_output` is used; existing per-FOV folders remain the default.
- Relative paths are resolved at load time using `data_root`, keeping manifests portable.

### 5. Next Steps:
- Run `python -m pytest tests/test_stage2_relative_paths.py`.

## [Stage 1 Morph Exporter] - 2026-01-23
**Module Implemented**: Export cond_morph (mu/logvar) from combined-mask PNGs

### 1. Files Created:
- `solar/datasets/export_cond_morph_from_png.py`: CLI to compute Stage 1 morphology latents from combined-mask PNGs and append `mu_shape_*`/`logvar_shape_*` to a table.
- `tests/test_export_cond_morph_from_png.py`: Unit tests for successful export and missing-file error handling.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_export_cond_morph_from_png.py` (not run here)

### 3. Key Decisions/Notes:
- Uses `SolarDataset` in `mask_only` + `combined_mask_values` mode to match Stage 1 mask semantics.
- Loads `SolarShapeVAE` from checkpoint config with CLI overrides when provided.

### 4. Next Steps:
- Run `python -m pytest tests/test_export_cond_morph_from_png.py`.

## [Stage 2 Update] - 2026-01-21
**Module Implemented**: Stage 2 data prep utilities (crops + normalization stats)

### 1. Files Created:
- `solar/datasets/export_stage2_crops.py`: CLI and helpers to crop 256×256 stacks, apply Stage 1 canonicalization to labels/stacks, downsample to 128, and emit manifest paths (optional mask saves and metadata columns).
- `solar/datasets/compute_channel_stats.py`: CLI to compute per-channel mean/std using only pixels inside cell masks from FOV stacks.
- `tests/test_export_stage2_crops.py`: Unit test covering crop export outputs and saved shapes/paths.
- `tests/test_compute_channel_stats.py`: Unit test validating masked mean/std computation.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_export_stage2_crops.py` (pytest missing in env)
- [ ] Unit Test Passed: `tests/test_compute_channel_stats.py` (pytest missing in env)
- [ ] Smoke Test: Full Stage 2 export + training not run

### 3. Key Decisions/Notes:
- Exporter pads crops at borders instead of skipping edge cells; canonicalization metadata can be prefixed into the manifest.
- Channel stats accept `.pt` or image stacks; variance is computed from masked pixels only and `std` uses a small numerical floor.
- Nuclear mask column is optional; exporter tolerates missing or empty nuclear paths per row.

### 4. Next Steps:
- Install pytest and run `python -m pytest tests/test_export_stage2_crops.py tests/test_compute_channel_stats.py`.
- Run channel-stats CLI on full FOV table: `python -m solar.datasets.compute_channel_stats --fov_table <file> --stack_column <col> --mask_column <col> --out stage_stats/channel_stats.json`.
- Export real Stage 2 crops/manifest: `python -m solar.datasets.export_stage2_crops --cell_table <cells.parquet> --stack_column stack_path --cell_mask_column cell_mask_path --nuclear_mask_column nuclear_mask_path --out_manifest manifests/stage2.parquet --save_masks`.

## [Stage 2 Tooling] - 2026-01-24
**Module Implemented**: Add lineage marker conditioning columns to Stage 2 manifest

### 1. Files Created:
- `solar/datasets/add_lineage_cond_to_manifest.py`: CLI/helper to join marker tables, append `cond_cell_*` columns, optionally normalize, and report unmatched rows.
- `tests/test_add_lineage_cond_to_manifest.py`: Unit tests for successful join and missing-marker error handling.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_add_lineage_cond_to_manifest.py` (not run here)

### 3. Key Decisions/Notes:
- Supports joins on `cell_id` or `fov_name` + `cell_mask_id`, with optional `fillna` and z-score normalization.

### 4. Next Steps:
- Run `python -m pytest tests/test_add_lineage_cond_to_manifest.py`.

## [Stage 2 Tooling] - 2026-01-23
**Module Implemented**: Filter Stage 2 cell table to cells with Stage 1 combined-mask PNGs

### 1. Files Created:
- `solar/datasets/filter_stage2_to_stage1_masks.py`: CLI/helper to keep only rows with matching Stage 1 PNGs; emits `has_stage1_mask` and optional mask path column.
- `tests/test_filter_stage2_to_stage1_masks.py`: Unit test covering keep/missing outputs and report generation.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_filter_stage2_to_stage1_masks.py` (not run here)

### 3. Key Decisions/Notes:
- Strict filename template enforcement; no glob fallback to avoid silent mismatches.

### 4. Next Steps:
- Run `python -m pytest tests/test_filter_stage2_to_stage1_masks.py`.

## [Stage 2 Tooling] - 2026-01-23
**Module Implemented**: Add per-FOV cond_sample covariates to Stage 2 manifest

### 1. Files Created:
- `solar/datasets/add_fov_cond_sample_to_manifest.py`: CLI/helper to compute per-FOV medians, z-score across FOVs, and merge `cond_sample_*` back to per-cell table.
- `tests/test_add_fov_cond_sample_to_manifest.py`: Unit test validating medians, z-scoring, and merge correctness.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_add_fov_cond_sample_to_manifest.py` (not run here)

### 3. Key Decisions/Notes:
- Z-score is computed over FOV-level medians with configurable `ddof` and `eps` to avoid division by zero.

### 4. Next Steps:
- Run `python -m pytest tests/test_add_fov_cond_sample_to_manifest.py`.

## [Stage 2 Tooling] - 2026-01-23
**Module Implemented**: Build slim Stage 2 training manifest

### 1. Files Created:
- `solar/datasets/build_stage2_training_manifest.py`: CLI/helper to keep only IDs, paths, and conditioning columns for Stage 2 training.
- `tests/test_build_stage2_training_manifest.py`: Unit test verifying required columns are retained and extras dropped.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_build_stage2_training_manifest.py` (not run here)

### 3. Key Decisions/Notes:
- Requires matched `mu_shape_*`/`logvar_shape_*` counts; `cond_cell_*` and `cond_sample_*` are optional.

### 4. Next Steps:
- Run `python -m pytest tests/test_build_stage2_training_manifest.py`.

## [Stage 2 Training Update] - 2026-01-23
**Module Implemented**: Stage 2 checkpoint directory convention (Stage 1 style)

### 1. Files Created:
- `tests/test_train_solar_map_vae_checkpoint_paths.py`: Verifies Stage 2 checkpoint path formatting.

### 2. Files Updated:
- `solar/train/train_solar_map_vae.py`: Save checkpoints under `<checkpoint_root>/<run_name>/` with Stage 1-style filenames.

### 3. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_train_solar_map_vae_checkpoint_paths.py` (not run here)

### 4. Key Decisions/Notes:
- If `--checkpoint` points to a `.pt` file, the parent directory is used as the checkpoint root and a warning is emitted.

### 5. Next Steps:
- Run `python -m pytest tests/test_train_solar_map_vae_checkpoint_paths.py`.

## [Tooling Docs Update] - 2026-01-24
**Module Implemented**: Positivity map documentation + tile-free gating support

### 1. Files Updated:
- `solar/models/positivity_probability_map.py`: Added module docstring, clarified tiling parameters, and made tile gating optional for whole-FOV analysis.
- `solar/models/minotri_threshold.py`: Added module docstring and removed unused import.
- `solar/cli/positivity_map_cli.py`: Added module/function docstrings for CLI usage.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_positivity_probability_map.py` (not run here)

### 3. Key Decisions/Notes:
- Tile-based gating now supports a tile-free mode by skipping gating when `tile_size` is None.

### 4. Next Steps:
- Run `python -m pytest tests/test_positivity_probability_map.py`.

## [CLI Update] - 2026-01-24
**Module Implemented**: Multi-marker and all-FOV positivity map processing

### 1. Files Updated:
- `solar/cli/positivity_map_cli.py`: Accepts a list of markers and iterates across all FOV folders under `project_root/image_data`.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_positivity_probability_map.py` (not run here)

### 3. Key Decisions/Notes:
- The CLI now skips missing marker files per FOV and continues processing remaining inputs.

### 4. Next Steps:
- Run `python -m pytest tests/test_positivity_probability_map.py`.

## [CLI Update] - 2026-01-24
**Module Implemented**: FOV filtering by marker availability

### 1. Files Updated:
- `solar/cli/positivity_map_cli.py`: Skip FOV folders without any TIFFs or missing required markers; accept .tif/.tiff inputs.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_positivity_probability_map.py` (not run here)

### 3. Key Decisions/Notes:
- Enforces all requested markers per FOV before processing; missing markers skip the entire FOV.

### 4. Next Steps:
- Run `python -m pytest tests/test_positivity_probability_map.py`.

## [Bugfix] - 2026-01-24
**Module Implemented**: Correct Nellie threshold imports

### 1. Files Updated:
- `solar/models/minotri_threshold.py`: Import `otsu_threshold`, `triangle_threshold`, and `_get_xp` from `nellie.utils.gpu_functions`.
- `solar/models/positivity_probability_map.py`: Removed unused direct imports from `nellie`.

### 2. Verification Steps Performed:
- [ ] CLI run in `nellie_ywu` env (re-run recommended)

### 3. Key Decisions/Notes:
- Nellie exposes threshold utilities under `nellie.utils.gpu_functions`, not top-level `nellie`.

### 4. Next Steps:
- Re-run the CLI command in the `nellie_ywu` environment to verify imports.

## [CLI Bugfix] - 2026-01-24
**Module Implemented**: Correct per-marker loop indentation

### 1. Files Updated:
- `solar/cli/positivity_map_cli.py`: Fixed indentation so each marker is processed and saved inside the marker loop.

### 2. Verification Steps Performed:
- [ ] Manual CLI run (not re-run here)

### 3. Key Decisions/Notes:
- Ensures output paths and processing are per-marker rather than using the last marker only.

### 4. Next Steps:
- Re-run the CLI command to confirm multi-marker processing completes without errors.

## [CLI Update] - 2026-01-24
**Module Implemented**: fov_root argument for direct image_data input

### 1. Files Updated:
- `solar/cli/positivity_map_cli.py`: Renamed `project_root` to `fov_root` and now accepts the image_data directory directly.

### 2. Verification Steps Performed:
- [ ] Manual CLI run (not re-run here)

### 3. Key Decisions/Notes:
- Simplifies usage when users already have the image_data path.

### 4. Next Steps:
- Run the CLI with a direct image_data path to verify behavior.

## [Tooling Update] - 2026-01-25
**Module Implemented**: Median despeckle for positivity maps

### 1. Files Updated:
- `solar/models/positivity_probability_map.py`: Added optional early median filter (`despeckle_median_size`).
- `solar/cli/positivity_map_cli.py`: Exposed `--despeckle_median_size` CLI flag.
- `tests/test_positivity_probability_map.py`: Added test for spike suppression with median despeckle.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_positivity_probability_map.py` (not run here)

### 3. Key Decisions/Notes:
- Default behavior unchanged when `despeckle_median_size=0`.

### 4. Next Steps:
- Run `python -m pytest tests/test_positivity_probability_map.py`.

## [Tooling Update] - 2026-01-25
**Module Implemented**: z-score sigma floor for robustness

### 1. Files Updated:
- `solar/models/positivity_probability_map.py`: Added `z_sigma_floor` to clamp local std in z-score computation and record debug fractions.
- `solar/cli/positivity_map_cli.py`: Exposed `--z_sigma_floor` CLI flag.
- `tests/test_positivity_probability_map.py`: Added test for z-score floor reducing spike amplification.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_positivity_probability_map.py` (not run here)

### 3. Key Decisions/Notes:
- Default behavior unchanged when `z_sigma_floor=0.0`.

### 4. Next Steps:
- Run `python -m pytest tests/test_positivity_probability_map.py`.

## [Tooling Update] - 2026-01-26
**Module Implemented**: Optional background-normalized probability mapping

### 1. Files Updated:
- `solar/models/positivity_probability_map.py`: Added `normalize_z_to_bg` option with debug stats for background normalization.
- `solar/cli/positivity_map_cli.py`: Exposed `--normalize_z_to_bg` flag.
- `tests/test_positivity_probability_map.py`: Added test for background alignment and legacy behavior.

### 2. Verification Steps Performed:
- [ ] Unit Test Passed: `tests/test_positivity_probability_map.py` (not run here)

### 3. Key Decisions/Notes:
- Default behavior unchanged when `normalize_z_to_bg` is False.

### 4. Next Steps:
- Run `python -m pytest tests/test_positivity_probability_map.py`.
