## Project Context: **SOLAR** for MIBI Organelle Analysis

### 1. The Scientific Goal
We are analyzing single-cell MIBI data (~400 samples, ~80k cells) to answer two biological questions:
- **Recruitment:** Is a specific protein (e.g., MTHFD2) recruited to specific locations (e.g., perinuclear vs peripheral) independent of cell crowding?
- **Coupling:** Is the texture/state of one organelle (e.g., mitochondrial fragmentation) statistically coupled to another (e.g., ER stress)?

### 2. The Problem
- **Crowding:** Cancer cells can have high N:C ratios; organelles are compressed, so the method must separate `crowding constraints` from `active recruitment.`
- **Batch effects:** ~400 samples can have different intensity ranges; the method must normalize/remove sample-driven intensity variation (and avoid leaking within-sample cell-type/state variation into “where” latents).
- **Rotation:** Cells are randomly oriented; the representation should be rotation-invariant (at least for morphology).

### 3. The Solution Architecture (3-Stage Hierarchy): **SOLAR**
**SOLAR = Subcellular Organelle Laddered Autoencoder for Representation.**
We build a hierarchical conditional VAE system with explicit disentanglement across scales.

#### Stage 1: The Container (**SolarShapeVAE**; VAE1 - Shape)
- **Task:** Learn cell/nucleus morphology (`available space`).
- **Input:** **Masks only** (nucleus mask + cell mask); no organelle intensity channels are used here.
- **Tech:** `e2cnn` (group-equivariant CNN) to enforce rotation-invariant morphology encoding.
- **Output latent:** \(z_{shape}\) = strictly rotation-invariant morphology descriptor.

#### Stage 2: The Map (**SolarMapVAE**; cVAE2 - Recruitment)
- **Task:** Learn **where** each organelle/marker is, given morphology and intensity covariates.
- **Input:** Low-res organelle images (per marker/channel).
- **Condition (revised):** \((\text{cond\_morph} + \text{cond\_sample} + \text{cond\_cell})\).
- **cond\_morph:** a per-cell morphology sample \(z_{shape}\) drawn via reparameterization from cached Stage-1 posterior stats \((\mu_{shape}, \log\sigma^2_{shape})\) each time the cell is fed.
- **cond\_sample:** sample-level subcellular-marker **median** intensities (continuous replacement for SampleID; one vector per sample).
- **cond\_cell:** cell-level lineage-marker **mean** intensities (captures within-sample cell-type/state effects).
- **Tech:** Standard CNN, **multi-head / independent encoders** (separate latent per marker).
- **Output latent:** \(z_{coarse}^{(marker)}\) = localization phenotype (perinuclear/polarized/diffuse, etc.).
- **Why independent:** Needed to test whether one marker moves independently of another (e.g., MTHFD2 vs TOM20).

#### Stage 3: The Texture (**SolarTextureVAE**; cVAE3 - Fine State)
- **Task:** Learn **what** the organelles look like at high resolution (texture/state: fission/fusion, reticulation, puncta, etc.).
- **Input:** High-res organelle images.
- **Condition (revised):** \((\text{cond\_morph} + z_{coarse}^{(marker)} + \text{cond\_sample} + \text{cond\_cell})\).
- **Tech:** Standard CNN with **parallel encoders** (per organelle/marker you want to compare).
- **Output latent:** \(z_{fine}^{(marker)}\) = structural/texture state.
- **Analysis:** Run CCA on \(z_{fine,Mito}\) vs \(z_{fine,ER}\) to quantify coupling.

## Implementation Roadmap for the Agent (SOLAR)
- **Data Engine:** Build `SolarDataset` (or `MIBIDataset`) with `CanonicalRotation` (align nucleus major axis) and `BalancedBatchSampler`.
- **Conditioning inputs (revised):**
  - Maintain a per-cell manifest that maps `cell_id -> {mask paths, organelle crop paths, sample_id}`.
  - Maintain an intensities table (CSV/Parquet) keyed by `cell_id` containing all marker summary intensities; derive:
    - `cond_cell` from lineage-marker means per cell.
    - `cond_sample` from subcellular-marker medians per sample (group by sample_id, then broadcast to cells).
  - After training Stage 1, precompute and cache a morphology stats table keyed by `cell_id`:
    - \(\mu_{shape}\) and \(\log\sigma^2_{shape}\) (per cell).
    - During Stage 2/3 training, resample \(\text{cond_morph}=z_{shape}\) each batch from these cached stats (fresh noise each step).
- **Model 1 (Shape):** Implement `SolarShapeVAE` using `e2cnn`; **inputs are nucleus+cell masks**.
- **Model 2 (Recruitment):** Implement `SolarMapVAE` with independent per-marker encoders, conditioned on \(\text{cond_morph}\), \(\text{cond_sample}\), and \(\text{cond_cell}\).
- **Model 3 (Texture):** Implement `SolarTextureVAE` with parallel encoders, conditioned on \(\text{cond_morph}\), \(z_{coarse}\), \(\text{cond_sample}\), and \(\text{cond_cell}\), then compute coupling (CCA).
