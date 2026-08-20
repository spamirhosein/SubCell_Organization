from pathlib import Path
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from cellpose import models, io

# ---- tuning knobs for the gradient-only reconstruction ----
CELLPROB_CUTOFF = 0.8    # foreground; raise to tighten past-gradient boundaries
DIV_SIGMA       = 1    # smoothing of divergence before seeding
MIN_DISTANCE    = 10      # min pixel gap between seeds; raise if over-segmenting
MIN_SIZE        = 30
USE_EDGE_AWARE_SEEDING = True  # use distance transform + gradient for seed placement
DIST_PERCENTILE = 35    # percentile of distance transform for seed candidates (higher = interior only)
GRAD_PERCENTILE = 25    # percentile of gradient magnitude for seed candidates (lower = smooth regions)
# -----------------------------------------------------------

def edge_aware_seeding(dP, cellprob, fg, div, CELLPROB_CUTOFF=0.8, MIN_DISTANCE=10):
    """
    Edge-aware seed placement using distance transform and gradient magnitude.
    
    Finds reliable cell centers by combining:
    1. Distance transform: prioritizes interior pixels (far from boundaries)
    2. Gradient magnitude: avoids noisy/weak boundary regions
    3. Divergence: combines with watershed elevation
    
    This helps segment dim cells by placing seeds in stable cell interiors,
    then letting watershed flow toward weakly-defined boundaries.
    """
    mag = np.sqrt(dP[0]**2 + dP[1]**2)
    mag_normalized = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    
    # Compute distance transform on foreground
    # High values = cell interior, low values = near boundary
    dist_transform = distance_transform_edt(fg.astype(np.uint8))
    dist_normalized = dist_transform / (dist_transform.max() + 1e-6)
    
    # Get percentile-based thresholds for this image
    dist_thresh = np.percentile(dist_normalized[fg], DIST_PERCENTILE)
    grad_thresh = np.percentile(mag_normalized[fg], GRAD_PERCENTILE)
    
    # Create seed mask: interior regions (high distance) with low gradient (smooth)
    # These are reliable cell centers even in dim regions
    seed_mask = (
        (dist_normalized > dist_thresh) &      # In cell interior
        (mag_normalized < grad_thresh) &        # In smooth region (not at noisy boundary)
        fg                                       # In foreground
    )
    
    print(f"    Edge-aware seeding: {seed_mask.sum()} candidate pixels")
    
    if seed_mask.sum() == 0:
        print("    Warning: No edge-aware seeds, using fallback")
        # Fallback: use original method
        seed_mask = (
            (div < np.percentile(div[fg], 5)) &
            (mag < np.percentile(mag[fg], 20)) &
            fg
        )
    
    # Place seeds at local maxima of distance transform within seed mask
    # This ensures seeds are maximally interior (most robust positions)
    seed_coords = peak_local_max(
        dist_normalized,
        min_distance=MIN_DISTANCE,
        labels=seed_mask.astype(np.uint8)
    )
    
    # Convert to seed array
    seeds = np.zeros_like(div, dtype=np.int32)
    for i, (y, x) in enumerate(seed_coords, start=1):
        seeds[y, x] = i
    
    print(f"    Placed {len(seed_coords)} seeds")
    return seeds, seed_coords

model_path = r"C:\Users\Amirhossein\.cellpose\models\TN2_hiRes_Amir_posMap2"
parent_dir = Path(r"D:\image_data\Ha Anh\stack2\sorted_by_quality\High")

print("Loading model...")
model = models.CellposeModel(pretrained_model=model_path)

# Iterate through each subfolder in parent directory
subfolders = sorted([d for d in parent_dir.iterdir() if d.is_dir()])
print(f"Found {len(subfolders)} subfolders in {parent_dir}\n")

for img_dir in subfolders:
    print(f"Processing folder: {img_dir.name}")
    
    tiff_files = sorted(img_dir.glob("*.tiff")) + sorted(img_dir.glob("*.tif"))
    if not tiff_files:
        print(f"  No TIFF files found in {img_dir.name}; skipping.")
        continue
    
    print(f"  Found {len(tiff_files)} TIFF file(s)")
    
    for img_path in tiff_files:
        print(f"\nProcessing: {img_path.name}")
        img = tiff.imread(str(img_path)).astype(np.float32)
        if img.ndim != 3: 
            print(f"  Skip: {img.shape}"); continue
        if img.shape[0] == 2:
            img_for_model = img[[1, 0], :, :]; channel_axis = 0
        elif img.shape[-1] == 2:
            img_for_model = img[:, :, [1, 0]]; channel_axis = -1
        else:
            print(f"  Skip: {img.shape}"); continue

        masks_default, flows, styles = model.eval(
            img_for_model, channel_axis=channel_axis, diameter=110,
            flow_threshold=0.6, cellprob_threshold=0.5,
            resample=True, interp=True, min_size=15,
        )

        # flows straight from eval: [0]=RGB(H,W,3), [1]=dP(2,H,W), [2]=cellprob(H,W), [3]=bd(2,H,W)
        print("  flow shapes:", [getattr(f, "shape", type(f)) for f in flows])
        dP       = flows[1].astype(np.float32)
        cellprob = flows[2].astype(np.float32)
        assert dP.ndim == 3 and dP.shape[0] == 2, f"Unexpected dP shape {dP.shape}"

        # --- Method: divergence-based watershed ---
        div = np.gradient(dP[0], axis=0) + np.gradient(dP[1], axis=1)
        div = gaussian_filter(div, sigma=DIV_SIGMA)

        fg = cellprob > CELLPROB_CUTOFF
        if fg.sum() == 0:
            print("  No foreground after cellprob cutoff; skipping."); continue
        
        # --- Seed placement ---
        if USE_EDGE_AWARE_SEEDING:
            print("  Using edge-aware seeding...")
            seeds, seed_coords = edge_aware_seeding(
                dP, cellprob, fg, div,
                CELLPROB_CUTOFF=CELLPROB_CUTOFF,
                MIN_DISTANCE=MIN_DISTANCE
            )
        else:
            print("  Using standard seeding...")
            mag = np.sqrt(dP[0]**2 + dP[1]**2)
            seed_mask = (
                (div < np.percentile(div[fg], 5)) &
                (mag < np.percentile(mag[fg], 20)) &
                fg
            )
            seed_coords = peak_local_max(-div, min_distance=MIN_DISTANCE,
                                         labels=seed_mask.astype(np.uint8))
            seeds = np.zeros_like(div, dtype=np.int32)
            for i, (y, x) in enumerate(seed_coords, start=1):
                seeds[y, x] = i

        labels = watershed(div, markers=seeds, mask=fg)
        labels = remove_small_objects(labels, min_size=MIN_SIZE)
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.reshape(div.shape).astype(np.uint16)
        
        print(f"  default masks: {masks_default.max()}   gradient-WS masks: {labels.max()}")

        # --- save two seg files so you can flip between them in the GUI ---
        out_prefix = img_path.with_suffix("")

        # (1) default Cellpose reconstruction
        io.masks_flows_to_seg(
            [img_for_model], [masks_default], [flows], [110.0],
            [str(img_path)], channels=[2, 1],
        )
        # rename so it's not overwritten
        Path(str(out_prefix) + "_seg.npy").rename(str(out_prefix) + "_default_seg.npy")

        # (2) gradient-watershed reconstruction — this is the one GUI will auto-load
        io.masks_flows_to_seg(
            [img_for_model], [labels], [flows], [110.0],
            [str(img_path)], channels=[2, 1],
        )

        tiff.imwrite(str(out_prefix) + "_mask_gradws.tif", labels.astype(np.uint16))
        tiff.imwrite(str(out_prefix) + "_mask_default.tif", masks_default.astype(np.uint16))
        print(f"  wrote: {out_prefix.name}_seg.npy (gradient-WS) and {out_prefix.name}_default_seg.npy")

print("\nAll segmentations complete.")
