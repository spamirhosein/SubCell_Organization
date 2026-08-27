"""
Nucleus segmentation from HH3 channel images.
Produces integer-labeled masks (same format as CellPose).

Pipeline:
  Gaussian blur -> Otsu threshold -> fill holes ->
  remove small objects -> distance-transform watershed -> save mask
"""

import os
import glob
import numpy as np
import tifffile
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, closing, disk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


# ── configuration ────────────────────────────────────────────────────────────
INPUT_ROOT  = r"D:\image_data\New_Data\Ha Anh\HH3_Channel3\sorted_by_quality\High"
OUTPUT_ROOT = r"D:\image_data\New_Data\Ha Anh\HH3_Channel3\sorted_by_quality\High\masks"

GAUSSIAN_SIGMA   = 1.5    # pre-blur to reduce noise before thresholding
MIN_NUCLEUS_SIZE = 500    # pixels — objects smaller than this are removed
CLOSING_RADIUS   = 3      # morphological closing radius (fills gaps in membrane)
MIN_DISTANCE     = 15     # min pixel distance between watershed seed peaks
# ─────────────────────────────────────────────────────────────────────────────


def segment_image(img: np.ndarray) -> np.ndarray:
    """Return a uint16 labeled mask from a 2-D float image."""
    # 1. Smooth
    smoothed = gaussian(img, sigma=GAUSSIAN_SIGMA)

    # 2. Threshold
    thresh = threshold_otsu(smoothed)
    binary = smoothed > thresh

    # 3. Morphological closing to fill small gaps
    binary = closing(binary, disk(CLOSING_RADIUS))

    # 4. Fill holes
    binary = ndi.binary_fill_holes(binary)

    # 5. Remove tiny debris
    binary = remove_small_objects(binary, min_size=MIN_NUCLEUS_SIZE)

    # 6. Distance-transform watershed to split touching nuclei
    distance = ndi.distance_transform_edt(binary)
    coords   = peak_local_max(distance, min_distance=MIN_DISTANCE, labels=binary)
    markers  = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = ndi.label(markers)
    labeled  = watershed(-distance, markers, mask=binary)

    return labeled.astype(np.uint16)


def process_all():
    folders = sorted(glob.glob(os.path.join(INPUT_ROOT, "*")))
    folders = [f for f in folders if os.path.isdir(f)]

    print(f"Found {len(folders)} folders to process.")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for i, folder in enumerate(folders, 1):
        sample_name = os.path.basename(folder)
        tiff_path   = os.path.join(folder, f"{sample_name}.tiff")

        if not os.path.exists(tiff_path):
            # try any tiff in the folder
            hits = glob.glob(os.path.join(folder, "*.tiff")) + \
                   glob.glob(os.path.join(folder, "*.tif"))
            if not hits:
                print(f"  [{i}/{len(folders)}] SKIP {sample_name} — no TIFF found")
                continue
            tiff_path = hits[0]

        img  = tifffile.imread(tiff_path).astype(np.float32)
        # normalise if not already in [0,1]
        if img.max() > 1.0:
            img = img / img.max()

        mask = segment_image(img)

        out_path = os.path.join(OUTPUT_ROOT, f"{sample_name}_nuclear.tiff")
        tifffile.imwrite(out_path, mask, compression="zlib")

        n_nuclei = mask.max()
        print(f"  [{i}/{len(folders)}] {sample_name}: {n_nuclei} nuclei  -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    process_all()