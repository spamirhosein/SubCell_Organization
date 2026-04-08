import os
import numpy as np
import cv2
import tifffile as tiff

NUC_STEM = "NaK_ATPase_HLA-I"
MEM_STEM = "HH3"

# Input and Output directories
PARENT_DIR = r"/omics/odcf/analysis/OE0622_projects/mibi_shared/Amir/SubCOrg_Opt/New_Data/PM/positivity_map"
OUTPUT_DIR = r"/omics/odcf/analysis/OE0622_projects/mibi_shared/Amir/SubCOrg_Opt/New_Data/PM/Mem_Nuc_Clahe"

def find_marker_file(folder, stem):
    for ext in (".tif", ".tiff"):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None

def clahe_uint8(img, tilesize=(12, 12), cliplimit=25.0):
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    if img.dtype == np.uint8:
        u8 = img
    else:
        img_f = img.astype(np.float32)
        p = np.percentile(img_f, 99.5)
        if p > 0:
            img_f = np.clip(img_f / p, 0, 1)
        else:
            img_f = np.zeros_like(img_f) # Failsafe against empty/black images
        u8 = np.round(img_f * 255).astype(np.uint8)

    u8 = cv2.equalizeHist(u8)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tilesize)
    return clahe.apply(u8)

# Create the output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

folders = sorted([
    os.path.join(PARENT_DIR, d)
    for d in os.listdir(PARENT_DIR)
    if os.path.isdir(os.path.join(PARENT_DIR, d))
])

if not folders:
    raise RuntimeError(f"No subfolders found in {PARENT_DIR}")

done, skipped = 0, 0

for fdir in folders:
    folder_name = os.path.basename(fdir)
    
    nuc_path = find_marker_file(fdir, NUC_STEM)
    mem_path = find_marker_file(fdir, MEM_STEM)

    if nuc_path is None or mem_path is None:
        print(f"SKIP (missing markers): {folder_name}")
        skipped += 1
        continue

    # Read and process the images
    nuc = clahe_uint8(tiff.imread(nuc_path))
    mem = clahe_uint8(tiff.imread(mem_path))

    # Stack them: Channel 0 = Membrane, Channel 1 = Nucleus
    out = np.stack([mem, nuc], axis=0).astype(np.uint8)  # (2, H, W)
    
    # Save to the new specific output directory instead of the input folder
    out_path = os.path.join(OUTPUT_DIR, f"{folder_name}.tiff")
    
    # Save with ImageJ format so Cellpose reads the channels correctly
    tiff.imwrite(out_path, out, imagej=True)

    print(f"DONE: {folder_name} -> {out_path}")
    done += 1

print(f"\nFinished. DONE={done}, SKIPPED={skipped}")