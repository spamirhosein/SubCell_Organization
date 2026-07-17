import numpy as np
import tifffile as tiff
from pathlib import Path
import shutil  # <-- New import for copying files

# 1. Define your main input and output folders
input_base_dir = Path(r"D:\image_data\Hi-res_Data\Phase 3\Training")
output_base_dir = Path(r"D:\image_data\Hi-res_Data\Phase 3\Training\No_BG")

files_processed = 0
print(f"Scanning directory: {input_base_dir}\n")

# 2. Find all TIFF files recursively (Updated to .tiff)
# IMPORTANT: Collect all files FIRST before processing to avoid infinite loop
# (since output directory is inside input directory)
tiff_files = list(input_base_dir.rglob("*.tiff"))
# Filter out files that are already in the output directory
tiff_files = [f for f in tiff_files if not str(f).startswith(str(output_base_dir))]

print(f"Found {len(tiff_files)} TIFF files to process\n")

for mibi_path in tiff_files:
    
    # 3. Construct the expected mask filename
    # .stem gets the name without extension (e.g., "Image1" from "Image1.tiff")
    mask_name = mibi_path.stem + "_seg.npy"
    mask_path = mibi_path.parent / mask_name

    if not mask_path.exists():
        print(f"⚠️ Skipping '{mibi_path.name}': No mask named '{mask_name}' found.")
        continue

    print(f"🔄 Processing '{mibi_path.name}'...")

    try:
        # --- Load Data ---
        mibi_image = tiff.imread(mibi_path)
        cellpose_data = np.load(mask_path, allow_pickle=True)

        # Extract the actual mask array for our calculations
        if cellpose_data.shape == (): 
            cellpose_mask = cellpose_data.item()['masks']
        else:
            cellpose_mask = cellpose_data

        # --- Create and Apply Mask ---
        binary_mask = cellpose_mask > 0 
        masked_image = np.zeros_like(mibi_image)

        # Apply mask based on array dimensions
        if mibi_image.ndim == 3 and mibi_image.shape[1:] == cellpose_mask.shape:
            masked_image = mibi_image * binary_mask[np.newaxis, :, :]
        elif mibi_image.ndim == 3 and mibi_image.shape[:2] == cellpose_mask.shape:
            masked_image = mibi_image * binary_mask[:, :, np.newaxis]
        elif mibi_image.ndim == 2:
            masked_image = mibi_image * binary_mask
        else:
            print(f"   ❌ Shape mismatch! Image: {mibi_image.shape}, Mask: {cellpose_mask.shape}. Skipping.")
            continue 

        # --- Setup Output Paths ---
        relative_path = mibi_path.relative_to(input_base_dir)
        output_image_path = output_base_dir / relative_path
        output_folder = output_image_path.parent
        
        # Ensure the output folders exist
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Path for the copied .npy file
        output_mask_path = output_folder / mask_name
        
        # --- Save and Copy Data ---
        # 1. Save the newly masked MIBI image
        tiff.imwrite(output_image_path, masked_image)
        
        # 2. Copy and paste the original .npy file (shutil.copy2 preserves file metadata)
        shutil.copy2(mask_path, output_mask_path)
        
        print(f"   ✅ Saved masked image and copied mask to: {output_folder.name}/")
        files_processed += 1

    except Exception as e:
        print(f"   ❌ Error processing {mibi_path.name}: {e}")

print(f"\n🎉 Batch processing complete! Successfully processed {files_processed} files.")