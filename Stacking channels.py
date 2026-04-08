"""
Stack multi-channel TIFF images for Cellpose processing.

This script processes all FOV (Field of View) subfolders within a parent directory,
merging specified channel files into a single multi-channel TIFF file suitable for Cellpose.

Usage:
    python "Stacking channels.py" <parent_folder_path>

Example:
    python "Stacking channels.py" /path/to/data/positivity_map
"""

import os
import sys
import tifffile
import numpy as np

# ============================================================================
# VALIDATION: Ensure parent folder is provided via command-line argument
# ============================================================================

if len(sys.argv) < 2:
    print("Usage: python 'Stacking channels.py' <parent_folder_path>")
    print("Example: python 'Stacking channels.py' /omics/odcf/analysis/OE0622_projects/mibi_shared/Amir/SubCOrg_Opt/New_Data/PM/positivity_map")
    sys.exit(1)

parent_folder = sys.argv[1]

if not os.path.isdir(parent_folder):
    print(f"Error: Folder does not exist: {parent_folder}")
    sys.exit(1)

# ============================================================================
# CONFIGURATION: Define channels to merge
# ============================================================================

# Channel file names in the order they should be stacked
# Channel 1: Nucleus marker (NaK_ATPase_HLA-I)
# Channel 2: Membrane/Cytoplasm marker (HH3)
channel_files = [
    "NaK_ATPase_HLA-I.tiff",
    "HH3.tiff"
]

# ============================================================================
# PROCESSING: Iterate through all subfolders and create stacked images
# ============================================================================

# Discover all subfolders in the parent directory
fov_folders = [os.path.join(parent_folder, subdir) 
               for subdir in os.listdir(parent_folder)
               if os.path.isdir(os.path.join(parent_folder, subdir))]

for fov_path in fov_folders:
    print(f"Processing FOV: {fov_path}")
    
    images = []
    
    # Load each channel image in the specified order
    for chan_file in channel_files:
        file_path = os.path.join(fov_path, chan_file)
        
        if os.path.exists(file_path):
            img = tifffile.imread(file_path)
            images.append(img)
        else:
            print(f"  Warning: '{chan_file}' not found in {fov_path}. Skipping this FOV.")
            break
    
    # Save the stacked image only if all channels were successfully loaded
    if len(images) == len(channel_files):
        # Stack channels along axis 0: resulting shape is (Channels, Height, Width)
        stacked_img = np.stack(images, axis=0)
        
        output_filename = os.path.join(fov_path, "merged_for_cellpose.tiff")
        
        # Write as ImageJ-compatible multi-channel TIFF
        # imagej=True and metadata ensure Cellpose reads dimensions correctly
        tifffile.imwrite(
            output_filename, 
            stacked_img, 
            imagej=True, 
            metadata={'axes': 'CYX'}
        )
        print(f"  Successfully saved: {output_filename}\n")