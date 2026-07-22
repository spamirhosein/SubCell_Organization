import tifffile
import numpy as np
import os
from pathlib import Path

# Define directories
# input_dir is the PARENT folder that contains one or more subfolders (named
# after markers), and the .tiff files live inside those subfolders.
input_dir = '/omics/odcf/analysis/OE0622_projects/mibi_shared/Haanh/subCOrg_pixel/mibi/data/Ilastik_Projects/Annotations'  # Change this to your parent input directory
# output_dir is where the *_Channel3 folders are written.
output_dir = '/omics/odcf/analysis/OE0622_projects/mibi_shared/Amir/preprocessing/Segmentation/positivity_map_v3/Channel3'  # Change this to your desired output directory

os.makedirs(output_dir, exist_ok=True)

# Marker names to exclude from processing.
exclude_markers = {
    'NaK_ATPase_HLA-I_old',
    'Rab5a_PE_not_annotated',
}


def should_process(subfolder):
    """Return True if this subfolder should be processed."""
    return subfolder not in exclude_markers


# Process each subfolder inside the parent input directory
for subfolder in os.listdir(input_dir):
    subfolder_path = os.path.join(input_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    if not should_process(subfolder):
        print(f'Excluded subfolder: {subfolder}')
        continue

    # Process all TIFF files in this subfolder
    for filename in os.listdir(subfolder_path):
        if not filename.lower().endswith(('.tiff', '.tif')):
            continue

        input_path = os.path.join(subfolder_path, filename)

        try:
            # Load the multi-channel TIFF
            img_array = tifffile.imread(input_path)

            # Extract channel 3 (0-indexed, so index 2)
            if img_array.ndim == 3:
                channel_3 = img_array[:, :, 2]
            else:
                print(f'Skipped {filename}: Expected 3D array (height, width, channels)')
                continue

            # Filename has FOV_MARKER structure, e.g. A_1a_C01_R01_Catalase.tiff
            # FOV = everything before the last underscore, MARKER = the last segment.
            base_name = Path(filename).stem
            if '_' not in base_name:
                print(f'Skipped {filename}: cannot split into FOV_MARKER')
                continue
            fov, marker = base_name.rsplit('_', 1)

            # Output layout: <output_dir>/<FOV>/<MARKER>.tiff
            # Markers from every input subfolder are grouped by FOV, so all
            # markers of the same FOV land in the same folder.
            fov_dir = os.path.join(output_dir, fov)
            os.makedirs(fov_dir, exist_ok=True)
            output_path = os.path.join(fov_dir, f'{marker}.tiff')

            # Save as new TIFF
            tifffile.imwrite(output_path, channel_3)
            print(f'Processed: {subfolder}/{filename} -> {fov}/{marker}.tiff')

        except Exception as e:
            print(f'Error processing {filename}: {e}')
