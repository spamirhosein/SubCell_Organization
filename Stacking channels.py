import os
import glob
import tifffile
import numpy as np

# 1. Specify the parent folder path
parent_folder = r"D:\image_data\Hi-res_Data\Intensity"

# Find all subdirectories (each represents a different FOV)
fov_folders = [
    os.path.join(parent_folder, d) 
    for d in os.listdir(parent_folder) 
    if os.path.isdir(os.path.join(parent_folder, d))
]

# 2. Specify the channel files to merge, IN THE EXACT ORDER you want them stacked.
# Note: Cellpose typically uses Channel 1 for Cytoplasm/Membrane and Channel 2 for Nucleus.
channel_files = [
    "HH3.tiff", # Channel 1
    "Membrane.tiff"   # Channel 2
]

# 3. Specify the output folder
output_folder = r"D:\image_data\Hi-res_Data\Intensity\Stacked_Mem_Nuc"
os.makedirs(output_folder, exist_ok=True)

for fov_path in fov_folders:
    print(f"Processing FOV: {fov_path}")
    
    images = []
    # Read each channel one by one to ensure the correct order
    for chan_file in channel_files:
        # Search recursively in all subfolders
        pattern = os.path.join(fov_path, "**", chan_file)
        found_files = glob.glob(pattern, recursive=True)
        
        if found_files:
            file_path = found_files[0]  # Use the first match found
            img = tifffile.imread(file_path)
            images.append(img)
        else:
            print(f"  Warning: '{chan_file}' not found in {fov_path} or its subfolders. Skipping this FOV.")
            break
            
    # Only save if all specified channels were successfully loaded
    if len(images) == len(channel_files):
        # Stack the images along a new first axis -> Shape becomes (Channels, Y, X)
        stacked_img = np.stack(images, axis=0)
        
        # Define the output file name and path using the FOV folder name
        fov_folder_name = os.path.basename(fov_path)
        output_filename = os.path.join(output_folder, f"{fov_folder_name}.tiff")
        
        # Save as a multi-channel tiff. 
        # imagej=True and axes='CYX' ensures Cellpose and ImageJ read the dimensions correctly.
        tifffile.imwrite(
            output_filename, 
            stacked_img, 
            imagej=True, 
            metadata={'axes': 'CYX'}
        )
        print(f"  Successfully saved stacked image: {output_filename}\n")