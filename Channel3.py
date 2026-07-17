import tifffile
import numpy as np
import os
from pathlib import Path

# Define directories
input_dir = r'D:\image_data\Ha Anh\NaK_ATPase_HLA-I'  # Change this to your input directory
output_dir = r'D:\image_data\Ha Anh\HLA-I_Channel3'  # Change this to your output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all TIFF files in input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.tiff', '.tif')):
        input_path = os.path.join(input_dir, filename)
        
        try:
            # Load the multi-channel TIFF
            img_array = tifffile.imread(input_path)
            
            # Extract channel 3 (0-indexed, so index 2)
            if img_array.ndim == 3:
                channel_3 = img_array[:, :, 2]
            else:
                print(f'Skipped {filename}: Expected 3D array (height, width, channels)')
                continue
            
            # Create output filename
            base_name = Path(filename).stem
            output_path = os.path.join(output_dir, f'{base_name}.tiff')
            
            # Save as new TIFF
            tifffile.imwrite(output_path, channel_3)
            print(f'Processed: {filename} -> {base_name}.tiff')
        
        except Exception as e:
            print(f'Error processing {filename}: {e}')