"""
merge_subcellular_channels.py
==========================

Purpose
-------
Merge several single-marker channels into ONE new channel image per FOV, written
back into the SAME folder the marker images came from, so the result behaves as
a stand-alone channel alongside the existing markers.

Input
-----
A parent folder in which each subdirectory is one FOV. Inside each FOV folder
(at any depth) there are single-channel TIFF files, one per marker. These are
expected to be POSITIVITY MAPS with values in the range 0-1.

What it does
------------
For every FOV:
  1. Locates each marker file listed in `channel_files` (recursive search).
  2. Skips the FOV entirely if any of the listed markers is missing.
  3. Merges the markers by taking the PER-PIXEL MAXIMUM across them.
     With 0-1 positivity maps this behaves like a logical OR: a pixel is positive if it is positive in ANY of the channels.
  4. Writes the result as `<output_channel_name>` into the same subfolder that
     holds the first marker file, overwriting any existing file of that name.

Output
------
One single-channel float32 TIFF per FOV, values still in 0-1, sitting next to
the source marker TIFFs.

Note
----
The merge is order-independent: `channel_files` order does not affect the
result, since the maximum is commutative. The order matters only in that the
first listed marker determines the folder the output is written to, and for
predictable console output.
"""

import os
import glob
import tifffile
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Input: parent folder containing one subfolder per FOV
parent_folder = "/omics/odcf/analysis/OE0622_projects/mibi_shared/Amir/preprocessing/Segmentation/positivity_map_v3/sorted_by_quality/High"

# Name of the new channel file written into each marker folder
output_channel_name = "Lamin_A_C_H3K9me3.tiff"

# Markers to merge into the merged image
channel_files = [
    "Lamin_A_C.tiff",
    "H3K9me3.tiff",
]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

fov_folders = [
    os.path.join(parent_folder, d)
    for d in os.listdir(parent_folder)
    if os.path.isdir(os.path.join(parent_folder, d))
]

# ---------------------------------------------------------------------------
# Merge loop
# ---------------------------------------------------------------------------

for fov_path in fov_folders:
    print(f"Processing FOV: {fov_path}")

    channel_images = []
    marker_folder = None  # folder holding the first marker -> output location

    # Load each marker in the order given above
    for chan_file in channel_files:
        pattern = os.path.join(fov_path, "**", chan_file)
        found_files = glob.glob(pattern, recursive=True)

        if found_files:
            img = tifffile.imread(found_files[0])
            channel_images.append(img)
            if marker_folder is None:
                marker_folder = os.path.dirname(found_files[0])
        else:
            print(f"  Warning: '{chan_file}' not found in {fov_path} or its subfolders. Skipping this FOV.")
            break

    # Only write output if every listed marker was found
    if len(channel_images) != len(channel_files):
        continue

    # Shape (Channels, Y, X) -> per-pixel maximum -> shape (Y, X)
    channel_array = np.stack(channel_images, axis=0)
    merged_image = np.max(channel_array, axis=0)

    # ImageJ-format TIFFs support float32 but not float64
    merged_image = merged_image.astype(np.float32)

    # Write next to the source markers, so it acts as a stand-alone channel
    output_filename = os.path.join(marker_folder, output_channel_name)

    # Overwrite any previous run
    if os.path.exists(output_filename):
        os.remove(output_filename)
        print(f"  Removed existing {os.path.basename(output_filename)} to redo it.")

    tifffile.imwrite(output_filename, merged_image, imagej=True)
    print(f"  Successfully saved merged channel: {output_filename}\n")
