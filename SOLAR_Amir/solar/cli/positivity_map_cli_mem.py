"""CLI for generating positivity probability maps from individual MIBI images.

This variant processes each TIFF file independently without channel combination.
Each image is processed separately and output with _PM.tiff suffix.

Examples:
    python -m solar.cli.positivity_map_cli_mem /path/to/images /path/to/output
    python -m solar.cli.positivity_map_cli_mem /path/to/images /path/to/output --tile_free
    python -m solar.cli.positivity_map_cli_mem /path/to/images /path/to/output --debug --tile_size 256 --tile_overlap 64
"""

import argparse
import numpy as np
import tifffile as tiff
import os
from pathlib import Path
from solar.models.positivity_probability_map import positivity_probability_map

def main():
    """Parse arguments, find all TIFF files, and compute positivity maps individually."""
    parser = argparse.ArgumentParser(description="Generate positivity probability maps for individual MIBI images.")
    parser.add_argument("image_folder", type=str, help="Path to folder containing TIFF images.")
    parser.add_argument("output_root", type=str, help="Path to the root directory for saving output positivity maps.")
    parser.add_argument("--asinh_cofactor", type=float, default=1.0, help="Cofactor for asinh transformation.")
    parser.add_argument("--despeckle_median_size", type=int, default=0, help="Median filter size for early despeckle (0 disables).")
    parser.add_argument("--gaussian_sigmas_px", type=float, nargs='+', default=[1.5, 2.3, 3.0], help="Gaussian smoothing sigmas in pixels.")
    parser.add_argument("--z_windows_px", type=int, nargs='+', default=[31, 101], help="Window sizes for spatial z-scoring.")
    parser.add_argument("--z_sigma_floor", type=float, default=0.0, help="Minimum local std for z-score normalization (0 disables).")
    parser.add_argument("--normalize_z_to_bg", action="store_true", help="Normalize Z to background scale before sigmoid mapping.")
    parser.add_argument("--nbins", type=int, default=256, help="Number of bins for histogram computation.")
    parser.add_argument("--low_percentile", type=float, default=1.0, help="Percentile for masking low-intensity pixels.")
    parser.add_argument("--sigmoid_slope", type=float, default=0.7, help="Slope for sigmoid mapping.")
    parser.add_argument("--tile_size", type=int, default=256, help="Size of tiles for gating.")
    parser.add_argument("--tile_overlap", type=int, default=64, help="Overlap between tiles.")
    parser.add_argument("--min_component_area_px", type=int, default=30, help="Minimum area for connected components.")
    parser.add_argument("--debug", action="store_true", help="Save debug information.")
    parser.add_argument("--tile_free", action="store_true", help="Enable tile-free (whole FOV) analysis.")

    args = parser.parse_args()

    image_folder = args.image_folder
    if not os.path.isdir(image_folder):
        raise ValueError(f"Input image folder not found: {image_folder}")

    # Find all TIFF files recursively (in subdirectories)
    tiff_files = []
    for root, dirs, files in os.walk(image_folder):
        for name in files:
            if name.lower().endswith((".tif", ".tiff")):
                tiff_files.append(os.path.join(root, name))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in: {image_folder}")

    print(f"Found {len(tiff_files)} TIFF file(s) to process")

    for input_path in sorted(tiff_files):
        # Generate output filename with _PM suffix before extension
        base_name = Path(input_path).stem
        output_filename = f"{base_name}_PM.tiff"
        output_path = os.path.join(args.output_root, output_filename)

        os.makedirs(args.output_root, exist_ok=True)

        print(f"\nProcessing: {input_path}")
        
        # Load input image
        try:
            if input_path.endswith(".npy"):
                I = np.load(input_path)
            elif input_path.endswith(".tif") or input_path.endswith(".tiff"):
                I = tiff.imread(input_path)
            else:
                raise ValueError("Unsupported input file format. Use .npy or .tif/.tiff.")
        except Exception as e:
            print(f"Error loading {tiff_file}: {e}")
            continue

        try:
            # Generate positivity probability map
            result = positivity_probability_map(
                I,
                asinh_cofactor=args.asinh_cofactor,
                despeckle_median_size=args.despeckle_median_size,
                gaussian_sigmas_px=args.gaussian_sigmas_px,
                z_windows_px=args.z_windows_px,
                z_sigma_floor=args.z_sigma_floor,
                normalize_z_to_bg=args.normalize_z_to_bg,
                nbins=args.nbins,
                low_percentile=args.low_percentile,
                sigmoid_slope=args.sigmoid_slope,
                tile_size=None if args.tile_free else args.tile_size,
                tile_overlap=0 if args.tile_free else args.tile_overlap,
                min_component_area_px=args.min_component_area_px,
                return_debug=args.debug
            )

            if args.debug:
                P, debug = result
                debug_path = output_path.replace(".tiff", "_debug.npy")
                np.save(debug_path, debug)
                print(f"  Debug information saved to {debug_path}")
            else:
                P = result

            tiff.imwrite(output_path, P.astype(np.float32))
            print(f"  Probability map saved to {output_path}")
        except Exception as e:
            print(f"Error processing {tiff_file}: {e}")
            continue

if __name__ == "__main__":
    main()
