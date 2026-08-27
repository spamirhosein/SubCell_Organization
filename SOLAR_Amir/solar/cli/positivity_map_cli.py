"""CLI for generating positivity probability maps from MIBI marker images.

Examples:
    python -m solar.cli.positivity_map_cli /path/to/image_data /path/to/output marker_x marker_y
    python -m solar.cli.positivity_map_cli /path/to/image_data /path/to/output marker_x --tile_free
    python -m solar.cli.positivity_map_cli /path/to/image_data /path/to/output marker_x --debug --tile_size 256 --tile_overlap 64
"""

import argparse
import numpy as np
import tifffile as tiff
import os
from solar.models.positivity_probability_map import positivity_probability_map

def main():
    """Parse arguments, load input, compute positivity map, and write outputs."""
    parser = argparse.ArgumentParser(description="Generate positivity probability maps for MIBI images.")
    parser.add_argument("fov_root", type=str, help="Path to the FOV root containing marker TIFFs (image_data).")
    parser.add_argument("output_root", type=str, help="Path to the root directory for saving output positivity maps.")
    parser.add_argument("markers", nargs='+', help="One or more marker names (e.g., marker_x marker_y).")
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
    parser.add_argument("--fov_filter", nargs='+', default=None, help="Specific FOV subfolder names to process (e.g., --fov_filter FOV1 FOV2). If not provided, processes all FOVs.")

    args = parser.parse_args()

    image_root = args.fov_root
    if not os.path.isdir(image_root):
        raise ValueError(f"Input FOV root not found: {image_root}")

    fov_dirs = [
        name for name in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, name))
    ]
    if not fov_dirs:
        raise ValueError(f"No FOV folders found under: {image_root}")

    # Filter FOVs if --fov_filter is specified
    if args.fov_filter:
        fov_dirs = [fov for fov in fov_dirs if fov in args.fov_filter]
        if not fov_dirs:
            raise ValueError(f"No matching FOV folders found. Requested: {args.fov_filter}")
        print(f"Processing filtered FOVs: {', '.join(fov_dirs)}")


    for fov in sorted(fov_dirs):
        fov_path = os.path.join(image_root, fov)
        has_tiff = any(
            name.lower().endswith((".tif", ".tiff"))
            for name in os.listdir(fov_path)
        )
        if not has_tiff:
            print(f"Skipping {fov}: no TIFF files found")
            continue

        missing_markers = []
        for marker in args.markers:
            marker_tif = os.path.join(fov_path, f"{marker}.tif")
            marker_tiff = os.path.join(fov_path, f"{marker}.tiff")
            if not (os.path.exists(marker_tif) or os.path.exists(marker_tiff)):
                missing_markers.append(marker)

        if missing_markers:
            print(f"Skipping {fov}: missing markers {', '.join(missing_markers)}")
            continue

        for marker in args.markers:
            input_path = os.path.join(fov_path, f"{marker}.tiff")
            if not os.path.exists(input_path):
                input_path = os.path.join(fov_path, f"{marker}.tif")
            output_path = os.path.join(args.output_root, "positivity_map", fov, f"{marker}.tiff")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load input image
            if input_path.endswith(".npy"):
                I = np.load(input_path)
            elif input_path.endswith(".tif") or input_path.endswith(".tiff"):
                I = tiff.imread(input_path)
            else:
                raise ValueError("Unsupported input file format. Use .npy or .tif/.tiff.")

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
                debug_path = output_path.replace(".tif", "_debug.npy")
                np.save(debug_path, debug)
                print(f"Debug information saved to {debug_path}")
            else:
                P = result

            tiff.imwrite(output_path, P.astype(np.float32))
            print(f"Probability map saved to {output_path}")

if __name__ == "__main__":
    main()