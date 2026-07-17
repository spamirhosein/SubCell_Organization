import argparse
import numpy as np
import tifffile as tiff
import os
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from solar.models.positivity_probability_map import positivity_probability_map

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_marker(marker_file, fov_name, fov_path, args):
    """Processes a specific TIFF file within a FOV folder."""
    input_path = os.path.join(fov_path, marker_file)
    marker_name = os.path.splitext(marker_file)[0]
    
    # Define output structure
    output_dir = os.path.join(args.output_root, fov_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{marker_name}_pm.tiff")

    try:
        I = tiff.imread(input_path)
        
        result = positivity_probability_map(
            I,
            asinh_cofactor=args.asinh_cofactor,
            # ... [remaining params stay the same] ...
            sigmoid_slope=args.sigmoid_slope,
            tile_size=None if args.tile_free else args.tile_size,
            tile_overlap=0 if args.tile_free else args.tile_overlap,
            min_component_area_px=args.min_component_area_px,
            return_debug=args.debug
        )

        P = result[0] if args.debug else result
        tiff.imwrite(output_path, P.astype(np.float32), compression='zlib')
        return f"Done: {fov_name}/{marker_name}"
    
    except Exception as e:
        return f"FAILED: {fov_name}/{marker_name} -> {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="SubCell Positivity Mapper")
    parser.add_argument("fov_root", type=str)
    parser.add_argument("output_root", type=str)
    parser.add_argument("--markers", nargs='*', help="Specific markers or leave empty for all.")
    parser.add_argument("--jobs", type=int, default=6, help="Optimized for your 8-core CPU")
    # ... [Keep other arguments as defaults] ...
    args = parser.parse_args()

    fov_dirs = [d for d in os.listdir(args.fov_root) if os.path.isdir(os.path.join(args.fov_root, d))]
    tasks = []

    for fov in fov_dirs:
        fov_path = os.path.join(args.fov_root, fov)
        # Get all .tif files, but ignore the Excel/Metadata
        files = [f for f in os.listdir(fov_path) if f.lower().endswith(('.tif', '.tiff'))]
        
        for f in files:
            marker_name = os.path.splitext(f)[0]
            # If markers are specified, only do those. Otherwise, do all.
            if not args.markers or marker_name in args.markers:
                tasks.append((f, fov, fov_path))

    logging.info(f"Initialized: {len(tasks)} total maps to generate.")

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        process_func = partial(process_single_marker, args=args)
        results = list(executor.map(lambda p: process_func(*p), tasks))

    for idx, res in enumerate(results):
        logging.info(f"[{idx+1}/{len(tasks)}] {res}")

if __name__ == "__main__":
    main()