"""Grid-search cellpose 2.x parameters on a few images before committing to a full run.

Loads the model once, then evaluates every combination of the --diameter /
--flow-threshold / --cellprob-threshold / --min-size values given.

Writes, into --out-dir:
  <tag>/<stem>_mask.tif   label image per combination (no _seg.npy - those are ~165 MB each)
  sweep_summary.tsv       one row per (combination, image) with cell counts and areas

where <tag> looks like d110.7_f0.60_p0.50_m15.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import tifffile as tiff
from cellpose import models


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-m", "--model", required=True, type=Path)
    p.add_argument("-i", "--img-dir", required=True, type=Path)
    p.add_argument("-o", "--out-dir", required=True, type=Path)
    p.add_argument("-n", "--n-images", type=int, default=3,
                   help="how many images to sweep, evenly spaced through the "
                        "sorted list (default: 3)")
    p.add_argument("--images", nargs="+", default=None,
                   help="explicit filenames instead of the -n sample")
    p.add_argument("-d", "--diameters", nargs="+", type=float, default=None,
                   help="default: the model's own diam_labels")
    p.add_argument("--flow-thresholds", nargs="+", type=float, default=[0.4, 0.6])
    p.add_argument("--cellprob-thresholds", nargs="+", type=float, default=[0.0, 0.5])
    p.add_argument("--min-sizes", nargs="+", type=int, default=[15])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def load_2ch(img_path):
    """Return (channel-swapped image, channel_axis) or (None, None) if unusable."""
    img = tiff.imread(str(img_path)).astype(np.float32)
    if img.ndim != 3:
        return None, None
    if img.shape[0] == 2:
        return img[[1, 0], :, :], 0
    if img.shape[-1] == 2:
        return img[:, :, [1, 0]], -1
    return None, None


def mask_stats(masks):
    labels, counts = np.unique(masks[masks > 0], return_counts=True)
    if len(counts) == 0:
        return dict(n_cells=0, median_area=0, min_area=0, max_area=0,
                    median_diam=0.0, frac_covered=0.0)
    return dict(
        n_cells=len(labels),
        median_area=int(np.median(counts)),
        min_area=int(counts.min()),
        max_area=int(counts.max()),
        median_diam=round(2 * np.sqrt(np.median(counts) / np.pi), 1),
        frac_covered=round(float(counts.sum()) / masks.size, 4),
    )


def main():
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    all_files = sorted(args.img_dir.glob("*.tiff")) + sorted(args.img_dir.glob("*.tif"))
    if not all_files:
        raise FileNotFoundError(f"No TIFF files found in {args.img_dir}")

    if args.images:
        by_name = {f.name: f for f in all_files}
        missing = [n for n in args.images if n not in by_name]
        if missing:
            raise FileNotFoundError(f"Not in {args.img_dir}: {missing}")
        files = [by_name[n] for n in args.images]
    else:
        n = min(args.n_images, len(all_files))
        idx = np.linspace(0, len(all_files) - 1, n).round().astype(int)
        files = [all_files[i] for i in idx]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model.name}")
    model = models.CellposeModel(gpu=not args.cpu, pretrained_model=str(args.model))

    diameters = args.diameters if args.diameters else [float(model.diam_labels)]
    grid = list(itertools.product(diameters, args.flow_thresholds,
                                 args.cellprob_thresholds, args.min_sizes))
    print(f"{len(grid)} combinations x {len(files)} images = {len(grid) * len(files)} evals")
    print("Images: " + ", ".join(f.name for f in files))

    # Read each image once; the grid is the expensive part, not the IO.
    loaded = []
    for f in files:
        img, axis = load_2ch(f)
        if img is None:
            print(f"  Skip {f.name}: not a 2-channel 3D image")
            continue
        loaded.append((f, img, axis))

    cols = ["tag", "image", "diameter", "flow_threshold", "cellprob_threshold",
            "min_size", "n_cells", "median_area", "min_area", "max_area",
            "median_diam", "frac_covered"]
    rows = []

    for diam, flow, prob, msize in grid:
        tag = f"d{diam:.1f}_f{flow:.2f}_p{prob:.2f}_m{msize}"
        combo_dir = args.out_dir / tag
        combo_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {tag} ===")

        for f, img, axis in loaded:
            masks, flows, styles = model.eval(
                img,
                channel_axis=axis,
                diameter=diam,
                flow_threshold=flow,
                cellprob_threshold=prob,
                resample=True,
                interp=True,
                min_size=msize,
            )
            tiff.imwrite(str(combo_dir / f"{f.stem}_mask.tif"), masks.astype(np.uint16))
            st = mask_stats(masks)
            rows.append([tag, f.name, diam, flow, prob, msize] +
                        [st[c] for c in cols[6:]])
            print(f"  {f.name}: {st['n_cells']} cells, median area {st['median_area']} px "
                  f"(diam ~{st['median_diam']} px), {st['frac_covered']:.1%} of frame")

    summary = args.out_dir / "sweep_summary.tsv"
    with open(summary, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    print(f"\nWrote {len(rows)} rows to {summary}")
    print("Compare the mask tifs across subfolders in napari/ImageJ, then pass the "
          "winning values to Run_cellpose.py.")


if __name__ == "__main__":
    main()
