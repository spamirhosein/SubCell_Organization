"""Run a cellpose 2.x model over a directory of 2-channel (nucleus, membrane) TIFFs.

Outputs, per image, into --out-dir:
  <stem>_seg.npy   loadable in the cellpose GUI
  <stem>_mask.tif  uint16 label image
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from cellpose import models, io


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-m", "--model", required=True, type=Path,
                   help="path to the cellpose model file")
    p.add_argument("-i", "--img-dir", required=True, type=Path,
                   help="directory of *.tif / *.tiff images")
    p.add_argument("-o", "--out-dir", type=Path, default=None,
                   help="output directory (default: <img-dir>/<model name>)")
    p.add_argument("-d", "--diameter", type=float, default=None,
                   help="cell diameter in px (default: the model's own diam_labels)")
    p.add_argument("--flow-threshold", type=float, default=0.6)
    p.add_argument("--cellprob-threshold", type=float, default=0.5)
    p.add_argument("--min-size", type=int, default=15)
    p.add_argument("--cpu", action="store_true",
                   help="force CPU (default: GPU)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    tiff_files = sorted(args.img_dir.glob("*.tiff")) + sorted(args.img_dir.glob("*.tif"))
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {args.img_dir}")

    # Keep runs from different models out of each other's way.
    out_dir = args.out_dir if args.out_dir is not None else args.img_dir / args.model.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model.name}")
    # cellpose 2.x indexes pretrained_model, so it must be a str, not a Path.
    model = models.CellposeModel(gpu=not args.cpu, pretrained_model=str(args.model))

    diameter = args.diameter if args.diameter is not None else float(model.diam_labels)
    print(f"Using diameter={diameter:.2f} | {len(tiff_files)} images -> {out_dir}")

    for img_path in tiff_files:
        print(f"\nProcessing: {img_path.name}")
        img = tiff.imread(str(img_path)).astype(np.float32)

        if img.ndim != 3:
            print(f"  Skip: expected 3D, got {img.shape}")
            continue
        if img.shape[0] == 2:
            img_for_model = img[[1, 0], :, :]; channel_axis = 0
        elif img.shape[-1] == 2:
            img_for_model = img[:, :, [1, 0]]; channel_axis = -1
        else:
            print(f"  Skip: expected 2 channels, got {img.shape}")
            continue

        masks, flows, styles = model.eval(
            img_for_model,
            channel_axis=channel_axis,
            diameter=diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            resample=True,
            interp=True,
            min_size=args.min_size,
        )

        out_prefix = out_dir / img_path.stem

        # --- Cellpose 2.x: pass LISTS ---
        io.masks_flows_to_seg(
            [img_for_model],
            [masks],
            [flows],
            [diameter],
            [str(out_prefix)],
            channels=[2, 1],          # membrane, nucleus (1-indexed for GUI)
        )

        seg_path = str(out_prefix) + "_seg.npy"
        if not Path(seg_path).exists():
            print(f"  WARNING: seg file not written for {img_path.name}")
            continue

        tiff.imwrite(str(out_prefix) + "_mask.tif", masks.astype(np.uint16))
        print(f"  Saved: {img_path.stem}_seg.npy and {img_path.stem}_mask.tif "
              f"({masks.max()} cells)")

    print("\nAll segmentations complete.")


if __name__ == "__main__":
    main()
