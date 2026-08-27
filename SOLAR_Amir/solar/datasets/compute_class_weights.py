from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def parse_manifest(manifest: Path) -> List[Path]:
    paths: List[Path] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 1:
            paths.append(Path(parts[0]))
    return paths


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def compute_weights(paths: List[Path], root: Path | None) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(3, dtype=np.int64)
    for p in paths:
        full = root / p if root else p
        arr = load_mask(full)
        vals, cts = np.unique(arr, return_counts=True)
        for v, c in zip(vals, cts):
            if 0 <= v < len(counts):
                counts[v] += int(c)
    total = counts.sum()
    if total == 0:
        raise SystemExit("No pixels counted; check manifest and paths.")
    weights = counts / float(total)
    return counts, weights


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute class weights (0/1/2) from combined mask manifest.")
    parser.add_argument("--mask_manifest", type=Path, required=True, help="Manifest TSV (first column = label map path)")
    parser.add_argument("--mask_root", type=Path, default=None, help="Optional root to prepend to relative paths")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    paths = parse_manifest(args.mask_manifest)
    if not paths:
        raise SystemExit("Manifest contained no paths after filtering comments/blank lines.")
    counts, weights = compute_weights(paths, args.mask_root)
    print(f"Pixel counts: bg={counts[0]}, cyt={counts[1]}, nuc={counts[2]}")
    print(f"Class weights (sum=1): [{weights[0]:.6f} {weights[1]:.6f} {weights[2]:.6f}]")


if __name__ == "__main__":
    main()
