from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from skimage import io, transform

# Expected label semantics in outputs: 0=background, 1=cytoplasm, 2=nucleus.


def _load_mask(path: Path) -> np.ndarray:
    arr = io.imread(path)
    if arr.ndim == 3 and arr.shape[0] <= 4:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _center_crop(mask: np.ndarray, center: Tuple[int, int], size: int) -> np.ndarray:
    h, w = mask.shape[:2]
    half = size // 2
    r, c = center
    r0, r1 = r - half, r + half
    c0, c1 = c - half, c + half
    if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
        raise ValueError("crop exceeds image bounds")
    return np.copy(mask[r0:r1, c0:c1])


def _downsample(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return mask
    target = (mask.shape[0] // factor, mask.shape[1] // factor)
    resized = transform.resize(
        mask,
        target,
        order=0,
        preserve_range=True,
        anti_aliasing=True,
    )
    return resized


def _find_pairs(cell_dir: Path, nuc_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    def _map(dir_path: Path) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for ext in ("*.tif", "*.tiff", "*.png"):
            for p in dir_path.glob(ext):
                stem = p.stem
                if stem.endswith("_cleaned_mask"):
                    key = stem.replace("_cleaned_mask", "")
                elif stem.endswith("_nuclear"):
                    key = stem.replace("_nuclear", "")
                else:
                    key = stem
                mapping[key] = p
        return mapping

    cell_map = _map(cell_dir)
    nuc_map = _map(nuc_dir)
    if not cell_map:
        raise SystemExit("No cell masks found in cell_dir.")
    pairs: Dict[str, Tuple[Path, Path]] = {}
    missing_nuc: List[str] = []
    for key, cell_path in cell_map.items():
        nuc_path = nuc_map.get(key)
        if nuc_path is None:
            missing_nuc.append(key)
            continue
        pairs[key] = (cell_path, nuc_path)
    if not pairs:
        raise SystemExit("No matching FOV stems found between cell_dir (reference) and nuclear_dir.")
    if missing_nuc:
        print(f"Warning: {len(missing_nuc)} cell files had no matching nuclear mask and were skipped.")
    return pairs


def _iter_cells(mask: np.ndarray) -> Iterable[int]:
    ids = np.unique(mask)
    for cid in ids:
        if cid > 0:
            yield int(cid)


def _crop_and_label(cell_mask: np.ndarray, nuc_mask: np.ndarray, framesize: int, cell_id: int) -> np.ndarray:
    mask = cell_mask == cell_id
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("cell id missing")
    r_center = int(rows.mean())
    c_center = int(cols.mean())
    half = framesize // 2
    h, w = cell_mask.shape[:2]
    if (
        r_center < half
        or c_center < half
        or r_center > h - half
        or c_center > w - half
    ):
        raise ValueError("cell too close to edge")

    cell_crop = _center_crop(cell_mask, (r_center, c_center), framesize)
    nuc_crop = _center_crop(nuc_mask, (r_center, c_center), framesize)

    label = np.zeros_like(cell_crop, dtype=np.uint8)
    label[cell_crop == cell_id] = 1
    label[(cell_crop == cell_id) & (nuc_crop > 0)] = 2
    return label


def _calculate_centroid(label: np.ndarray, value: int) -> Tuple[float, float]:
    ys, xs = np.where(label == value)
    if ys.size == 0 or xs.size == 0:
        raise ValueError(f"No pixels found for label value {value}")
    return float(xs.mean()), float(ys.mean())


def _align_label(label: np.ndarray) -> np.ndarray:
    # Center cell mass to image center.
    try:
        cell_cx, cell_cy = _calculate_centroid(label > 0, True)
    except ValueError:
        return label
    h, w = label.shape
    target = (w / 2.0, h / 2.0)
    translation = (target[0] - cell_cx, target[1] - cell_cy)
    tform = transform.AffineTransform(translation=translation)
    aligned = transform.warp(label, tform.inverse, order=0, mode="constant", preserve_range=True)

    # Rotate long axis of cell (value 1 or 2) to diagonal.
    ys, xs = np.where(aligned > 0)
    if xs.size >= 2:
        coords = np.column_stack((xs, ys))
        cov = np.cov(coords.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        angle = np.degrees(np.arctan2(principal[1], principal[0])) - 45.0
        aligned = transform.rotate(aligned, angle, resize=False, order=0, mode="constant", preserve_range=True)

    # Ensure nucleus (value 2) is upper-left-ish.
    try:
        nuc_cx, nuc_cy = _calculate_centroid(aligned, 2)
        step1 = nuc_cx + nuc_cy - aligned.shape[1] < 0
        if step1:
            aligned = transform.rotate(aligned, 180, resize=False, order=0, mode="constant", preserve_range=True)
        nuc_cx, nuc_cy = _calculate_centroid(aligned, 2)
        step2 = nuc_cx - nuc_cy > 0
        if step2:
            aligned = np.flipud(aligned)
            aligned = transform.rotate(aligned, -90, resize=False, order=0, mode="constant", preserve_range=True)
    except ValueError:
        pass

    aligned = np.rint(aligned).astype(np.uint8)
    return aligned


def process_fov(cell_path: Path, nuc_path: Path, out_dir: Path, framesize: int, downsample: int) -> int:
    cell_mask = _load_mask(cell_path)
    nuc_mask = _load_mask(nuc_path)
    saved = 0
    for cid in _iter_cells(cell_mask):
        try:
            label = _crop_and_label(cell_mask, nuc_mask, framesize, cid)
        except ValueError:
            continue
        label = _align_label(label)
        if downsample > 1:
            label = _downsample(label, downsample)
        label = label.astype(np.uint8)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cell_path.stem}_cell_{cid}.png"
        io.imsave(out_path, label, check_contrast=False)
        saved += 1
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build combined masks (0 background, 1 cytoplasm, 2 nucleus) cropped per cell. "
            "Pairs cell and nuclear masks by FOV stem and writes PNG crops."
        )
    )
    parser.add_argument("--cell_dir", type=Path, required=True, help="Directory with cell masks (e.g., *_cleaned_mask.tiff)")
    parser.add_argument("--nuclear_dir", type=Path, required=True, help="Directory with nuclear masks (e.g., *_nuclear.tiff)")
    parser.add_argument("--out_dir", type=Path, default=Path("SCALER/SCALER_masks"), help="Output directory for combined PNGs")
    parser.add_argument("--framesize", type=int, default=256, help="Crop size for each cell")
    parser.add_argument("--downsample", type=int, default=2, help="Downsample factor after cropping (1 keeps original size)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = _find_pairs(args.cell_dir, args.nuclear_dir)
    total = 0
    for stem, (cell_path, nuc_path) in pairs.items():
        saved = process_fov(cell_path, nuc_path, args.out_dir, args.framesize, args.downsample)
        total += saved
        print(f"Processed {stem}: saved {saved} cells")
    print(f"Done. Saved {total} crops to {args.out_dir}")


if __name__ == "__main__":
    main()
