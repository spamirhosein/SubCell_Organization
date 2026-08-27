from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from solar.datasets.build_paired_split import extract_group_id


def _list_images(folder: Path) -> List[Path]:
    exts = ("*.png", "*.tif", "*.tiff")
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(folder.glob(ext)))
    return files


def _relativize(path: Path, base: Path | None) -> Path:
    if base is None:
        return path
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def _build_maps(paths: Sequence[Path]) -> Dict[str, Path]:
    mapped: Dict[str, Path] = {}
    for p in paths:
        try:
            gid = extract_group_id(p)
        except ValueError:
            continue
        mapped[gid] = p
    return mapped


def _infer_id_from_stem(path: Path) -> str:
    stem = path.stem
    if "_cell_" in stem:
        return stem.split("_cell_")[0]
    return stem


def make_manifest(
    nucleus_dir: Path,
    cell_dir: Path,
    sample_id: int,
    relative_to: Path | None,
    infer_sample_id: bool,
) -> List[str]:
    nucleus_paths = _list_images(nucleus_dir)
    cell_paths = _list_images(cell_dir)

    nuc_map = _build_maps(nucleus_paths)
    cell_map = _build_maps(cell_paths)

    common = sorted(set(nuc_map.keys()) & set(cell_map.keys()))
    missing_nucleus = sorted(set(cell_map.keys()) - set(nuc_map.keys()))
    missing_cell = sorted(set(nuc_map.keys()) - set(cell_map.keys()))

    if missing_nucleus or missing_cell:
        raise SystemExit(
            f"Incomplete pairs: {len(missing_nucleus)} missing nucleus, {len(missing_cell)} missing cell."
        )
    if not common:
        raise SystemExit("No matching _cell_XXXX stems found in the provided directories.")

    lines: List[str] = []
    for gid in common:
        nuc_path = _relativize(nuc_map[gid], relative_to)
        cell_path = _relativize(cell_map[gid], relative_to)
        sid = _infer_id_from_stem(nuc_map[gid]) if infer_sample_id else sample_id
        lines.append(f"{nuc_path}\t{cell_path}\t{sid}")
    return lines


def make_manifest_combined(mask_dir: Path, sample_id: int, relative_to: Path | None, infer_sample_id: bool) -> List[str]:
    label_paths = _list_images(mask_dir)
    if not label_paths:
        raise SystemExit("No mask files found; expected PNG/TIF in mask_dir.")
    lines: List[str] = []
    for p in sorted(label_paths):
        rel = _relativize(p, relative_to)
        sid = _infer_id_from_stem(p) if infer_sample_id else sample_id
        lines.append(f"{rel}\t{sid}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a mask manifest for SolarShapeVAE. Default output: nucleus_mask\tcell_mask\tsample_id. "
            "With --combined_mask, output: labelmap\tsample_id where labelmap has 0/1/2 (background/cytoplasm/nucleus)."
        )
    )
    parser.add_argument("--nucleus_dir", type=Path, required=True, help="Directory with nucleus masks or combined label maps")
    parser.add_argument(
        "--cell_dir",
        type=Path,
        default=None,
        help="Directory with cell masks; defaults to nucleus_dir if omitted",
    )
    parser.add_argument(
        "--combined_mask",
        action="store_true",
        help="Treat files as single label maps with 0/1/2 -> background/cytoplasm/nucleus; outputs one path per line",
    )
    parser.add_argument("--sample_id", type=int, default=0, help="Sample ID assigned to all entries (ignored if --infer_sample_id)")
    parser.add_argument(
        "--infer_sample_id",
        action="store_true",
        help="Infer sample_id from filename stem before _cell_; useful when multiple samples are mixed",
    )
    parser.add_argument(
        "--relative_to",
        type=Path,
        default=None,
        help="Optional base to make manifest paths relative (use with --mask_root at train time)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("manifests/masks_manifest.tsv"),
        help="Output TSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.combined_mask:
        lines = make_manifest_combined(args.nucleus_dir, args.sample_id, args.relative_to, args.infer_sample_id)
    else:
        cell_dir = args.cell_dir or args.nucleus_dir
        lines = make_manifest(args.nucleus_dir, cell_dir, args.sample_id, args.relative_to, args.infer_sample_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {args.out}")


if __name__ == "__main__":
    main()
