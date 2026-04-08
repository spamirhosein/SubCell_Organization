from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, Subset


_CELL_RE = re.compile(r"^(.*_cell_\d+)")


def extract_group_id(p: Path) -> str:
    """Extract stable cell key from filename stem (e.g., '..._cell_412')."""
    match = _CELL_RE.match(p.stem)
    if not match:
        raise ValueError(f"Filename does not contain _cell_<digits>: {p}")
    return match.group(1)


@dataclass
class PairedRecord:
    group_id: str
    mask_path: Path
    map_path: Path


class PairedCellDataset(Dataset):
    """Index-based paired dataset returning paths (no I/O in __getitem__)."""

    def __init__(self, records: List[PairedRecord]) -> None:
        self.records = records

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Path]:  # type: ignore[override]
        rec = self.records[idx]
        return {"group_id": rec.group_id, "mask_path": rec.mask_path, "map_path": rec.map_path}


def scan_pairs(masks_dir: Path, maps_dir: Path) -> List[PairedRecord]:
    masks = sorted(masks_dir.glob("*.png"))
    maps = sorted(list(maps_dir.glob("*.tiff")) + list(maps_dir.glob("*.tif")))

    mask_map: Dict[str, Path] = {}
    for p in masks:
        gid = extract_group_id(p)
        mask_map[gid] = p

    map_map: Dict[str, Path] = {}
    for p in maps:
        gid = extract_group_id(p)
        map_map[gid] = p

    common = sorted(set(mask_map.keys()) & set(map_map.keys()))
    missing_mask = sorted(set(map_map.keys()) - set(mask_map.keys()))
    missing_map = sorted(set(mask_map.keys()) - set(map_map.keys()))

    if missing_mask or missing_map:
        raise RuntimeError(
            f"Incomplete pairs: {len(missing_mask)} missing masks, {len(missing_map)} missing maps. "
            "Ensure modalities are aligned."
        )

    records = [PairedRecord(group_id=g, mask_path=mask_map[g], map_path=map_map[g]) for g in common]
    return records


def split_indices(n: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(train_ratio * n)
    return perm[:n_train], perm[n_train:]


def save_manifest(paths: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(paths) + "\n", encoding="utf-8")


def build_and_save_manifests(records: List[PairedRecord], train_idx: List[int], val_idx: List[int], out_dir: Path) -> None:
    train_groups = [records[i].group_id for i in train_idx]
    val_groups = [records[i].group_id for i in val_idx]

    save_manifest(train_groups, out_dir / "train_groups.txt")
    save_manifest(val_groups, out_dir / "val_groups.txt")

    train_files = [f"{records[i].mask_path}\t{records[i].map_path}" for i in train_idx]
    val_files = [f"{records[i].mask_path}\t{records[i].map_path}" for i in val_idx]

    save_manifest(train_files, out_dir / "train_files.txt")
    save_manifest(val_files, out_dir / "val_files.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired train/val split for SOLAR Stage 1/2 data")
    parser.add_argument("--masks_dir", type=Path, required=True, help="Directory with mask PNGs")
    parser.add_argument("--maps_dir", type=Path, required=True, help="Directory with map TIFFs")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic split")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--out_dir", type=Path, default=Path("manifests"), help="Output directory for manifests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = scan_pairs(args.masks_dir, args.maps_dir)
    full_ds = PairedCellDataset(records)

    train_idx, val_idx = split_indices(len(full_ds), args.train_ratio, args.seed)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    build_and_save_manifests(records, train_idx, val_idx, args.out_dir)

    print(
        f"Found {len(records)} complete pairs | Train groups: {len(train_ds)} | Val groups: {len(val_ds)} | "
        f"Seed: {args.seed} | Train ratio: {args.train_ratio}"
    )


if __name__ == "__main__":
    main()
