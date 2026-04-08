from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from skimage import io

from solar.datasets.canonicalize import canonicalize_label_and_stack, downsample_mask, downsample_stack


@dataclass
class ExportStage2Config:
    framesize: int = 256
    target_size: int = 128
    out_dir_128: Path = Path("stage_crops_128")
    out_dir_256: Path = Path("stage_crops_256")
    mask_dir_128: Path | None = Path("stage_masks_128")
    mask_dir_256: Path | None = Path("stage_masks_256")
    save_masks: bool = False
    stack_column: str = "stack_path"
    cell_mask_column: str = "cell_mask_path"
    nuclear_mask_column: str | None = "nuclear_mask_path"
    fov_column: str = "fov_name"
    cell_id_column: str | None = None
    cell_mask_id_column: str = "cell_mask_id"
    sample_id_column: str = "sample_id"
    meta_prefix: str | None = "canon_"
    channel_names: Sequence[str] | None = None
    channel_ext: str = "tiff"
    use_centroids: bool = False
    x_column: str = "X"
    y_column: str = "Y"
    flat_output: bool = False
    filename_template: str = "{fov}_{cell_mask_id}.pt"
    relative_paths: bool = False


def _load_table(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        return path.copy()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _load_stack(path: Path, channel_names: Sequence[str] | None = None, channel_ext: str = "tiff") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        if not channel_names:
            raise ValueError("channel_names must be provided when stack_path is a directory")
        planes: list[np.ndarray] = []
        for name in channel_names:
            tiff_path = path / f"{name}.{channel_ext}"
            if not tiff_path.exists():
                alt = path / f"{name}.tif"
                if alt.exists():
                    tiff_path = alt
                else:
                    raise FileNotFoundError(f"Missing channel file: {tiff_path}")
            img = io.imread(tiff_path)
            img = np.asarray(img)
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            if img.ndim != 2:
                raise ValueError(f"Channel {name} should be 2D; got {img.shape}")
            planes.append(img.astype(np.float32))
        return np.stack(planes, axis=0)
    if path.suffix.lower() == ".pt":
        arr = torch.load(path, map_location="cpu")
        arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    else:
        arr = io.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
            pass
        else:
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported stack shape: {arr.shape}")
    return arr.astype(np.float32)


def _load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pt":
        arr = torch.load(path, map_location="cpu")
        arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    else:
        arr = io.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError(f"Mask must be single-channel; got shape {arr.shape}")
    return arr


def _center_from_mask(mask: np.ndarray, cell_id: int) -> Tuple[int, int]:
    ys, xs = np.where(mask == cell_id)
    if ys.size == 0 or xs.size == 0:
        raise ValueError(f"Cell id {cell_id} not found")
    return int(round(ys.mean())), int(round(xs.mean()))


def _pad_and_crop(arr: np.ndarray, center: Tuple[int, int], size: int, fill: float = 0.0) -> np.ndarray:
    half = size // 2
    r_center, c_center = center
    r0, r1 = r_center - half, r_center + half
    c0, c1 = c_center - half, c_center + half
    if arr.ndim == 2:
        h, w = arr.shape
        pad_top = max(0, -r0)
        pad_bottom = max(0, r1 - h)
        pad_left = max(0, -c0)
        pad_right = max(0, c1 - w)
        if pad_top or pad_bottom or pad_left or pad_right:
            arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=fill)
        r0 += pad_top
        r1 += pad_top
        c0 += pad_left
        c1 += pad_left
        return np.copy(arr[r0:r1, c0:c1])
    if arr.ndim == 3:
        c, h, w = arr.shape
        pad_top = max(0, -r0)
        pad_bottom = max(0, r1 - h)
        pad_left = max(0, -c0)
        pad_right = max(0, c1 - w)
        if pad_top or pad_bottom or pad_left or pad_right:
            arr = np.pad(arr, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=fill)
        r0 += pad_top
        r1 += pad_top
        c0 += pad_left
        c1 += pad_left
        return np.copy(arr[:, r0:r1, c0:c1])
    raise ValueError("Array must be 2D or 3D channel-first")


def _build_label(cell_mask: np.ndarray, nuc_mask: np.ndarray | None, cell_id: int) -> np.ndarray:
    label = np.zeros_like(cell_mask, dtype=np.uint8)
    label[cell_mask == cell_id] = 1
    if nuc_mask is not None:
        label[(cell_mask == cell_id) & (nuc_mask > 0)] = 2
    return label


def _save_tensor(tensor: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    torch.save(torch.from_numpy(tensor), path)


def _render_filename(template: str, fov: str, cell_mask_id: int, cell_id: str) -> str:
    return template.format(fov=fov, cell_mask_id=cell_mask_id, cell_id=cell_id)


def export_stage2_crops(df: pd.DataFrame, config: ExportStage2Config) -> pd.DataFrame:
    required_cols = [config.stack_column, config.cell_mask_column, config.cell_mask_id_column, config.fov_column]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in cell table")
    if config.sample_id_column not in df.columns:
        raise ValueError(f"Missing sample_id column '{config.sample_id_column}'")

    stack_cache: Dict[str, np.ndarray] = {}
    cell_mask_cache: Dict[str, np.ndarray] = {}
    nuc_mask_cache: Dict[str, np.ndarray] = {}
    out_rows: List[Dict[str, object]] = []

    for row in df.itertuples(index=False):
        fov = getattr(row, config.fov_column)
        cell_id_value = getattr(row, config.cell_id_column) if config.cell_id_column else None
        cell_mask_id = int(getattr(row, config.cell_mask_id_column))
        stack_path = Path(getattr(row, config.stack_column))
        cell_mask_path = Path(getattr(row, config.cell_mask_column))
        nuc_mask_path = None
        if config.nuclear_mask_column and hasattr(row, config.nuclear_mask_column):
            nuc_value = getattr(row, config.nuclear_mask_column)
            if pd.notna(nuc_value) and str(nuc_value) != "":
                nuc_mask_path = Path(nuc_value)

        if fov not in stack_cache:
            stack_cache[fov] = _load_stack(
                stack_path, channel_names=config.channel_names, channel_ext=config.channel_ext
            )
        if fov not in cell_mask_cache:
            cell_mask_cache[fov] = _load_mask(cell_mask_path)
        if nuc_mask_path and fov not in nuc_mask_cache:
            nuc_mask_cache[fov] = _load_mask(nuc_mask_path)

        stack_full = stack_cache[fov]
        cell_mask = cell_mask_cache[fov]
        nuc_mask = nuc_mask_cache.get(fov) if nuc_mask_path else None

        if config.use_centroids:
            if not hasattr(row, config.x_column) or not hasattr(row, config.y_column):
                raise ValueError("Centroid columns not found in table")
            center = (int(round(getattr(row, config.y_column))), int(round(getattr(row, config.x_column))))
        else:
            center = _center_from_mask(cell_mask, cell_mask_id)
        stack_crop = _pad_and_crop(stack_full, center, config.framesize, fill=0.0)
        cell_crop = _pad_and_crop(cell_mask, center, config.framesize, fill=0)
        nuc_crop = _pad_and_crop(nuc_mask, center, config.framesize, fill=0) if nuc_mask is not None else None

        if config.use_centroids and (cell_crop == cell_mask_id).sum() == 0:
            print(
                f"Warning: centroid crop missed cell_id={cell_mask_id} in fov={fov}; falling back to mask centroid."
            )
            center = _center_from_mask(cell_mask, cell_mask_id)
            stack_crop = _pad_and_crop(stack_full, center, config.framesize, fill=0.0)
            cell_crop = _pad_and_crop(cell_mask, center, config.framesize, fill=0)
            nuc_crop = _pad_and_crop(nuc_mask, center, config.framesize, fill=0) if nuc_mask is not None else None

        label = _build_label(cell_crop, nuc_crop, cell_mask_id)
        label_aligned, stack_aligned, meta = canonicalize_label_and_stack(label, stack_crop)
        cellmask256 = (label_aligned > 0).astype(np.float32)

        stack256 = stack_aligned.astype(np.float32)
        stack128 = downsample_stack(stack256, config.target_size)
        cellmask128 = downsample_mask(cellmask256, config.target_size) > 0.5

        resolved_cell_id = cell_id_value if cell_id_value is not None else f"{fov}__{cell_mask_id}"
        if config.flat_output:
            filename = _render_filename(config.filename_template, fov, cell_mask_id, resolved_cell_id)
            stack256_path = config.out_dir_256 / filename
            stack128_path = config.out_dir_128 / filename
        else:
            stack256_path = config.out_dir_256 / str(fov) / f"{resolved_cell_id}.pt"
            stack128_path = config.out_dir_128 / str(fov) / f"{resolved_cell_id}.pt"
        _save_tensor(stack256, stack256_path)
        _save_tensor(stack128, stack128_path)

        mask256_path = mask128_path = None
        if config.save_masks:
            if config.mask_dir_256 is None or config.mask_dir_128 is None:
                raise ValueError("mask_dir_128 and mask_dir_256 must be set when save_masks is True")
            if config.flat_output:
                filename = _render_filename(config.filename_template, fov, cell_mask_id, resolved_cell_id)
                mask256_path = config.mask_dir_256 / filename
                mask128_path = config.mask_dir_128 / filename
            else:
                mask256_path = config.mask_dir_256 / str(fov) / f"{resolved_cell_id}.pt"
                mask128_path = config.mask_dir_128 / str(fov) / f"{resolved_cell_id}.pt"
            _save_tensor(cellmask256.astype(np.float32), mask256_path)
            _save_tensor(cellmask128.astype(np.float32), mask128_path)

        if hasattr(row, "_asdict"):
            row_dict = row._asdict()
            base_row: Dict[str, object] = {col: row_dict.get(col) for col in df.columns}
        else:
            base_row = {col: row[col] for col in df.columns}
        base_row["cell_id"] = resolved_cell_id
        if config.relative_paths:
            base_row["stack256_path"] = stack256_path.name
            base_row["stack128_path"] = stack128_path.name
        else:
            base_row["stack256_path"] = stack256_path
            base_row["stack128_path"] = stack128_path
        if mask256_path and mask128_path:
            if config.relative_paths:
                base_row["mask256_path"] = mask256_path.name
                base_row["mask128_path"] = mask128_path.name
            else:
                base_row["mask256_path"] = mask256_path
                base_row["mask128_path"] = mask128_path
        if config.meta_prefix:
            for k, v in meta.items():
                base_row[f"{config.meta_prefix}{k}"] = float(v)
        out_rows.append(base_row)

    return pd.DataFrame(out_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export canonicalized Stage 2 crops (256->128) with matching masks.")
    parser.add_argument("--cell_table", type=Path, required=True, help="Parquet/CSV/TSV with per-cell rows")
    parser.add_argument("--framesize", type=int, default=256, help="Crop size before downsample")
    parser.add_argument("--target_size", type=int, default=128, help="Output size after downsample")
    parser.add_argument("--stack_column", type=str, default="stack_path")
    parser.add_argument("--cell_mask_column", type=str, default="cell_mask_path")
    parser.add_argument("--nuclear_mask_column", type=str, default="nuclear_mask_path")
    parser.add_argument("--fov_column", type=str, default="fov_name")
    parser.add_argument("--cell_id_column", type=str, default=None)
    parser.add_argument("--cell_mask_id_column", type=str, default="cell_mask_id")
    parser.add_argument("--sample_id_column", type=str, default="sample_id")
    parser.add_argument("--out_dir_256", type=Path, default=Path("stage_crops_256"))
    parser.add_argument("--out_dir_128", type=Path, default=Path("stage_crops_128"))
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--mask_dir_256", type=Path, default=Path("stage_masks_256"))
    parser.add_argument("--mask_dir_128", type=Path, default=Path("stage_masks_128"))
    parser.add_argument("--out_manifest", type=Path, default=Path("manifests/stage2_manifest.parquet"))
    parser.add_argument("--meta_prefix", type=str, default="canon_", help="Prefix for canonicalization metadata columns")
    parser.add_argument("--channel_names", nargs="+", default=None, help="Channel names (required for directory stack_path)")
    parser.add_argument("--channel_ext", type=str, default="tiff", help="Extension for channel files")
    parser.add_argument("--use_centroids", action="store_true", help="Use centroid columns for crop center")
    parser.add_argument("--x_column", type=str, default="X")
    parser.add_argument("--y_column", type=str, default="Y")
    parser.add_argument("--flat_output", action="store_true", help="Save crops in a flat folder without fov subdirs")
    parser.add_argument(
        "--filename_template",
        type=str,
        default="{fov}_{cell_mask_id}.pt",
        help="Filename template for flat output (uses fov, cell_mask_id, cell_id)",
    )
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Store relative filenames in manifest instead of absolute paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_table(args.cell_table)
    relative_paths = args.relative_paths
    if args.flat_output and not args.relative_paths:
        print("Warning: --flat_output enabled without --relative_paths; defaulting to relative filenames.")
        relative_paths = True

    cfg = ExportStage2Config(
        framesize=args.framesize,
        target_size=args.target_size,
        out_dir_128=args.out_dir_128,
        out_dir_256=args.out_dir_256,
        mask_dir_128=args.mask_dir_128 if args.save_masks else None,
        mask_dir_256=args.mask_dir_256 if args.save_masks else None,
        save_masks=args.save_masks,
        stack_column=args.stack_column,
        cell_mask_column=args.cell_mask_column,
        nuclear_mask_column=args.nuclear_mask_column,
        fov_column=args.fov_column,
        cell_id_column=args.cell_id_column,
        cell_mask_id_column=args.cell_mask_id_column,
        sample_id_column=args.sample_id_column,
        meta_prefix=args.meta_prefix,
        channel_names=args.channel_names,
        channel_ext=args.channel_ext,
        use_centroids=args.use_centroids,
        x_column=args.x_column,
        y_column=args.y_column,
        flat_output=args.flat_output,
        filename_template=args.filename_template,
        relative_paths=relative_paths,
    )
    manifest = export_stage2_crops(df, cfg)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.out_manifest.suffix.lower() in {".parquet", ".pq"}:
        manifest.to_parquet(args.out_manifest, index=False)
    elif args.out_manifest.suffix.lower() in {".tsv", ".tab"}:
        manifest.to_csv(args.out_manifest, sep="\t", index=False)
    else:
        manifest.to_csv(args.out_manifest, index=False)
    print(f"Saved manifest with {len(manifest)} rows to {args.out_manifest}")


if __name__ == "__main__":
    main()
