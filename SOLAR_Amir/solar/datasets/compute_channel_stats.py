from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from skimage import io


@dataclass
class ChannelStatsConfig:
    stack_column: str = "stack_path"
    mask_column: str = "cell_mask_path"
    channel_names: Sequence[str] | None = None
    mask_threshold: float = 0.0
    channel_ext: str = "tiff"


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


def compute_channel_stats(df: pd.DataFrame, config: ChannelStatsConfig) -> dict:
    if config.stack_column not in df.columns:
        raise ValueError(f"Missing stack column '{config.stack_column}'")
    if config.mask_column not in df.columns:
        raise ValueError(f"Missing mask column '{config.mask_column}'")

    sums: np.ndarray | None = None
    sumsq: np.ndarray | None = None
    total_count: float = 0.0

    for row in df.itertuples(index=False):
        stack_path = Path(getattr(row, config.stack_column))
        mask_path = Path(getattr(row, config.mask_column))
        stack = _load_stack(stack_path, channel_names=config.channel_names, channel_ext=config.channel_ext)
        mask = _load_mask(mask_path)
        if stack.ndim != 3:
            raise ValueError(f"Expected stack (C,H,W), got {stack.shape}")
        if mask.shape != stack.shape[1:]:
            raise ValueError(f"Mask shape {mask.shape} does not match stack spatial shape {stack.shape[1:]}")
        mask_bool = mask > config.mask_threshold
        count = float(mask_bool.sum())
        if count == 0:
            continue
        masked = stack[:, mask_bool]
        if sums is None:
            sums = masked.sum(axis=1, dtype=np.float64)
            sumsq = np.square(masked, dtype=np.float64).sum(axis=1)
        else:
            sums += masked.sum(axis=1, dtype=np.float64)
            sumsq += np.square(masked, dtype=np.float64).sum(axis=1)
        total_count += count

    if total_count == 0 or sums is None or sumsq is None:
        raise ValueError("No pixels found in any mask; cannot compute stats")

    mean = sums / total_count
    var = (sumsq / total_count) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))

    channel_names = list(config.channel_names) if config.channel_names is not None else [str(i) for i in range(len(mean))]
    if len(channel_names) != len(mean):
        raise ValueError("Length of channel_names must match number of channels")

    return {
        "channel_names": channel_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std using only pixels inside cell masks (Stage 2 normalization)."
    )
    parser.add_argument("--fov_table", type=Path, required=True, help="CSV/TSV/Parquet with stack and mask paths")
    parser.add_argument("--stack_column", type=str, default="stack_path", help="Column with full FOV stack paths")
    parser.add_argument("--mask_column", type=str, default="cell_mask_path", help="Column with integer ID masks")
    parser.add_argument("--channel_names", nargs="+", default=None, help="Optional channel names; otherwise indices")
    parser.add_argument("--channel_ext", type=str, default="tiff", help="Extension for channel files")
    parser.add_argument(
        "--mask_threshold", type=float, default=0.0, help="Pixels greater than this are treated as inside a cell"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("stage_stats/channel_stats.json"),
        help="Output path (.json or .pt) for channel stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_table(args.fov_table)
    cfg = ChannelStatsConfig(
        stack_column=args.stack_column,
        mask_column=args.mask_column,
        channel_names=args.channel_names,
        mask_threshold=args.mask_threshold,
        channel_ext=args.channel_ext,
    )
    stats = compute_channel_stats(df, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".pt":
        torch.save(stats, args.out)
    else:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
    print(f"Saved channel stats to {args.out}")


if __name__ == "__main__":
    main()
