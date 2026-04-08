from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SolarStackedDatasetStage2Config:
    channel_names: Sequence[str]
    mean: Sequence[float]
    std: Sequence[float]
    stack_key: str = "stack128_path"
    mask_key: str | None = None
    zero_background: bool = False
    data_root: Path | None = None
    mask_root: Path | None = None
    sample_id_key: str = "sample_id"
    cell_id_key: str = "cell_id"
    cond_cell_prefix: str = "cond_cell_"
    cond_sample_prefix: str = "cond_sample_"
    mu_shape_prefix: str = "mu_shape_"
    logvar_shape_prefix: str = "logvar_shape_"
    dtype: torch.dtype = torch.float32


def _load_manifest(manifest: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(manifest, pd.DataFrame):
        return manifest.copy()
    path = Path(manifest)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_channel_stats(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pt":
        obj: Mapping[str, Any] = torch.load(path)
    else:
        with open(path, "r") as f:
            obj = json.load(f)
    for key in ("channel_names", "mean", "std"):
        if key not in obj:
            raise ValueError(f"Missing key '{key}' in channel stats: {path}")
    return obj


class SolarStackedDatasetStage2(Dataset):
    """Stage 2 dataset that reads canonicalized stacked crops and conditioning vectors."""

    def __init__(self, manifest: str | Path | pd.DataFrame, config: SolarStackedDatasetStage2Config) -> None:
        self.df = _load_manifest(manifest).reset_index(drop=True)
        self.config = config

        self.mean = torch.as_tensor(config.mean, dtype=config.dtype).view(-1, 1, 1)
        self.std = torch.as_tensor(config.std, dtype=config.dtype).clamp_min(1e-6).view(-1, 1, 1)
        self.channel_names = list(config.channel_names)
        if len(self.mean) != len(self.channel_names) or len(self.std) != len(self.channel_names):
            raise ValueError("mean/std length must match number of channel_names")

        self.cond_cell_cols = self._find_columns(config.cond_cell_prefix)
        self.cond_sample_cols = self._find_columns(config.cond_sample_prefix)
        self.mu_shape_cols = self._find_columns(config.mu_shape_prefix)
        self.logvar_shape_cols = self._find_columns(config.logvar_shape_prefix)
        if not self.mu_shape_cols or not self.logvar_shape_cols:
            raise ValueError("Manifest must contain mu_shape_* and logvar_shape_* columns")
        if len(self.mu_shape_cols) != len(self.logvar_shape_cols):
            raise ValueError("mu_shape_* and logvar_shape_* column counts must match")

    def _find_columns(self, prefix: str) -> List[str]:
        cols = [c for c in self.df.columns if c.startswith(prefix)]
        cols.sort()
        return cols

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.df)

    def __getitem__(self, idx: int) -> MutableMapping[str, Any]:  # type: ignore[override]
        row = self.df.iloc[idx]
        stack_path = self._resolve_path(row[self.config.stack_key], root=self.config.data_root)
        tensor = torch.load(stack_path, map_location="cpu")
        if tensor.dim() != 3:
            raise ValueError(f"Expected stacked crop with shape (C,H,W), got {tensor.shape}")
        if tensor.shape[0] != len(self.channel_names):
            raise ValueError(
                f"Stack channels ({tensor.shape[0]}) do not match channel_names ({len(self.channel_names)})"
            )
        x = tensor.to(dtype=self.config.dtype)
        x = (x - self.mean) / self.std

        cond_cell = self._row_to_tensor(row, self.cond_cell_cols)
        cond_sample = self._row_to_tensor(row, self.cond_sample_cols)
        mu_shape = self._row_to_tensor(row, self.mu_shape_cols)
        logvar_shape = self._row_to_tensor(row, self.logvar_shape_cols)

        item: MutableMapping[str, Any] = {
            "low_res": x,
            "sample_id": int(row[self.config.sample_id_key]),
            "cell_id": str(row.get(self.config.cell_id_key, f"cell_{idx}")),
            "cond_cell": cond_cell,
            "cond_sample": cond_sample,
            "mu_shape": mu_shape,
            "logvar_shape": logvar_shape,
        }

        if self.config.mask_key and self.config.mask_key in self.df.columns:
            mask_root = self.config.mask_root if self.config.mask_root is not None else self.config.data_root
            mask_path = self._resolve_path(row[self.config.mask_key], root=mask_root)
            mask = torch.load(mask_path, map_location="cpu")
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            elif mask.dim() == 3:
                if mask.shape[0] != 1:
                    raise ValueError(f"Mask must have shape (H,W) or (1,H,W); got {tuple(mask.shape)}")
            else:
                raise ValueError(f"Mask must have shape (H,W) or (1,H,W); got {tuple(mask.shape)}")
            mask = mask.to(dtype=self.config.dtype)
            if self.config.zero_background:
                x = x * mask
                item["low_res"] = x
            item["mask"] = mask
        return item

    def _resolve_path(self, value: Any, root: Path | None) -> Path:
        path = Path(str(value))
        if path.is_absolute() or root is None:
            return path
        return root / path

    def _row_to_tensor(self, row, cols: Sequence[str]) -> torch.Tensor:
        vals = [row[c] for c in cols]
        return torch.as_tensor(vals, dtype=self.config.dtype)

    @property
    def sample_ids(self) -> list[int]:
        return [int(sid) for sid in self.df[self.config.sample_id_key].tolist()]

    @property
    def cond_cell_dim(self) -> int:
        return len(self.cond_cell_cols)

    @property
    def cond_sample_dim(self) -> int:
        return len(self.cond_sample_cols)

    @property
    def shape_latent_dim(self) -> int:
        return len(self.mu_shape_cols)