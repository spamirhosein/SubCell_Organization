from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


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


def _save_table(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() in {".tsv", ".tab"}:
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def _ensure_numeric(df: pd.DataFrame, cols: Sequence[str], coerce: bool) -> pd.DataFrame:
    out = df.copy()
    if coerce:
        for col in cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
    non_numeric = [col for col in cols if not pd.api.types.is_numeric_dtype(out[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}. Use --coerce_numeric to force.")
    return out


def _zscore(arr: np.ndarray, ddof: int, eps: float) -> np.ndarray:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=ddof)
    return (arr - mean) / (std + eps)


def add_fov_cond_sample_to_manifest(
    cell_table: str | Path | pd.DataFrame,
    out_table: Path,
    fov_column: str,
    feature_cols: Sequence[str],
    prefix: str = "cond_sample_",
    name_mode: str = "by_col",
    ddof: int = 0,
    eps: float = 1e-6,
    overwrite: bool = False,
    coerce_numeric: bool = False,
) -> pd.DataFrame:
    df = _load_table(cell_table)
    if fov_column not in df.columns:
        raise ValueError(f"Missing fov column '{fov_column}'")
    if not feature_cols:
        raise ValueError("feature_cols must be provided")
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing feature column '{col}'")

    df = _ensure_numeric(df, feature_cols, coerce=coerce_numeric)

    fov_table = df.groupby(fov_column, as_index=False)[list(feature_cols)].median()
    z = _zscore(fov_table[list(feature_cols)].to_numpy(dtype=float), ddof=ddof, eps=eps)

    if name_mode == "by_col":
        cond_cols = [f"{prefix}{col}" for col in feature_cols]
    elif name_mode == "enumerate":
        cond_cols = [f"{prefix}{idx}" for idx in range(len(feature_cols))]
    else:
        raise ValueError("name_mode must be 'by_col' or 'enumerate'")

    for idx, col in enumerate(cond_cols):
        fov_table[col] = z[:, idx]

    merged = df.merge(fov_table[[fov_column] + cond_cols], on=fov_column, how="left")

    if merged[cond_cols].isna().any().any():
        raise ValueError("NaNs introduced in cond_sample columns after merge")

    for col in cond_cols:
        if not pd.api.types.is_numeric_dtype(merged[col]):
            raise ValueError(f"cond_sample column '{col}' is not numeric")

    # Values must be constant within each FOV.
    for col in cond_cols:
        if (merged.groupby(fov_column)[col].nunique(dropna=False) > 1).any():
            raise ValueError(f"cond_sample column '{col}' varies within a FOV")

    _save_table(merged, out_table, overwrite=overwrite)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add per-FOV cond_sample covariates (median + zscore) to a per-cell Stage 2 manifest"
    )
    parser.add_argument("--cell_table", type=Path, required=True)
    parser.add_argument("--out_table", type=Path, required=True)
    parser.add_argument("--fov_column", type=str, default="fov_name")
    parser.add_argument("--feature_cols", nargs="+", required=True)
    parser.add_argument("--prefix", type=str, default="cond_sample_")
    parser.add_argument("--name_mode", type=str, default="by_col", choices=["by_col", "enumerate"])
    parser.add_argument("--ddof", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--coerce_numeric", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_fov_cond_sample_to_manifest(
        cell_table=args.cell_table,
        out_table=args.out_table,
        fov_column=args.fov_column,
        feature_cols=args.feature_cols,
        prefix=args.prefix,
        name_mode=args.name_mode,
        ddof=args.ddof,
        eps=args.eps,
        overwrite=args.overwrite,
        coerce_numeric=args.coerce_numeric,
    )


if __name__ == "__main__":
    main()
