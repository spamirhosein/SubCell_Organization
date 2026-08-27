from __future__ import annotations

"""Build a slim Stage 2 training manifest with only IDs, paths, and conditioning columns.

Example:
python -m solar.datasets.build_stage2_training_manifest \
  --manifest_in manifests/stage2_manifest.parquet \
  --out manifests/stage2_train_manifest_slim.parquet \
  --stack_key stack128_path \
  --mask_key mask128_path \
  --sample_id_key sample_id \
  --cell_id_key cell_id \
  --passthrough_cols split
"""

import argparse
from pathlib import Path
from typing import Iterable, Sequence

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


def _cols_with_prefix(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def build_stage2_training_manifest(
    manifest_in: str | Path | pd.DataFrame,
    out: Path,
    stack_key: str = "stack128_path",
    mask_key: str | None = None,
    sample_id_key: str = "sample_id",
    cell_id_key: str = "cell_id",
    cond_cell_prefix: str = "cond_cell_",
    cond_sample_prefix: str = "cond_sample_",
    mu_shape_prefix: str = "mu_shape_",
    logvar_shape_prefix: str = "logvar_shape_",
    passthrough_cols: Sequence[str] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    df = _load_table(manifest_in)

    if stack_key not in df.columns:
        raise ValueError(f"Missing required stack_key column '{stack_key}'")
    if sample_id_key not in df.columns:
        raise ValueError(f"Missing required sample_id_key column '{sample_id_key}'")

    mu_cols = _cols_with_prefix(df, mu_shape_prefix)
    logvar_cols = _cols_with_prefix(df, logvar_shape_prefix)
    if len(mu_cols) == 0 or len(logvar_cols) == 0:
        raise ValueError("mu_shape_* and logvar_shape_* columns are required")
    if len(mu_cols) != len(logvar_cols):
        raise ValueError(
            f"mu_shape/logvar_shape count mismatch: {len(mu_cols)} vs {len(logvar_cols)}"
        )

    cond_cell_cols = _cols_with_prefix(df, cond_cell_prefix)
    cond_sample_cols = _cols_with_prefix(df, cond_sample_prefix)

    keep_cols: list[str] = [stack_key, sample_id_key]
    if cell_id_key in df.columns:
        keep_cols.append(cell_id_key)
    if mask_key and mask_key in df.columns:
        keep_cols.append(mask_key)

    keep_cols.extend(cond_cell_cols)
    keep_cols.extend(cond_sample_cols)
    keep_cols.extend(mu_cols)
    keep_cols.extend(logvar_cols)

    if passthrough_cols:
        for col in passthrough_cols:
            if col in df.columns and col not in keep_cols:
                keep_cols.append(col)

    slim = df.loc[:, keep_cols].copy()

    latent_dim = len(mu_cols)
    print(f"Inferred shape latent dim: {latent_dim}")
    print(
        "Prefixes used: "
        f"cond_cell='{cond_cell_prefix}', cond_sample='{cond_sample_prefix}', "
        f"mu_shape='{mu_shape_prefix}', logvar_shape='{logvar_shape_prefix}'"
    )

    _save_table(slim, out, overwrite=overwrite)
    return slim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a slim Stage 2 training manifest with only IDs, paths, and conditioning columns."
    )
    parser.add_argument("--manifest_in", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stack_key", type=str, default="stack128_path")
    parser.add_argument("--mask_key", type=str, default=None)
    parser.add_argument("--sample_id_key", type=str, default="sample_id")
    parser.add_argument("--cell_id_key", type=str, default="cell_id")
    parser.add_argument("--cond_cell_prefix", type=str, default="cond_cell_")
    parser.add_argument("--cond_sample_prefix", type=str, default="cond_sample_")
    parser.add_argument("--mu_shape_prefix", type=str, default="mu_shape_")
    parser.add_argument("--logvar_shape_prefix", type=str, default="logvar_shape_")
    parser.add_argument(
        "--passthrough_cols",
        nargs="*",
        default=None,
        help="Optional extra columns to keep verbatim (e.g. split, group labels).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_stage2_training_manifest(
        manifest_in=args.manifest_in,
        out=args.out,
        stack_key=args.stack_key,
        mask_key=args.mask_key,
        sample_id_key=args.sample_id_key,
        cell_id_key=args.cell_id_key,
        cond_cell_prefix=args.cond_cell_prefix,
        cond_sample_prefix=args.cond_sample_prefix,
        mu_shape_prefix=args.mu_shape_prefix,
        logvar_shape_prefix=args.logvar_shape_prefix,
        passthrough_cols=args.passthrough_cols,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(manifest)} rows with {len(manifest.columns)} columns to {args.out}")


if __name__ == "__main__":
    main()
