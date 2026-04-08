from __future__ import annotations

import argparse
import json
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


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() in {".tsv", ".tab"}:
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def _parse_markers(markers: str) -> list[str]:
    if not markers:
        raise ValueError("--markers must be provided")
    items = [m.strip() for m in markers.split(",")]
    items = [m for m in items if m]
    if not items:
        raise ValueError("--markers must contain at least one marker name")
    return items


def _ensure_numeric(df: pd.DataFrame, cols: Sequence[str], coerce: bool) -> pd.DataFrame:
    out = df.copy()
    if coerce:
        for col in cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    else:
        non_numeric = [col for col in cols if not pd.api.types.is_numeric_dtype(out[col])]
        if non_numeric:
            raise ValueError(f"Non-numeric marker columns: {non_numeric}. Use --coerce_numeric to force.")
    return out


def _join_on_cell_id(
    manifest: pd.DataFrame,
    source: pd.DataFrame,
    manifest_cell_id_col: str,
    source_cell_id_col: str | None,
    source_fov_col: str,
    source_label_col: str,
) -> pd.DataFrame:
    if manifest_cell_id_col not in manifest.columns:
        raise ValueError(f"Manifest missing cell id column '{manifest_cell_id_col}'")
    src = source.copy()
    if source_cell_id_col and source_cell_id_col in src.columns:
        src_key = source_cell_id_col
    else:
        if source_fov_col not in src.columns or source_label_col not in src.columns:
            raise ValueError("Source table must include fov/label columns to derive cell_id")
        src["__cell_id"] = src[source_fov_col].astype(str) + "__" + src[source_label_col].astype(int).astype(str)
        src_key = "__cell_id"
    merged = manifest.merge(src, how="left", left_on=manifest_cell_id_col, right_on=src_key, suffixes=("", "__src"))
    return merged


def _join_on_fov_label(
    manifest: pd.DataFrame,
    source: pd.DataFrame,
    manifest_fov_col: str,
    manifest_cell_mask_id_col: str,
    source_fov_col: str,
    source_label_col: str,
) -> pd.DataFrame:
    for col in (manifest_fov_col, manifest_cell_mask_id_col):
        if col not in manifest.columns:
            raise ValueError(f"Manifest missing join column '{col}'")
    for col in (source_fov_col, source_label_col):
        if col not in source.columns:
            raise ValueError(f"Source missing join column '{col}'")
    src = source.copy()
    src["__fov"] = src[source_fov_col].astype(str)
    src["__label"] = src[source_label_col].astype(int)
    man = manifest.copy()
    man["__fov"] = man[manifest_fov_col].astype(str)
    man["__label"] = man[manifest_cell_mask_id_col].astype(int)
    merged = man.merge(src, how="left", left_on=["__fov", "__label"], right_on=["__fov", "__label"], suffixes=("", "__src"))
    merged = merged.drop(columns=["__fov", "__label"])
    return merged


def add_lineage_cond_to_manifest(
    manifest_in: str | Path | pd.DataFrame,
    source_in: str | Path | pd.DataFrame,
    out: Path,
    markers: str,
    join_on: str = "cell_id",
    cond_prefix: str = "cond_cell_",
    manifest_cell_id_col: str = "cell_id",
    manifest_fov_col: str = "fov_name",
    manifest_cell_mask_id_col: str = "cell_mask_id",
    source_cell_id_col: str | None = "cell_id",
    source_fov_col: str = "fov",
    source_label_col: str = "label",
    overwrite_existing: bool = False,
    fillna: float | None = None,
    coerce_numeric: bool = False,
    normalize: str = "none",
    stats_out: Path | None = None,
    report_unmatched: Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    manifest = _load_table(manifest_in)
    source = _load_table(source_in)

    marker_list = _parse_markers(markers)
    missing = [m for m in marker_list if m not in source.columns]
    if missing:
        raise ValueError(f"Markers missing from source table: {missing}")

    source = _ensure_numeric(source, marker_list, coerce=coerce_numeric)

    if join_on == "cell_id":
        merged = _join_on_cell_id(
            manifest,
            source,
            manifest_cell_id_col=manifest_cell_id_col,
            source_cell_id_col=source_cell_id_col,
            source_fov_col=source_fov_col,
            source_label_col=source_label_col,
        )
    elif join_on == "fov_name,cell_mask_id":
        merged = _join_on_fov_label(
            manifest,
            source,
            manifest_fov_col=manifest_fov_col,
            manifest_cell_mask_id_col=manifest_cell_mask_id_col,
            source_fov_col=source_fov_col,
            source_label_col=source_label_col,
        )
    else:
        raise ValueError("join_on must be 'cell_id' or 'fov_name,cell_mask_id'")

    out_cols = []
    for marker in marker_list:
        out_col = f"{cond_prefix}{marker}"
        if out_col in manifest.columns and not overwrite_existing:
            raise ValueError(f"Column already exists: {out_col}. Use --overwrite_existing to replace.")
        merged[out_col] = merged[marker]
        out_cols.append(out_col)

    missing_mask = merged[out_cols].isna().any(axis=1)
    if missing_mask.any():
        if report_unmatched is not None:
            report_unmatched.parent.mkdir(parents=True, exist_ok=True)
            merged.loc[missing_mask].to_csv(report_unmatched, index=False)
        if fillna is None:
            raise ValueError("Missing values found after join; use --fillna or provide complete source table.")
        merged.loc[:, out_cols] = merged[out_cols].fillna(fillna)

    if normalize == "zscore":
        stats = {}
        for col in out_cols:
            mean = float(merged[col].mean())
            std = float(merged[col].std())
            if std < 1e-6:
                std = 1e-6
            merged[col] = (merged[col] - mean) / std
            stats[col] = {"mean": mean, "std": std}
        if stats_out is not None:
            stats_out.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_out, "w") as f:
                json.dump(stats, f, indent=2)
    elif normalize != "none":
        raise ValueError("normalize must be 'none' or 'zscore'")

    # Preserve original column order; append new columns at the end if needed.
    final = manifest.copy()
    for col in out_cols:
        final[col] = merged[col].to_numpy()

    if dry_run:
        print(f"Rows: {len(final)}")
        print(f"Added columns: {out_cols}")
        return final

    _save_table(final, Path(out))
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add lineage marker conditioning columns to Stage 2 manifest")
    parser.add_argument("--manifest_in", type=Path, required=True)
    parser.add_argument("--source_in", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markers", type=str, required=True)
    parser.add_argument("--join_on", type=str, default="cell_id")
    parser.add_argument("--cond_prefix", type=str, default="cond_cell_")
    parser.add_argument("--manifest_cell_id_col", type=str, default="cell_id")
    parser.add_argument("--manifest_fov_col", type=str, default="fov_name")
    parser.add_argument("--manifest_cell_mask_id_col", type=str, default="cell_mask_id")
    parser.add_argument("--source_cell_id_col", type=str, default="cell_id")
    parser.add_argument("--source_fov_col", type=str, default="fov")
    parser.add_argument("--source_label_col", type=str, default="label")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--fillna", type=float, default=None)
    parser.add_argument("--coerce_numeric", action="store_true")
    parser.add_argument("--normalize", type=str, default="none")
    parser.add_argument("--stats_out", type=Path, default=None)
    parser.add_argument("--report_unmatched", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_lineage_cond_to_manifest(
        manifest_in=args.manifest_in,
        source_in=args.source_in,
        out=args.out,
        markers=args.markers,
        join_on=args.join_on,
        cond_prefix=args.cond_prefix,
        manifest_cell_id_col=args.manifest_cell_id_col,
        manifest_fov_col=args.manifest_fov_col,
        manifest_cell_mask_id_col=args.manifest_cell_mask_id_col,
        source_cell_id_col=args.source_cell_id_col,
        source_fov_col=args.source_fov_col,
        source_label_col=args.source_label_col,
        overwrite_existing=args.overwrite_existing,
        fillna=args.fillna,
        coerce_numeric=args.coerce_numeric,
        normalize=args.normalize,
        stats_out=args.stats_out,
        report_unmatched=args.report_unmatched,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
