from __future__ import annotations

import argparse
from pathlib import Path

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


def _resolve_mask_path(mask_dir: Path, template: str, fov: str, cell_mask_id: int) -> Path:
    return mask_dir / template.format(fov=fov, cell_mask_id=cell_mask_id)


def filter_stage2_to_stage1_masks(
    cell_table_in: str | Path | pd.DataFrame,
    mask_dir: Path,
    cell_table_out: Path | None,
    fov_column: str = "fov_name",
    cell_mask_id_column: str = "cell_mask_id",
    mask_filename_template: str = "{fov}_cleaned_mask_cell_{cell_mask_id}.png",
    report_missing_out: Path | None = None,
    require_unique: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    add_mask_path_column: str | None = None,
) -> pd.DataFrame:
    df = _load_table(cell_table_in)
    if fov_column not in df.columns:
        raise ValueError(f"Missing fov column '{fov_column}'")
    if cell_mask_id_column not in df.columns:
        raise ValueError(f"Missing cell id column '{cell_mask_id_column}'")

    if limit is not None:
        df = df.head(limit).copy()

    dup_mask = df.duplicated(subset=[fov_column, cell_mask_id_column], keep=False)
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        msg = f"Found {dup_count} duplicate rows for ({fov_column}, {cell_mask_id_column})"
        if require_unique:
            raise ValueError(msg)
        print(f"Warning: {msg}")

    mask_paths: list[Path] = []
    has_masks: list[bool] = []
    for _, row in df.iterrows():
        fov = str(row[fov_column])
        try:
            cell_mask_id = int(row[cell_mask_id_column])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid cell_mask_id in column '{cell_mask_id_column}'") from exc
        mask_path = _resolve_mask_path(mask_dir, mask_filename_template, fov, cell_mask_id)
        mask_paths.append(mask_path)
        has_masks.append(mask_path.exists())

    df = df.copy()
    df["has_stage1_mask"] = has_masks
    if add_mask_path_column:
        df[add_mask_path_column] = [str(p) for p in mask_paths]

    df_keep = df[df["has_stage1_mask"]].copy()
    df_missing = df[~df["has_stage1_mask"]].copy()

    total = len(df)
    kept = len(df_keep)
    missing = len(df_missing)
    pct = (kept / total * 100.0) if total else 0.0
    print(f"Total rows: {total}")
    print(f"Kept rows: {kept}")
    print(f"Missing rows: {missing}")
    print(f"Percent kept: {pct:.2f}%")

    if missing:
        cols = [fov_column, cell_mask_id_column]
        if add_mask_path_column:
            cols.append(add_mask_path_column)
        else:
            preview_paths = [str(p) for p, ok in zip(mask_paths, has_masks) if not ok]
            df_missing = df_missing.copy()
            df_missing["_mask_path"] = preview_paths
            cols.append("_mask_path")
        print("Missing examples (first 5):")
        print(df_missing[cols].head(5).to_string(index=False))
        if "_mask_path" in df_missing.columns:
            df_missing = df_missing.drop(columns=["_mask_path"])

    if dry_run:
        return df_keep

    if cell_table_out is None:
        raise ValueError("cell_table_out is required unless --dry_run is set")
    _save_table(df_keep, cell_table_out)
    if report_missing_out is not None:
        _save_table(df_missing, report_missing_out)

    return df_keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Stage 2 cell table to rows with Stage 1 combined-mask PNGs"
    )
    parser.add_argument("--cell_table_in", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--cell_table_out", type=Path, required=False)
    parser.add_argument("--fov_column", type=str, default="fov_name")
    parser.add_argument("--cell_mask_id_column", type=str, default="cell_mask_id")
    parser.add_argument(
        "--mask_filename_template",
        type=str,
        default="{fov}_cleaned_mask_cell_{cell_mask_id}.png",
    )
    parser.add_argument("--report_missing_out", type=Path, default=None)
    parser.add_argument("--require_unique", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--add_mask_path_column", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_table_out = None if args.dry_run else args.cell_table_out
    filter_stage2_to_stage1_masks(
        cell_table_in=args.cell_table_in,
        mask_dir=args.mask_dir,
        cell_table_out=cell_table_out,
        fov_column=args.fov_column,
        cell_mask_id_column=args.cell_mask_id_column,
        mask_filename_template=args.mask_filename_template,
        report_missing_out=args.report_missing_out,
        require_unique=args.require_unique,
        dry_run=args.dry_run,
        limit=args.limit,
        add_mask_path_column=args.add_mask_path_column,
    )


if __name__ == "__main__":
    main()
