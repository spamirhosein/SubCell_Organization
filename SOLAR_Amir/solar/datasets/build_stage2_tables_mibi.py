from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

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


def _resolve_mask_path(root: Path, fov: str, suffix: str) -> Path:
    return root / f"{fov}{suffix}"


def build_tables(
    df: pd.DataFrame,
    image_root: Path,
    cell_mask_root: Path,
    nuc_mask_root: Path | None,
    fov_col: str,
    label_col: str,
    x_col: str,
    y_col: str,
    sample_id_mode: str,
    sample_id_from_col: str | None,
    keep_cell_id: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fov_col not in df.columns:
        raise ValueError(f"Missing fov column '{fov_col}'")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'")
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Missing centroid columns '{x_col}'/'{y_col}'")

    fov_names = sorted({str(v) for v in df[fov_col].astype(str).tolist()})
    fov_to_sample: Dict[str, int] = {}
    if sample_id_mode == "enumerate":
        fov_to_sample = {fov: idx for idx, fov in enumerate(fov_names)}
    elif sample_id_mode == "from_col":
        if not sample_id_from_col or sample_id_from_col not in df.columns:
            raise ValueError("sample_id_from_col must be provided when sample_id_mode=from_col")
        sample_vals = df[[fov_col, sample_id_from_col]].drop_duplicates()
        fov_to_sample = {
            str(row[fov_col]): int(row[sample_id_from_col]) for _, row in sample_vals.iterrows()
        }
    else:
        raise ValueError("sample_id_mode must be 'enumerate' or 'from_col'")

    cell_df = df.copy()
    cell_df["fov_name"] = cell_df[fov_col].astype(str)
    cell_df["cell_mask_id"] = cell_df[label_col].astype(int)
    if "cell_id" not in cell_df.columns or not keep_cell_id:
        cell_df["cell_id"] = cell_df.apply(lambda r: f"{r['fov_name']}__{int(r['cell_mask_id'])}", axis=1)

    cell_df["stack_path"] = cell_df["fov_name"].apply(lambda f: str(image_root / f))
    cell_df["cell_mask_path"] = cell_df["fov_name"].apply(
        lambda f: str(_resolve_mask_path(cell_mask_root, f, "_cleaned_mask.tiff"))
    )
    if nuc_mask_root is not None:
        cell_df["nuclear_mask_path"] = cell_df["fov_name"].apply(
            lambda f: str(_resolve_mask_path(nuc_mask_root, f, "_nuclear.tiff"))
        )
    else:
        cell_df["nuclear_mask_path"] = pd.NA

    cell_df["sample_id"] = cell_df["fov_name"].map(fov_to_sample).astype(int)

    fov_rows: List[Dict[str, str | int]] = []
    for fov in fov_names:
        fov_rows.append(
            {
                "fov_name": fov,
                "stack_path": str(image_root / fov),
                "cell_mask_path": str(_resolve_mask_path(cell_mask_root, fov, "_cleaned_mask.tiff")),
                "sample_id": fov_to_sample.get(fov, 0),
            }
        )
    fov_df = pd.DataFrame(fov_rows)

    return cell_df, fov_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 2 cell/fov tables for MIBI folder layout.")
    parser.add_argument("--cell_table", type=Path, required=True, help="Input per-cell table")
    parser.add_argument("--image_root", type=Path, default=Path("image_data"))
    parser.add_argument("--cell_mask_root", type=Path, default=Path("segmentation/cleaned_segmasks"))
    parser.add_argument("--nuc_mask_root", type=Path, default=Path("segmentation/cellpose_output"))
    parser.add_argument("--fov_col", type=str, default="fov")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--x_col", type=str, default="X")
    parser.add_argument("--y_col", type=str, default="Y")
    parser.add_argument("--sample_id_mode", type=str, default="enumerate", choices=["enumerate", "from_col"])
    parser.add_argument("--sample_id_from_col", type=str, default=None)
    parser.add_argument("--keep_cell_id", action="store_true", help="Keep existing cell_id column if present")
    parser.add_argument(
        "--out_cell_table",
        type=Path,
        default=Path("manifests/cell_table_stage2.parquet"),
    )
    parser.add_argument(
        "--out_fov_table",
        type=Path,
        default=Path("manifests/fov_table_stage2.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_table(args.cell_table)
    cell_df, fov_df = build_tables(
        df,
        image_root=args.image_root,
        cell_mask_root=args.cell_mask_root,
        nuc_mask_root=args.nuc_mask_root,
        fov_col=args.fov_col,
        label_col=args.label_col,
        x_col=args.x_col,
        y_col=args.y_col,
        sample_id_mode=args.sample_id_mode,
        sample_id_from_col=args.sample_id_from_col,
        keep_cell_id=args.keep_cell_id,
    )

    args.out_cell_table.parent.mkdir(parents=True, exist_ok=True)
    args.out_fov_table.parent.mkdir(parents=True, exist_ok=True)

    if args.out_cell_table.suffix.lower() in {".parquet", ".pq"}:
        cell_df.to_parquet(args.out_cell_table, index=False)
    elif args.out_cell_table.suffix.lower() in {".tsv", ".tab"}:
        cell_df.to_csv(args.out_cell_table, sep="\t", index=False)
    else:
        cell_df.to_csv(args.out_cell_table, index=False)

    if args.out_fov_table.suffix.lower() in {".parquet", ".pq"}:
        fov_df.to_parquet(args.out_fov_table, index=False)
    elif args.out_fov_table.suffix.lower() in {".tsv", ".tab"}:
        fov_df.to_csv(args.out_fov_table, sep="\t", index=False)
    else:
        fov_df.to_csv(args.out_fov_table, index=False)

    print(f"Wrote cell table to {args.out_cell_table} ({len(cell_df)} rows)")
    print(f"Wrote fov table to {args.out_fov_table} ({len(fov_df)} rows)")


if __name__ == "__main__":
    main()
