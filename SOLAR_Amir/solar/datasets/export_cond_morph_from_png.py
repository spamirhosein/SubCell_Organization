from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from solar.datasets.solar_dataset import SolarDataset, SolarDatasetConfig
from solar.models.solar_shape_vae import SolarShapeVAE


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


def _resolve_mask_path(mask_dir: Path, template: str, fov: str, cell_mask_id: int) -> Path:
    return mask_dir / template.format(fov=fov, cell_mask_id=cell_mask_id)


def _build_cells(
    df: pd.DataFrame,
    mask_dir: Path,
    fov_column: str,
    cell_mask_id_column: str,
    mask_filename_template: str,
    sample_id_column: str | None,
) -> list[Dict[str, Any]]:
    cells: list[Dict[str, Any]] = []
    for _, row in df.iterrows():
        fov = str(row[fov_column])
        cell_id = int(row[cell_mask_id_column])
        mask_path = _resolve_mask_path(mask_dir, mask_filename_template, fov, cell_id)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        sample_id = int(row[sample_id_column]) if sample_id_column and sample_id_column in df.columns else 0
        cells.append(
            {
                "combined_mask": mask_path,
                "organelle_channels": {},
                "sample_id": sample_id,
            }
        )
    return cells


def _load_model(
    checkpoint: Path,
    latent_dim: int | None,
    group_N: int | None,
    input_size: int | None,
    base_filters: int | None,
    use_e2cnn: bool | None,
) -> SolarShapeVAE:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    cfg = ckpt.get("config", {})

    def _get(key: str, fallback: Any) -> Any:
        return cfg.get(key, fallback)

    model = SolarShapeVAE(
        latent_dim=latent_dim if latent_dim is not None else _get("latent_dim", 32),
        group_N=group_N if group_N is not None else _get("group_N", 8),
        input_size=input_size if input_size is not None else _get("input_size", 128),
        base_filters=base_filters if base_filters is not None else _get("base_filters", 32),
        nlayers=_get("nlayers", 3),
        kernel_size=_get("kernel_size", 5),
        use_e2cnn=use_e2cnn if use_e2cnn is not None else _get("use_e2cnn", True),
    )
    model.load_state_dict(state, strict=True)
    return model


def export_cond_morph(
    cell_table: str | Path | pd.DataFrame,
    mask_dir: Path,
    checkpoint: Path,
    out_table: Path,
    fov_column: str = "fov_name",
    cell_mask_id_column: str = "cell_mask_id",
    mask_filename_template: str = "{fov}_cleaned_mask_cell_{cell_mask_id}.png",
    low_res_size: int = 128,
    high_res_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str = "auto",
    limit: int | None = None,
    overwrite: bool = False,
    sample_id_column: str | None = "sample_id",
    latent_dim: int | None = None,
    group_N: int | None = None,
    base_filters: int | None = None,
    use_e2cnn: bool | None = None,
) -> pd.DataFrame:
    df = _load_table(cell_table)
    if fov_column not in df.columns:
        raise ValueError(f"Missing fov column '{fov_column}'")
    if cell_mask_id_column not in df.columns:
        raise ValueError(f"Missing cell id column '{cell_mask_id_column}'")

    if limit is not None:
        df = df.head(limit).copy()

    cells = _build_cells(
        df,
        mask_dir=mask_dir,
        fov_column=fov_column,
        cell_mask_id_column=cell_mask_id_column,
        mask_filename_template=mask_filename_template,
        sample_id_column=sample_id_column,
    )

    config = SolarDatasetConfig(
        channel_names=[],
        high_res_size=high_res_size,
        low_res_size=low_res_size,
        normalize_channels=False,
        mask_only=True,
        combined_mask_values={"background": 0, "cytoplasm": 1, "nucleus": 2},
    )
    dataset = SolarDataset(cells=cells, config=config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = _load_model(
        checkpoint=checkpoint,
        latent_dim=latent_dim,
        group_N=group_N,
        input_size=low_res_size,
        base_filters=base_filters,
        use_e2cnn=use_e2cnn,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            masks = batch["masks"].to(device)
            mu, logvar = model.encode(masks)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

    mu_arr = np.concatenate(mus, axis=0).astype(np.float32)
    logvar_arr = np.concatenate(logvars, axis=0).astype(np.float32)

    if not np.isfinite(mu_arr).all() or not np.isfinite(logvar_arr).all():
        raise ValueError("Non-finite values found in mu/logvar outputs")

    for k in range(mu_arr.shape[1]):
        df[f"mu_shape_{k}"] = mu_arr[:, k]
        df[f"logvar_shape_{k}"] = logvar_arr[:, k]

    _save_table(df, out_table, overwrite=overwrite)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage 1 cond_morph (mu/logvar) from combined-mask PNGs")
    parser.add_argument("--cell_table", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_table", type=Path, required=True)
    parser.add_argument("--fov_column", type=str, default="fov_name")
    parser.add_argument("--cell_mask_id_column", type=str, default="cell_mask_id")
    parser.add_argument(
        "--mask_filename_template",
        type=str,
        default="{fov}_cleaned_mask_cell_{cell_mask_id}.png",
    )
    parser.add_argument("--low_res_size", type=int, default=128)
    parser.add_argument("--high_res_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample_id_column", type=str, default="sample_id")
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--group_N", type=int, default=None)
    parser.add_argument("--base_filters", type=int, default=None)
    parser.add_argument("--use_e2cnn", action="store_true")
    parser.add_argument("--no_e2cnn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_e2cnn: bool | None
    if args.use_e2cnn and args.no_e2cnn:
        raise SystemExit("Use only one of --use_e2cnn or --no_e2cnn")
    if args.use_e2cnn:
        use_e2cnn = True
    elif args.no_e2cnn:
        use_e2cnn = False
    else:
        use_e2cnn = None

    export_cond_morph(
        cell_table=args.cell_table,
        mask_dir=args.mask_dir,
        checkpoint=args.checkpoint,
        out_table=args.out_table,
        fov_column=args.fov_column,
        cell_mask_id_column=args.cell_mask_id_column,
        mask_filename_template=args.mask_filename_template,
        low_res_size=args.low_res_size,
        high_res_size=args.high_res_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
        sample_id_column=args.sample_id_column,
        latent_dim=args.latent_dim,
        group_N=args.group_N,
        base_filters=args.base_filters,
        use_e2cnn=use_e2cnn,
    )


if __name__ == "__main__":
    main()
