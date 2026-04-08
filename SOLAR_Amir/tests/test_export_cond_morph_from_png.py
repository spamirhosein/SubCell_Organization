from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from solar.datasets.export_cond_morph_from_png import export_cond_morph
from solar.models.solar_shape_vae import SolarShapeVAE


def _write_label_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.save(path)


def _make_checkpoint(path: Path, input_size: int = 8) -> None:
    model = SolarShapeVAE(
        latent_dim=2,
        group_N=1,
        input_size=input_size,
        base_filters=4,
        nlayers=1,
        kernel_size=3,
        use_e2cnn=False,
    )
    ckpt = {"model_state": model.state_dict(), "config": model.config.__dict__}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def test_export_cond_morph_from_png(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    fov = "FOV1"

    label1 = np.zeros((8, 8), dtype=np.uint8)
    label1[2:6, 2:6] = 1
    label1[3:5, 3:5] = 2
    label2 = np.zeros((8, 8), dtype=np.uint8)
    label2[1:4, 1:4] = 1
    label2[2:3, 2:3] = 2

    _write_label_png(mask_dir / f"{fov}_cleaned_mask_cell_1.png", label1)
    _write_label_png(mask_dir / f"{fov}_cleaned_mask_cell_2.png", label2)

    table = pd.DataFrame(
        {
            "fov_name": [fov, fov],
            "cell_mask_id": [1, 2],
        }
    )

    ckpt_path = tmp_path / "shape.pt"
    _make_checkpoint(ckpt_path, input_size=8)

    out_path = tmp_path / "out.parquet"
    df = export_cond_morph(
        cell_table=table,
        mask_dir=mask_dir,
        checkpoint=ckpt_path,
        out_table=out_path,
        low_res_size=8,
        high_res_size=8,
        batch_size=2,
        num_workers=0,
        device="cpu",
        overwrite=True,
        use_e2cnn=False,
    )

    assert out_path.exists()
    assert "mu_shape_0" in df.columns
    assert "logvar_shape_0" in df.columns
    assert np.isfinite(df["mu_shape_0"]).all()
    assert np.isfinite(df["logvar_shape_0"]).all()


def test_export_cond_morph_missing_file(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    table = pd.DataFrame({"fov_name": ["FOVX"], "cell_mask_id": [99]})
    ckpt_path = tmp_path / "shape.pt"
    _make_checkpoint(ckpt_path, input_size=8)

    with pytest.raises(FileNotFoundError) as excinfo:
        export_cond_morph(
            cell_table=table,
            mask_dir=mask_dir,
            checkpoint=ckpt_path,
            out_table=tmp_path / "out.parquet",
            low_res_size=8,
            high_res_size=8,
            batch_size=1,
            num_workers=0,
            device="cpu",
            overwrite=True,
            use_e2cnn=False,
        )
    assert "FOVX_cleaned_mask_cell_99.png" in str(excinfo.value)
