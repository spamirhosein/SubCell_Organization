from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from solar.datasets.export_stage2_crops import ExportStage2Config, export_stage2_crops
from solar.datasets.solar_stacked_dataset import SolarStackedDatasetStage2, SolarStackedDatasetStage2Config


def _make_inputs(tmp_path: Path) -> pd.DataFrame:
    stack = torch.ones((2, 8, 8), dtype=torch.float32)
    cell_mask = torch.zeros((8, 8), dtype=torch.int32)
    cell_mask[2:6, 2:6] = 1
    nuc_mask = torch.zeros((8, 8), dtype=torch.int32)
    nuc_mask[3:5, 3:5] = 1

    stack_path = tmp_path / "fov_stack.pt"
    cell_mask_path = tmp_path / "fov_cell_mask.pt"
    nuc_mask_path = tmp_path / "fov_nuc_mask.pt"
    torch.save(stack, stack_path)
    torch.save(cell_mask, cell_mask_path)
    torch.save(nuc_mask, nuc_mask_path)

    return pd.DataFrame(
        {
            "fov_name": ["FOV1"],
            "stack_path": [stack_path],
            "cell_mask_path": [cell_mask_path],
            "nuclear_mask_path": [nuc_mask_path],
            "cell_mask_id": [1],
            "sample_id": [0],
            "cond_cell_0": [0.1],
            "cond_sample_0": [0.2],
            "mu_shape_0": [0.0],
            "logvar_shape_0": [0.0],
        }
    )


def test_flat_output_relative_paths_and_data_root(tmp_path: Path) -> None:
    df = _make_inputs(tmp_path)
    out128 = tmp_path / "stage2_crops_128"
    out256 = tmp_path / "stage2_crops_256"
    masks128 = tmp_path / "stage2_masks_128"
    masks256 = tmp_path / "stage2_masks_256"

    cfg = ExportStage2Config(
        framesize=8,
        target_size=4,
        out_dir_128=out128,
        out_dir_256=out256,
        mask_dir_128=masks128,
        mask_dir_256=masks256,
        save_masks=True,
        stack_column="stack_path",
        cell_mask_column="cell_mask_path",
        nuclear_mask_column="nuclear_mask_path",
        fov_column="fov_name",
        cell_mask_id_column="cell_mask_id",
        sample_id_column="sample_id",
        flat_output=True,
        relative_paths=True,
    )

    manifest = export_stage2_crops(df, cfg)
    row = manifest.iloc[0]
    assert row["stack128_path"] == "FOV1_1.pt"
    assert row["stack256_path"] == "FOV1_1.pt"
    assert row["mask128_path"] == "FOV1_1.pt"
    assert row["mask256_path"] == "FOV1_1.pt"

    assert (out128 / "FOV1_1.pt").exists()
    assert (out256 / "FOV1_1.pt").exists()

    dataset = SolarStackedDatasetStage2(
        manifest,
        SolarStackedDatasetStage2Config(
            channel_names=["c0", "c1"],
            mean=[0.0, 0.0],
            std=[1.0, 1.0],
            mask_key="mask128_path",
            data_root=out128,
            mask_root=masks128,
        ),
    )
    item = dataset[0]
    assert item["low_res"].shape == (2, 4, 4)
    assert item["mask"].shape == (4, 4)


def test_absolute_paths_still_work(tmp_path: Path) -> None:
    df = _make_inputs(tmp_path)
    cfg = ExportStage2Config(
        framesize=8,
        target_size=4,
        out_dir_128=tmp_path / "stage2_crops_128",
        out_dir_256=tmp_path / "stage2_crops_256",
        save_masks=False,
        stack_column="stack_path",
        cell_mask_column="cell_mask_path",
        nuclear_mask_column="nuclear_mask_path",
        fov_column="fov_name",
        cell_mask_id_column="cell_mask_id",
        sample_id_column="sample_id",
        flat_output=False,
        relative_paths=False,
    )
    manifest = export_stage2_crops(df, cfg)
    dataset = SolarStackedDatasetStage2(
        manifest,
        SolarStackedDatasetStage2Config(
            channel_names=["c0", "c1"],
            mean=[0.0, 0.0],
            std=[1.0, 1.0],
        ),
    )
    item = dataset[0]
    assert item["low_res"].shape == (2, 4, 4)
