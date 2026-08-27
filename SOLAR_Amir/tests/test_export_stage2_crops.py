from __future__ import annotations

import pandas as pd
import torch

from solar.datasets.export_stage2_crops import ExportStage2Config, export_stage2_crops


def test_export_stage2_crops_saves_outputs(tmp_path):
    stack = torch.ones((2, 16, 16), dtype=torch.float32)
    cell_mask = torch.zeros((16, 16), dtype=torch.int32)
    cell_mask[6:10, 6:10] = 1
    nuc_mask = torch.zeros((16, 16), dtype=torch.int32)
    nuc_mask[7:9, 7:9] = 1

    stack_path = tmp_path / "fov_stack.pt"
    cell_mask_path = tmp_path / "fov_cell_mask.pt"
    nuc_mask_path = tmp_path / "fov_nuc_mask.pt"
    torch.save(stack, stack_path)
    torch.save(cell_mask, cell_mask_path)
    torch.save(nuc_mask, nuc_mask_path)

    df = pd.DataFrame(
        {
            "fov_name": ["fovA"],
            "stack_path": [stack_path],
            "cell_mask_path": [cell_mask_path],
            "nuclear_mask_path": [nuc_mask_path],
            "cell_mask_id": [1],
            "sample_id": [0],
        }
    )

    cfg = ExportStage2Config(
        framesize=16,
        target_size=8,
        out_dir_128=tmp_path / "stage_crops_128",
        out_dir_256=tmp_path / "stage_crops_256",
        mask_dir_128=tmp_path / "stage_masks_128",
        mask_dir_256=tmp_path / "stage_masks_256",
        save_masks=True,
        stack_column="stack_path",
        cell_mask_column="cell_mask_path",
        nuclear_mask_column="nuclear_mask_path",
        fov_column="fov_name",
        cell_mask_id_column="cell_mask_id",
        sample_id_column="sample_id",
    )

    manifest = export_stage2_crops(df, cfg)
    assert len(manifest) == 1
    row = manifest.iloc[0]

    stack128 = torch.load(row["stack128_path"], map_location="cpu")
    stack256 = torch.load(row["stack256_path"], map_location="cpu")
    mask128 = torch.load(row["mask128_path"], map_location="cpu")
    mask256 = torch.load(row["mask256_path"], map_location="cpu")

    assert stack128.shape == (2, 8, 8)
    assert stack256.shape == (2, 16, 16)
    assert mask128.shape == (8, 8)
    assert mask256.shape == (16, 16)
    assert stack128.dtype == torch.float32
    assert stack256.dtype == torch.float32
    assert mask256.sum() > 0
    assert row["cell_id"] == "fovA__1"
