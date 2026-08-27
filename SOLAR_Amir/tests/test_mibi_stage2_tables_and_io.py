from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage import io

from solar.datasets.build_stage2_tables_mibi import build_tables
from solar.datasets.compute_channel_stats import ChannelStatsConfig, compute_channel_stats
from solar.datasets.export_stage2_crops import ExportStage2Config, export_stage2_crops


def _write_tiff(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, arr.astype(np.float32), check_contrast=False)


def test_mibi_tables_and_directory_loading(tmp_path: Path) -> None:
    image_root = tmp_path / "image_data"
    cell_mask_root = tmp_path / "segmentation" / "cleaned_segmasks"
    nuc_mask_root = tmp_path / "segmentation" / "cellpose_output"

    fov = "FOV1"
    ch1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    ch2 = np.array([[10, 20], [30, 40]], dtype=np.float32)
    _write_tiff(image_root / fov / "ch1.tiff", ch1)
    _write_tiff(image_root / fov / "ch2.tiff", ch2)

    cell_mask = np.array([[1, 0], [1, 0]], dtype=np.uint16)
    nuc_mask = np.array([[1, 0], [0, 0]], dtype=np.uint16)
    _write_tiff(cell_mask_root / f"{fov}_cleaned_mask.tiff", cell_mask)
    _write_tiff(nuc_mask_root / f"{fov}_nuclear.tiff", nuc_mask)

    cell_table = pd.DataFrame(
        {
            "fov": [fov],
            "label": [1],
            "X": [0.5],
            "Y": [0.5],
        }
    )

    cell_df, fov_df = build_tables(
        cell_table,
        image_root=image_root,
        cell_mask_root=cell_mask_root,
        nuc_mask_root=nuc_mask_root,
        fov_col="fov",
        label_col="label",
        x_col="X",
        y_col="Y",
        sample_id_mode="enumerate",
        sample_id_from_col=None,
        keep_cell_id=False,
    )

    assert cell_df.loc[0, "stack_path"].endswith("image_data/FOV1")
    assert fov_df.loc[0, "stack_path"].endswith("image_data/FOV1")

    stats = compute_channel_stats(
        fov_df,
        ChannelStatsConfig(
            stack_column="stack_path",
            mask_column="cell_mask_path",
            channel_names=["ch1", "ch2"],
            mask_threshold=0.0,
        ),
    )
    assert stats["mean"][0] == 2.0
    assert stats["mean"][1] == 20.0

    cfg = ExportStage2Config(
        framesize=2,
        target_size=2,
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
        channel_names=["ch1", "ch2"],
        channel_ext="tiff",
        use_centroids=True,
        x_column="X",
        y_column="Y",
    )
    manifest = export_stage2_crops(cell_df, cfg)
    row = manifest.iloc[0]
    assert Path(row["stack128_path"]).exists()
    assert Path(row["stack256_path"]).exists()
    stack128 = torch.load(row["stack128_path"], map_location="cpu")
    assert stack128.shape == (2, 2, 2)
