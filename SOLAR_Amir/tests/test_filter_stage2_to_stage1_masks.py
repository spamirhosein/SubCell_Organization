from __future__ import annotations

from pathlib import Path

import pandas as pd

from solar.datasets.filter_stage2_to_stage1_masks import filter_stage2_to_stage1_masks


def test_filter_stage2_to_stage1_masks(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    (mask_dir / "FOV1_cleaned_mask_cell_1.png").write_bytes(b"")
    (mask_dir / "FOV2_cleaned_mask_cell_2.png").write_bytes(b"")

    df = pd.DataFrame(
        {
            "fov_name": ["FOV1", "FOV1", "FOV2"],
            "cell_mask_id": [1, 2, 2],
            "extra": [10, 11, 12],
        }
    )

    in_path = tmp_path / "cell_table.csv"
    out_path = tmp_path / "filtered.parquet"
    missing_path = tmp_path / "missing.csv"
    df.to_csv(in_path, index=False)

    filtered = filter_stage2_to_stage1_masks(
        cell_table_in=in_path,
        mask_dir=mask_dir,
        cell_table_out=out_path,
        report_missing_out=missing_path,
        add_mask_path_column="combined_mask_path",
    )

    assert out_path.exists()
    assert missing_path.exists()
    assert len(filtered) == 2
    assert filtered["has_stage1_mask"].all()
    assert "combined_mask_path" in filtered.columns

    missing = pd.read_csv(missing_path)
    assert len(missing) == 1
    assert missing.loc[0, "cell_mask_id"] == 2
