from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
import pytest

from solar.datasets.add_lineage_cond_to_manifest import add_lineage_cond_to_manifest


def test_add_lineage_cond_to_manifest_cell_id(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "cell_id": ["FOV1__1", "FOV1__2"],
            "fov_name": ["FOV1", "FOV1"],
            "cell_mask_id": [1, 2],
        }
    )
    source = pd.DataFrame(
        {
            "cell_id": ["FOV1__1", "FOV1__2"],
            "CD3": [1.0, 3.0],
            "CD20": [2.0, 6.0],
        }
    )

    manifest_path = tmp_path / "manifest.csv"
    source_path = tmp_path / "source.csv"
    out_path = tmp_path / "out.csv"
    stats_path = tmp_path / "stats.json"

    manifest.to_csv(manifest_path, index=False)
    source.to_csv(source_path, index=False)

    df = add_lineage_cond_to_manifest(
        manifest_in=manifest_path,
        source_in=source_path,
        out=out_path,
        markers="CD3,CD20",
        join_on="cell_id",
        normalize="zscore",
        stats_out=stats_path,
    )

    assert out_path.exists()
    assert stats_path.exists()
    assert "cond_cell_CD3" in df.columns
    assert "cond_cell_CD20" in df.columns

    stats = json.loads(stats_path.read_text())
    assert "cond_cell_CD3" in stats
    assert "cond_cell_CD20" in stats


def test_add_lineage_cond_missing_marker(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "cell_id": ["FOV1__1"],
            "fov_name": ["FOV1"],
            "cell_mask_id": [1],
        }
    )
    source = pd.DataFrame({"cell_id": ["FOV1__1"], "CD3": [1.0]})

    manifest_path = tmp_path / "manifest.csv"
    source_path = tmp_path / "source.csv"

    manifest.to_csv(manifest_path, index=False)
    source.to_csv(source_path, index=False)

    with pytest.raises(ValueError, match="Markers missing"):
        add_lineage_cond_to_manifest(
            manifest_in=manifest_path,
            source_in=source_path,
            out=tmp_path / "out.csv",
            markers="CD3,CD20",
            join_on="cell_id",
        )
