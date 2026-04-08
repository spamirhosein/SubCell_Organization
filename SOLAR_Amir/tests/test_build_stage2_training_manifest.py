from __future__ import annotations

from pathlib import Path

import pandas as pd

from solar.datasets.build_stage2_training_manifest import build_stage2_training_manifest


def test_build_stage2_training_manifest(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "stack128_path": ["a.pt", "b.pt"],
            "sample_id": [0, 1],
            "cell_id": ["A__1", "B__2"],
            "cond_cell_a": [1.0, 2.0],
            "cond_sample_b": [0.1, -0.2],
            "mu_shape_0": [0.0, 0.1],
            "mu_shape_1": [0.2, 0.3],
            "logvar_shape_0": [0.0, 0.1],
            "logvar_shape_1": [0.2, 0.3],
            "junk": ["x", "y"],
        }
    )

    in_path = tmp_path / "manifest.csv"
    out_path = tmp_path / "out.parquet"
    df.to_csv(in_path, index=False)

    slim = build_stage2_training_manifest(
        manifest_in=in_path,
        out=out_path,
        stack_key="stack128_path",
        sample_id_key="sample_id",
        cell_id_key="cell_id",
        cond_cell_prefix="cond_cell_",
        cond_sample_prefix="cond_sample_",
        mu_shape_prefix="mu_shape_",
        logvar_shape_prefix="logvar_shape_",
        overwrite=True,
    )

    assert out_path.exists()
    assert "junk" not in slim.columns
    assert "stack128_path" in slim.columns
    assert "sample_id" in slim.columns
    assert "cell_id" in slim.columns
    assert "cond_cell_a" in slim.columns
    assert "cond_sample_b" in slim.columns

    mu_cols = [c for c in slim.columns if c.startswith("mu_shape_")]
    logvar_cols = [c for c in slim.columns if c.startswith("logvar_shape_")]
    assert len(mu_cols) == len(logvar_cols) == 2
