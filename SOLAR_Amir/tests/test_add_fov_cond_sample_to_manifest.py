from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from solar.datasets.add_fov_cond_sample_to_manifest import add_fov_cond_sample_to_manifest


def test_add_fov_cond_sample_to_manifest(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "fov_name": ["A", "A", "B", "B", "C"],
            "feat1": [1.0, 3.0, 2.0, 6.0, 5.0],
            "feat2": [10.0, 20.0, 10.0, 40.0, 30.0],
        }
    )

    in_path = tmp_path / "cell_table.csv"
    out_path = tmp_path / "out.parquet"
    df.to_csv(in_path, index=False)

    out = add_fov_cond_sample_to_manifest(
        cell_table=in_path,
        out_table=out_path,
        fov_column="fov_name",
        feature_cols=["feat1", "feat2"],
        prefix="cond_sample_",
        name_mode="by_col",
        ddof=0,
        eps=1e-6,
        overwrite=True,
    )

    assert out_path.exists()
    assert "cond_sample_feat1" in out.columns
    assert "cond_sample_feat2" in out.columns

    fov_medians = out.groupby("fov_name")[["feat1", "feat2"]].median()
    med_arr = fov_medians.to_numpy(dtype=float)
    means = med_arr.mean(axis=0)
    stds = med_arr.std(axis=0, ddof=0)
    expected = (med_arr - means) / (stds + 1e-6)

    merged = out.groupby("fov_name")[["cond_sample_feat1", "cond_sample_feat2"]].first()
    np.testing.assert_allclose(merged.to_numpy(), expected, rtol=1e-6, atol=1e-6)


def test_add_fov_cond_sample_coerce_numeric(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "fov_name": ["A", "A", "B"],
            "feat1": ["1.0", "2.0", "3.0"],
            "feat2": [1.0, 2.0, 3.0],
        }
    )

    in_path = tmp_path / "cell_table.csv"
    out_path = tmp_path / "out.parquet"
    df.to_csv(in_path, index=False)

    out = add_fov_cond_sample_to_manifest(
        cell_table=in_path,
        out_table=out_path,
        fov_column="fov_name",
        feature_cols=["feat1", "feat2"],
        overwrite=True,
        coerce_numeric=True,
    )

    assert out_path.exists()
    assert "cond_sample_feat1" in out.columns
