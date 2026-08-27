from __future__ import annotations

import pandas as pd
import torch

from solar.datasets.solar_stacked_dataset import SolarStackedDatasetStage2, SolarStackedDatasetStage2Config


def test_stage2_dataset_loads_and_normalizes(tmp_path):
    stack = torch.ones((3, 128, 128)) * 2.0
    stack_path = tmp_path / "stack.pt"
    torch.save(stack, stack_path)

    df = pd.DataFrame(
        {
            "stack128_path": [stack_path],
            "sample_id": [0],
            "cell_id": ["fov__1"],
            "cond_cell_0": [0.1],
            "cond_sample_0": [0.2],
            "mu_shape_0": [0.0],
            "mu_shape_1": [0.1],
            "logvar_shape_0": [-0.1],
            "logvar_shape_1": [0.05],
        }
    )

    cfg = SolarStackedDatasetStage2Config(
        channel_names=["ch0", "ch1", "ch2"],
        mean=[1.0, 1.0, 1.0],
        std=[1.0, 1.0, 1.0],
    )
    dataset = SolarStackedDatasetStage2(df, cfg)
    item = dataset[0]

    assert item["low_res"].shape == (3, 128, 128)
    assert torch.isclose(item["low_res"].mean(), torch.tensor(1.0), atol=1e-4)
    assert item["cond_cell"].shape[0] == 1
    assert item["cond_sample"].shape[0] == 1
    assert item["mu_shape"].shape[0] == 2
    assert item["logvar_shape"].shape[0] == 2
    assert item["sample_id"] == 0
    assert item["cell_id"] == "fov__1"

