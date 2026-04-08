from __future__ import annotations

import pandas as pd
import pytest
import torch

from solar.datasets.compute_channel_stats import ChannelStatsConfig, compute_channel_stats


def test_channel_stats_masked_mean_std(tmp_path):
    stack = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 2.0], [2.0, 6.0]],
    ])
    mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8)
    stack_path = tmp_path / "stack.pt"
    mask_path = tmp_path / "mask.pt"
    torch.save(stack, stack_path)
    torch.save(mask, mask_path)

    df = pd.DataFrame({"stack_path": [stack_path], "cell_mask_path": [mask_path]})
    cfg = ChannelStatsConfig(stack_column="stack_path", mask_column="cell_mask_path", channel_names=["c0", "c1"])
    stats = compute_channel_stats(df, cfg)

    assert stats["channel_names"] == ["c0", "c1"]
    assert pytest.approx(stats["mean"][0], rel=1e-5) == 2.5
    assert pytest.approx(stats["mean"][1], rel=1e-5) == 4.0
    assert pytest.approx(stats["std"][0], rel=1e-5) == 1.5
    assert pytest.approx(stats["std"][1], rel=1e-5) == 2.0
