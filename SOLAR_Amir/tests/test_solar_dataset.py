from __future__ import annotations

from collections import Counter
from typing import Dict, List

import torch

from solar.datasets.samplers import BalancedBatchSampler
from solar.datasets.solar_dataset import SolarDataset, SolarDatasetConfig


def _make_cell(size: int, channel_names: List[str], sample_id: int, rng: torch.Generator) -> Dict:
    nucleus = torch.zeros((size, size))
    cell_mask = torch.zeros((size, size))
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    nucleus[((yy - size // 2) ** 2 + (xx - size // 2) ** 2) < (size // 6) ** 2] = 1.0
    cell_mask[((yy - size // 2) ** 2 + (xx - size // 2) ** 2) < (size // 4) ** 2] = 1.0
    channels: Dict[str, torch.Tensor] = {}
    for name in channel_names:
        blob = ((yy - size // 2 - 5) ** 2 + (xx - size // 2 - 5) ** 2) < (size // 8) ** 2
        noise = torch.randn((size, size), generator=rng) * 0.02
        channels[name] = blob.float() + noise
    return {
        "nucleus_mask": nucleus,
        "cell_mask": cell_mask,
        "organelle_channels": channels,
        "sample_id": sample_id,
    }


def _make_dataset(num_cells: int = 6) -> SolarDataset:
    rng = torch.Generator().manual_seed(123)
    channel_names = ["TOM20", "ATP5A", "ER"]
    cells = [
        _make_cell(size=int(torch.randint(120, 180, (1,), generator=rng)), channel_names=channel_names, sample_id=i % 3, rng=rng)
        for i in range(num_cells)
    ]
    config = SolarDatasetConfig(channel_names=channel_names)
    return SolarDataset(cells=cells, config=config)


def test_shapes_and_types():
    dataset = _make_dataset(num_cells=4)
    item = dataset[0]
    assert set(item.keys()) == {"masks", "low_res", "high_res", "sample_id"}
    assert item["masks"].shape == (2, 128, 128)
    assert item["low_res"].shape[1:] == (128, 128)
    assert item["high_res"].shape[1:] == (256, 256)
    assert item["masks"].dtype == torch.float32
    assert item["low_res"].dtype == torch.float32
    assert isinstance(item["sample_id"], int)


def test_balanced_batch_sampler_balances_sample_ids():
    dataset = _make_dataset(num_cells=9)
    sampler = BalancedBatchSampler(sample_ids=dataset.sample_ids, batch_size=4, shuffle=False, drop_last=False)
    batches = list(iter(sampler))
    flattened_ids = [dataset.sample_ids[idx] for batch in batches for idx in batch]
    counts = Counter(flattened_ids)
    assert len(counts) == 3
    assert max(counts.values()) == min(counts.values())


def test_sampler_batch_lengths():
    dataset = _make_dataset(num_cells=5)
    sampler = BalancedBatchSampler(sample_ids=dataset.sample_ids, batch_size=3, shuffle=False, drop_last=True)
    batches = list(iter(sampler))
    assert all(len(batch) == 3 for batch in batches)
    assert len(batches) == len(sampler)
