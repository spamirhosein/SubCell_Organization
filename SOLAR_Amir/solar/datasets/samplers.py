from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[List[int]]):
    """Sampler that balances sample_ids across an epoch.

    The sampler oversamples minority sample_ids to match the largest group,
    interleaves them, and then yields index lists of length batch_size.
    This keeps batches approximately sample-balanced without discarding data.
    """

    def __init__(
        self,
        sample_ids: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(sample_ids) == 0:
            raise ValueError("sample_ids must be non-empty")
        self.sample_ids = list(int(s) for s in sample_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self._id_to_indices = self._group_indices(self.sample_ids)
        self._max_len = max(len(v) for v in self._id_to_indices.values())

    def __iter__(self) -> Iterator[List[int]]:  # type: ignore[override]
        expanded = self._expand_groups()
        order = list(expanded.keys())
        if self.shuffle and len(order) > 1:
            order = self._shuffle_list(order)
        interleaved: List[int] = []
        for i in range(self._max_len):
            for sid in order:
                interleaved.append(expanded[sid][i])
        for start in range(0, len(interleaved), self.batch_size):
            batch = interleaved[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:  # type: ignore[override]
        total = len(self._id_to_indices) * self._max_len
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)

    def _group_indices(self, sample_ids: Sequence[int]) -> Dict[int, List[int]]:
        grouped: Dict[int, List[int]] = defaultdict(list)
        for idx, sid in enumerate(sample_ids):
            grouped[sid].append(idx)
        return grouped

    def _shuffle_list(self, items: List[int]) -> List[int]:
        if not self.shuffle or len(items) <= 1:
            return items
        perm = torch.randperm(len(items), generator=self.generator).tolist()
        return [items[i] for i in perm]

    def _expand_groups(self) -> Dict[int, List[int]]:
        expanded: Dict[int, List[int]] = {}
        for sid, idxs in self._id_to_indices.items():
            pool = self._shuffle_list(list(idxs))
            repeat = math.ceil(self._max_len / len(pool))
            tiled = (pool * repeat)[: self._max_len]
            expanded[sid] = tiled
        return expanded
