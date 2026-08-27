from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image


@dataclass
class SolarDatasetConfig:
    """Configuration holder for SolarDataset inputs and sizing."""

    channel_names: Sequence[str]
    high_res_size: int = 256
    low_res_size: int = 128
    pad_value: float = 0.0
    dtype: torch.dtype = torch.float32
    normalize_channels: bool = True
    mask_only: bool = False
    combined_mask_values: dict[str, int] | None = None  # e.g., {"background": 0, "cytoplasm": 1, "nucleus": 2}


class SolarDataset(Dataset):
    """Dataset that returns masks, low-res, and high-res organelle crops.

    Each item is a dict with:
    - masks: (2, low_res_size, low_res_size)
    - low_res: (C, low_res_size, low_res_size) (C=1 placeholder if mask_only)
    - high_res: (C, high_res_size, high_res_size) (C=1 placeholder if mask_only)
    - sample_id: int
    """

    def __init__(
        self,
        cells: Sequence[Mapping[str, Any]],
        config: SolarDatasetConfig,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    ) -> None:
        if not config.channel_names and not config.mask_only:
            raise ValueError("channel_names must be non-empty unless mask_only is True")
        self.cells = list(cells)
        self.config = config
        self.transform = transform
        self.sample_ids = [int(cell["sample_id"]) for cell in self.cells]
        self._channel_names = list(config.channel_names)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.cells)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        cell = self.cells[idx]
        if self.config.combined_mask_values and "combined_mask" in cell:
            label = self._to_chw(cell["combined_mask"])
            if label.dim() != 3 or label.shape[0] != 1:
                raise ValueError("combined_mask must be single-channel")
            label2d = label[0]
            nuc_val = self.config.combined_mask_values.get("nucleus", 2)
            cyt_val = self.config.combined_mask_values.get("cytoplasm", 1)
            nucleus = (label2d == nuc_val).float().unsqueeze(0)
            cell_mask = ((label2d == cyt_val) | (label2d == nuc_val)).float().unsqueeze(0)
        else:
            nucleus = self._to_chw(cell["nucleus_mask"])
            cell_mask = self._to_chw(cell["cell_mask"])

        masks_high = torch.cat([nucleus, cell_mask], dim=0)
        masks_high = self._crop_or_pad(masks_high, self.config.high_res_size)
        masks_low = self._downsample_or_pad(masks_high, self.config.low_res_size)

        if self.config.mask_only:
            high_res = torch.full(
                (1, self.config.high_res_size, self.config.high_res_size),
                fill_value=self.config.pad_value,
                dtype=self.config.dtype,
            )
            low_res = self._downsample_or_pad(high_res, self.config.low_res_size)
        else:
            high_res_channels = []
            for name in self._channel_names:
                arr = self._to_chw(cell["organelle_channels"][name])
                arr = self._crop_or_pad(arr, self.config.high_res_size)
                if self.config.normalize_channels:
                    arr = self._normalize(arr)
                high_res_channels.append(arr)
            high_res = torch.cat(high_res_channels, dim=0)
            low_res = self._downsample_or_pad(high_res, self.config.low_res_size)

        output = {
            "masks": masks_low,
            "low_res": low_res,
            "high_res": high_res,
            "sample_id": int(cell["sample_id"]),
        }

        if self.transform:
            output = self.transform(output)
        return output

    def _to_chw(self, arr: Any) -> torch.Tensor:
        if isinstance(arr, (str, Path)):
            tensor = self._load_png(arr)
        elif isinstance(arr, torch.Tensor):
            tensor = arr
        else:
            tensor = torch.as_tensor(arr)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3 and tensor.shape[0] != 1:
            pass
        elif tensor.dim() == 3:
            tensor = tensor
        else:
            raise ValueError(f"Expected 2D or 3D array for channel, got {tensor.shape}")
        tensor = tensor.to(dtype=self.config.dtype)
        return tensor

    def _load_png(self, path: Union[str, Path]) -> torch.Tensor:
        img = Image.open(path).convert("F")
        arr = np.array(img, dtype="float32")
        return torch.from_numpy(arr)

    def _crop_or_pad(self, tensor: torch.Tensor, target: int) -> torch.Tensor:
        _, h, w = tensor.shape
        # Center crop if larger.
        if h > target:
            top = (h - target) // 2
            tensor = tensor[:, top : top + target, :]
            h = target
        if w > target:
            left = (w - target) // 2
            tensor = tensor[:, :, left : left + target]
            w = target
        # Symmetric pad if smaller.
        if h < target or w < target:
            pad_top = (target - h) // 2
            pad_bottom = target - h - pad_top
            pad_left = (target - w) // 2
            pad_right = target - w - pad_left
            tensor = F.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                value=self.config.pad_value,
            )
        return tensor

    def _downsample_or_pad(self, tensor: torch.Tensor, target: int) -> torch.Tensor:
        _, h, w = tensor.shape
        if h == target and w == target:
            return tensor
        if h > target or w > target:
            tensor = tensor.unsqueeze(0)
            tensor = F.interpolate(tensor, size=(target, target), mode="area")
            tensor = tensor.squeeze(0)
        else:
            tensor = self._crop_or_pad(tensor, target)
        return tensor

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std().clamp_min(1e-6)
        return (tensor - mean) / std


def _make_synthetic_cells(num_cells: int, channel_names: Sequence[str], rng: torch.Generator) -> list[Dict[str, Any]]:
    cells: list[Dict[str, Any]] = []
    for i in range(num_cells):
        size = int(torch.randint(220, 260, (1,), generator=rng))
        nucleus = torch.zeros((size, size))
        cell_mask = torch.zeros((size, size))
        cx, cy = size // 2 + int(torch.randint(-8, 9, (1,), generator=rng)), size // 2 + int(torch.randint(-8, 9, (1,), generator=rng))
        r_nuc = int(torch.randint(10, 20, (1,), generator=rng))
        r_cell = int(torch.randint(22, 35, (1,), generator=rng))
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        nucleus[((yy - cy) ** 2 + (xx - cx) ** 2) < r_nuc ** 2] = 1.0
        cell_mask[((yy - cy) ** 2 + (xx - cx) ** 2) < r_cell ** 2] = 1.0
        channels: Dict[str, torch.Tensor] = {}
        for name in channel_names:
            noise = torch.randn((size, size), generator=rng) * 0.05
            blob = ((yy - cy - 5) ** 2 + (xx - cx - 5) ** 2) < int(torch.randint(8, 14, (1,), generator=rng)) ** 2
            channels[name] = noise + blob.float()
        cells.append(
            {
                "nucleus_mask": nucleus,
                "cell_mask": cell_mask,
                "organelle_channels": channels,
                "sample_id": int(i % 3),
            }
        )
    return cells


def visualize_batch(dataset: Dataset, n: int = 4) -> None:
    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    batch = next(iter(loader))
    masks = batch["masks"]
    low_res = batch["low_res"]
    high_res = batch["high_res"]
    num = masks.shape[0]
    fig, axes = plt.subplots(num, 3, figsize=(10, 3 * num))
    if num == 1:
        axes = axes.reshape(1, -1)
    for i in range(num):
        axes[i, 0].imshow(masks[i, 0].cpu(), cmap="Blues")
        axes[i, 0].set_title(f"Nucleus mask ({masks.shape[-1]}x{masks.shape[-1]})")
        axes[i, 1].imshow(low_res[i, 0].cpu(), cmap="magma")
        axes[i, 1].set_title(f"Low-res ch0 ({low_res.shape[-1]}x{low_res.shape[-1]})")
        axes[i, 2].imshow(high_res[i, 0].cpu(), cmap="magma")
        axes[i, 2].set_title(f"High-res ch0 ({high_res.shape[-1]}x{high_res.shape[-1]})")
        for j in range(3):
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SolarDataset visualization helper")
    parser.add_argument("--visualize_batch", action="store_true", help="Draw a small batch for sanity checking")
    parser.add_argument("--num_cells", type=int, default=4, help="Number of synthetic cells to render")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["TOM20", "ATP5A"],
        help="List of organelle channels to include",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic data")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.visualize_batch:
        raise SystemExit("Use --visualize_batch to render a batch.")
    rng = torch.Generator().manual_seed(args.seed)
    cells = _make_synthetic_cells(args.num_cells, args.channels, rng)
    config = SolarDatasetConfig(channel_names=args.channels)
    dataset = SolarDataset(cells=cells, config=config)
    visualize_batch(dataset)


if __name__ == "__main__":
    main()
