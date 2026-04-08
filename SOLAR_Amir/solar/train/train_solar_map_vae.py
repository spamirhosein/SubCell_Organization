from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import smooth_l1_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from solar.datasets.samplers import BalancedBatchSampler
from solar.datasets.solar_stacked_dataset import (
    SolarStackedDatasetStage2,
    SolarStackedDatasetStage2Config,
    load_channel_stats,
    _load_manifest,
)
from solar.models.solar_map_vae import SolarMapVAE, SolarMapVAEConfig


def kl_warmup_factor(global_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return float(min(1.0, global_step / float(warmup_steps)))


def beta_factor(global_step: int, max_beta: float, warmup_steps: int, beta_cycle_steps: int) -> float:
    max_beta = max(0.0, max_beta)
    if beta_cycle_steps and beta_cycle_steps > 0:
        cycle_pos = (global_step % beta_cycle_steps) / float(beta_cycle_steps)
        if cycle_pos <= 0.5:
            return max_beta * (cycle_pos / 0.5)
        return max_beta * ((1.0 - cycle_pos) / 0.5)
    warm = kl_warmup_factor(global_step, warmup_steps)
    return min(max_beta, warm * max_beta)


def kl_with_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        kl_per_dim = torch.maximum(kl_per_dim, torch.full_like(kl_per_dim, free_bits))
    return kl_per_dim.sum(dim=-1).mean()


def split_by_sample_id(sample_ids: Iterable[int], val_fraction: float, seed: int) -> Tuple[list[int], list[int]]:
    uniq = sorted(set(int(s) for s in sample_ids))
    if not uniq:
        return [], []
    g = random.Random(seed)
    g.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_fraction))
    val_ids = set(uniq[:n_val])
    train_idx: list[int] = []
    val_idx: list[int] = []
    for idx, sid in enumerate(sample_ids):
        if int(sid) in val_ids:
            val_idx.append(idx)
        else:
            train_idx.append(idx)
    return train_idx, val_idx


def split_by_cell_count(n: int, val_fraction: float, seed: int) -> Tuple[list[int], list[int]]:
    if n <= 0:
        return [], []
    idxs = list(range(n))
    g = random.Random(seed)
    g.shuffle(idxs)
    n_val = max(1, int(n * val_fraction))
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    return train_idx, val_idx


def _normalize_for_viz(tensor: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    t = tensor.clamp(min=vmin, max=vmax)
    return (t - vmin) / (vmax - vmin)


def _as_mask_4d(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    if mask.ndim == 4:
        return mask
    raise ValueError(f"Mask must have shape (B,H,W) or (B,1,H,W); got {tuple(mask.shape)}")


def masked_reconstruction_loss(
    recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, beta: float
) -> torch.Tensor:
    mask = _as_mask_4d(mask).to(dtype=recon.dtype)
    per_pixel = smooth_l1_loss(recon, x, reduction="none", beta=beta)
    masked = per_pixel * mask
    denom = mask.sum(dim=(1, 2, 3)) * x.shape[1]
    denom = denom.clamp_min(1.0)
    loss_i = masked.sum(dim=(1, 2, 3)) / denom
    return loss_i.mean()


def weighted_unmasked_mse(recon: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _as_mask_4d(mask).to(dtype=recon.dtype)
    mse_i = (recon - x).pow(2).mean(dim=(1, 2, 3))
    area_i = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    w_i = 1.0 / area_i
    w_i = w_i / w_i.mean().clamp_min(1e-6)
    return (w_i * mse_i).mean()


def log_recon_montages(
    model: SolarMapVAE,
    dataset,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    seed: int,
    n_cells: int,
    channel_indices: list[int] | None,
) -> None:
    if len(dataset) == 0 or n_cells <= 0:
        return
    g = torch.Generator().manual_seed(seed)
    n_take = min(n_cells, len(dataset))
    idxs = torch.randperm(len(dataset), generator=g)[:n_take]
    batch = [dataset[int(i)] for i in idxs]

    x = torch.stack([b["low_res"] for b in batch]).to(device)
    mu_shape = torch.stack([b["mu_shape"] for b in batch]).to(device)
    logvar_shape = torch.stack([b["logvar_shape"] for b in batch]).to(device)
    cond_cell = torch.stack([b["cond_cell"] for b in batch]).to(device)
    cond_sample = torch.stack([b["cond_sample"] for b in batch]).to(device)
    mask = None
    if "mask" in batch[0]:
        mask = torch.stack([b["mask"] for b in batch]).to(device)
        mask = _as_mask_4d(mask)

    model.eval()
    with torch.no_grad():
        recon, _, _, _, _ = model(x, mu_shape, logvar_shape, cond_cell, cond_sample)

    if channel_indices is None:
        channel_indices = list(range(x.shape[1]))

    for i, idx in enumerate(idxs.tolist()):
        cell_id = batch[i].get("cell_id", f"cell_{idx}")
        x_i = x[i, channel_indices]
        r_i = recon[i, channel_indices]
        if mask is not None:
            mi = mask[i, 0]
            x_i = x_i * mi
            r_i = r_i * mi
            err = (x_i - r_i).abs() * mi
        else:
            err = (x_i - r_i).abs()

        x_viz = _normalize_for_viz(x_i, vmin=-3.0, vmax=3.0)
        r_viz = _normalize_for_viz(r_i, vmin=-3.0, vmax=3.0)
        e_viz = _normalize_for_viz(err, vmin=0.0, vmax=2.0)

        tiles = torch.cat([x_viz, r_viz, e_viz], dim=0).unsqueeze(1)
        grid = make_grid(tiles, nrow=len(channel_indices), padding=2)
        writer.add_image(f"recon_cells/cell_{i:02d}_{cell_id}", grid, global_step, dataformats="CHW")


def save_checkpoint(model: SolarMapVAE, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(model.config)}, path)


def resolve_checkpoint_dir(checkpoint_arg: Path, run_name: str) -> tuple[Path, Path]:
    if checkpoint_arg.suffix:
        print(
            f"Warning: --checkpoint should be a directory; using parent {checkpoint_arg.parent} for run '{run_name}'."
        )
        ckpt_root = checkpoint_arg.parent
    else:
        ckpt_root = checkpoint_arg
    ckpt_dir = ckpt_root / run_name
    return ckpt_root, ckpt_dir


def build_dataset(args: argparse.Namespace) -> SolarStackedDatasetStage2:
    stats = load_channel_stats(args.channel_stats)
    channel_names = args.channel_names if args.channel_names else list(stats["channel_names"])
    cfg = SolarStackedDatasetStage2Config(
        channel_names=channel_names,
        mean=stats["mean"],
        std=stats["std"],
        stack_key=args.stack_key,
        mask_key=args.mask_key,
        zero_background=args.zero_background_input,
        data_root=args.data_root,
        mask_root=args.mask_root,
        sample_id_key=args.sample_id_key,
        cell_id_key=args.cell_id_key,
        cond_cell_prefix=args.cond_cell_prefix,
        cond_sample_prefix=args.cond_sample_prefix,
        mu_shape_prefix=args.mu_shape_prefix,
        logvar_shape_prefix=args.logvar_shape_prefix,
    )
    return SolarStackedDatasetStage2(args.manifest, cfg)


def make_synthetic_dataset(args: argparse.Namespace) -> SolarStackedDatasetStage2:
    g = torch.Generator().manual_seed(args.seed)
    rows = []
    channel_names = args.channel_names or [f"ch{i}" for i in range(args.synthetic_channels)]
    mean = [0.0 for _ in channel_names]
    std = [1.0 for _ in channel_names]
    tmp_dir = Path(args.synthetic_tmp)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.synthetic_cells):
        stack = torch.randn((len(channel_names), args.input_size, args.input_size), generator=g)
        stack_path = tmp_dir / f"stack_{i}.pt"
        torch.save(stack, stack_path)
        row = {
            "stack128_path": stack_path,
            "sample_id": i % max(1, args.synthetic_samples),
            "cell_id": f"syn_{i}",
        }
        for j in range(args.synthetic_cond_cell):
            row[f"cond_cell_{j}"] = float(torch.randn((), generator=g))
        for j in range(args.synthetic_cond_sample):
            row[f"cond_sample_{j}"] = float(torch.randn((), generator=g))
        for j in range(args.synthetic_shape_dim):
            row[f"mu_shape_{j}"] = float(torch.randn((), generator=g))
            row[f"logvar_shape_{j}"] = float(torch.randn((), generator=g))
        rows.append(row)
    df = _load_manifest(pd.DataFrame(rows))
    cfg = SolarStackedDatasetStage2Config(
        channel_names=channel_names,
        mean=mean,
        std=std,
    )
    return SolarStackedDatasetStage2(df, cfg)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SolarMapVAE (Stage 2)")
    parser.add_argument("--manifest", type=Path, help="Parquet/CSV manifest with Stage 2 crops")
    parser.add_argument("--channel_stats", type=Path, help="JSON or .pt with channel_names/mean/std")
    parser.add_argument("--channel_names", nargs="+", default=None, help="Override channel names order")
    parser.add_argument("--stack_key", type=str, default="stack128_path")
    parser.add_argument("--mask_key", type=str, default=None)
    parser.add_argument("--data_root", type=Path, default=None, help="Root to prepend to relative paths")
    parser.add_argument("--mask_root", type=Path, default=None, help="Root to prepend to relative mask paths")
    parser.add_argument("--sample_id_key", type=str, default="sample_id")
    parser.add_argument("--cell_id_key", type=str, default="cell_id")
    parser.add_argument("--cond_cell_prefix", type=str, default="cond_cell_")
    parser.add_argument("--cond_sample_prefix", type=str, default="cond_sample_")
    parser.add_argument("--mu_shape_prefix", type=str, default="mu_shape_")
    parser.add_argument("--logvar_shape_prefix", type=str, default="logvar_shape_")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--kl_warmup_steps", type=int, default=2000)
    parser.add_argument("--max_beta", type=float, default=1.0)
    parser.add_argument("--beta_cycle_steps", type=int, default=0)
    parser.add_argument("--kl_free_bits", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="sample",
        choices=["sample", "cell"],
        help="Split by sample_id (default) or random per-cell. Cell split may mix FOVs across train/val.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_dir", type=Path, default=Path("runs/solarmap_vae"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/solar_map_vae.pt"))
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for smoke testing")
    parser.add_argument("--synthetic_cells", type=int, default=64)
    parser.add_argument("--synthetic_samples", type=int, default=4)
    parser.add_argument("--synthetic_channels", type=int, default=3)
    parser.add_argument("--synthetic_cond_cell", type=int, default=2)
    parser.add_argument("--synthetic_cond_sample", type=int, default=3)
    parser.add_argument("--synthetic_shape_dim", type=int, default=8)
    parser.add_argument("--synthetic_tmp", type=str, default="/tmp/solar_stage2")
    parser.add_argument("--viz_n_cells", type=int, default=4, help="Number of validation cells to log")
    parser.add_argument("--viz_channels", nargs="+", default=None, help="Optional channel names to log")
    parser.add_argument(
        "--use_masked_rec",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use masked reconstruction loss (requires mask_key)",
    )
    parser.add_argument(
        "--zero_background_input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero out background pixels in the input using the mask",
    )
    parser.add_argument(
        "--weighted_unmasked_rec",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use unmasked MSE weighted by inverse mask area (requires mask_key)",
    )
    parser.add_argument("--rec_huber_beta", type=float, default=0.5)
    return parser.parse_args()


def train() -> None:
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_masked_rec and args.mask_key is None:
        raise SystemExit("--use_masked_rec requires --mask_key (e.g., mask128_path)")
    if args.weighted_unmasked_rec and args.mask_key is None:
        raise SystemExit("--weighted_unmasked_rec requires --mask_key (e.g., mask128_path)")

    if args.synthetic:
        dataset = make_synthetic_dataset(args)
    else:
        dataset = build_dataset(args)

    if len(dataset) == 0:
        raise SystemExit("Dataset is empty")

    if args.split_mode == "sample":
        train_idx, val_idx = split_by_sample_id(
            dataset.sample_ids, val_fraction=args.val_fraction, seed=args.seed
        )
    else:
        train_idx, val_idx = split_by_cell_count(len(dataset), args.val_fraction, args.seed)
    print(f"Split mode: {args.split_mode} | train={len(train_idx)} val={len(val_idx)}")
    if args.split_mode == "sample":
        train_sids = {int(dataset.sample_ids[i]) for i in train_idx}
        val_sids = {int(dataset.sample_ids[i]) for i in val_idx}
        print(f"Unique sample_ids: train={len(train_sids)} val={len(val_sids)}")
    if not train_idx:
        raise SystemExit("No training samples after split; adjust val_fraction")
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_sampler = BalancedBatchSampler(
        sample_ids=[dataset.sample_ids[i] for i in train_idx],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cond_morph_dim = dataset.shape_latent_dim
    cond_cell_dim = dataset.cond_cell_dim
    cond_sample_dim = dataset.cond_sample_dim
    config = SolarMapVAEConfig(
        num_channels=len(dataset.channel_names),
        input_size=args.input_size,
        latent_dim=args.latent_dim,
        cond_morph_dim=cond_morph_dim,
        cond_cell_dim=cond_cell_dim,
        cond_sample_dim=cond_sample_dim,
        base_filters=args.base_filters,
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
    )
    model = SolarMapVAE(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    mask_tag = "masked" if args.use_masked_rec else "unmasked"
    run_dir = args.log_dir / (
        f"ld{args.latent_dim}_bf{args.base_filters}_nb{args.num_blocks}_wu{args.kl_warmup_steps}_lr{args.lr:g}_mb{args.max_beta:g}_{mask_tag}_hb{args.rec_huber_beta:g}"
    )
    writer = SummaryWriter(log_dir=run_dir)
    run_name = run_dir.name
    ckpt_root, ckpt_dir = resolve_checkpoint_dir(Path(args.checkpoint), run_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint root: {ckpt_root} | run_name: {run_name}")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_kl = 0.0
        for batch in train_loader:
            x = batch["low_res"].to(device)
            mu_shape = batch["mu_shape"].to(device)
            logvar_shape = batch["logvar_shape"].to(device)
            cond_cell = batch["cond_cell"].to(device)
            cond_sample = batch["cond_sample"].to(device)

            recon, mu, logvar, _, _ = model(x, mu_shape, logvar_shape, cond_cell, cond_sample)
            if args.use_masked_rec:
                if "mask" not in batch:
                    raise KeyError("Masked reconstruction requires batch['mask']; check mask_key in dataset")
                mask = batch["mask"].to(device)
                if mask.sum().item() == 0:
                    print("Warning: mask sum is zero for a training batch")
                writer.add_scalar("train/mask_frac", mask.float().mean().item(), global_step)
                rec = masked_reconstruction_loss(recon, x, mask, beta=args.rec_huber_beta)
            elif args.weighted_unmasked_rec:
                if "mask" not in batch:
                    raise KeyError("Weighted unmasked reconstruction requires batch['mask']; check mask_key in dataset")
                mask = batch["mask"].to(device)
                if mask.sum().item() == 0:
                    print("Warning: mask sum is zero for a training batch")
                area_i = _as_mask_4d(mask).sum(dim=(1, 2, 3)).clamp_min(1.0)
                w_i = (1.0 / area_i)
                w_i = w_i / w_i.mean().clamp_min(1e-6)
                writer.add_scalar("train/mask_area_mean", area_i.mean().item(), global_step)
                writer.add_scalar("train/weight_mean", w_i.mean().item(), global_step)
                rec = weighted_unmasked_mse(recon, x, mask)
            else:
                rec = model.reconstruction_loss(recon, x)
            kl = kl_with_free_bits(mu, logvar, free_bits=args.kl_free_bits)
            beta = beta_factor(global_step, args.max_beta, args.kl_warmup_steps, args.beta_cycle_steps)
            loss = rec + beta * kl

            opt.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()

            epoch_loss += loss.item()
            epoch_rec += rec.item()
            epoch_kl += kl.item()
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            writer.add_scalar("train/loss_rec", rec.item(), global_step)
            writer.add_scalar("train/loss_kl", kl.item(), global_step)
            writer.add_scalar("train/beta", beta, global_step)
            global_step += 1

        num_batches = len(train_loader)
        print(
            f"Epoch {epoch+1}/{args.epochs} | Loss {epoch_loss/num_batches:.4f} | Rec {epoch_rec/num_batches:.4f} | KL {epoch_kl/num_batches:.4f}"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_rec = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["low_res"].to(device)
                mu_shape = batch["mu_shape"].to(device)
                logvar_shape = batch["logvar_shape"].to(device)
                cond_cell = batch["cond_cell"].to(device)
                cond_sample = batch["cond_sample"].to(device)
                recon, mu, logvar, _, _ = model(x, mu_shape, logvar_shape, cond_cell, cond_sample)
                if args.use_masked_rec:
                    if "mask" not in batch:
                        raise KeyError("Masked reconstruction requires batch['mask']; check mask_key in dataset")
                    mask = batch["mask"].to(device)
                    if mask.sum().item() == 0:
                        print("Warning: mask sum is zero for a validation batch")
                    rec = masked_reconstruction_loss(recon, x, mask, beta=args.rec_huber_beta)
                elif args.weighted_unmasked_rec:
                    if "mask" not in batch:
                        raise KeyError(
                            "Weighted unmasked reconstruction requires batch['mask']; check mask_key in dataset"
                        )
                    mask = batch["mask"].to(device)
                    if mask.sum().item() == 0:
                        print("Warning: mask sum is zero for a validation batch")
                    rec = weighted_unmasked_mse(recon, x, mask)
                else:
                    rec = model.reconstruction_loss(recon, x)
                kl = kl_with_free_bits(mu, logvar, free_bits=0.0)
                loss = rec + args.max_beta * kl
                val_loss += loss.item()
                val_rec += rec.item()
                val_kl += kl.item()
        num_val_batches = max(1, len(val_loader))
        writer.add_scalar("val/loss_total", val_loss / num_val_batches, global_step)
        writer.add_scalar("val/loss_rec", val_rec / num_val_batches, global_step)
        writer.add_scalar("val/loss_kl", val_kl / num_val_batches, global_step)
        channel_indices = None
        if args.viz_channels:
            name_to_idx = {name: i for i, name in enumerate(dataset.channel_names)}
            missing = [name for name in args.viz_channels if name not in name_to_idx]
            if missing:
                raise ValueError(f"Requested viz_channels not found in dataset.channel_names: {missing}")
            channel_indices = [name_to_idx[name] for name in args.viz_channels]

        log_recon_montages(
            model,
            val_ds,
            device,
            writer,
            global_step,
            seed=args.seed,
            n_cells=args.viz_n_cells,
            channel_indices=channel_indices,
        )

        if args.save_every and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"{run_name}_epoch{epoch+1:04d}.pt"
            save_checkpoint(model, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    save_path = ckpt_dir / f"{run_name}_final.pt"
    save_checkpoint(model, save_path)
    print(f"Saved checkpoint to {save_path}")
    writer.close()


if __name__ == "__main__":
    train()

