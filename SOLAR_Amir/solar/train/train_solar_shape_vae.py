from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from solar.datasets.samplers import BalancedBatchSampler
from solar.datasets.solar_dataset import SolarDataset, SolarDatasetConfig, _make_synthetic_cells
from solar.models.solar_shape_vae import SolarShapeVAE


def kl_warmup_factor(global_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return float(min(1.0, global_step / float(warmup_steps)))


def beta_factor(global_step: int, args: argparse.Namespace) -> float:
    """Compute beta schedule with optional warmup cap and triangular cycles."""
    max_beta = max(0.0, args.max_beta)
    if args.beta_cycle_steps and args.beta_cycle_steps > 0:
        # Triangular 0 -> max_beta -> 0 over beta_cycle_steps.
        cycle_pos = (global_step % args.beta_cycle_steps) / float(args.beta_cycle_steps)
        if cycle_pos <= 0.5:
            return max_beta * (cycle_pos / 0.5)
        return max_beta * ((1.0 - cycle_pos) / 0.5)
    warm = kl_warmup_factor(global_step, args.kl_warmup_steps)
    return min(max_beta, warm * max_beta)


def kl_with_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
    """Standard Gaussian KL with optional per-dim free bits, summed per sample then averaged."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
    if free_bits > 0.0:
        kl_per_dim = torch.maximum(kl_per_dim, torch.full_like(kl_per_dim, free_bits))
    return kl_per_dim.sum(dim=1).mean()


def save_checkpoint(model: SolarShapeVAE, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": model.config.__dict__}, path)


def maybe_save_embeddings(model: SolarShapeVAE, loader: DataLoader, limit: int, path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        import pandas as pd
    except Exception:
        print("pandas not available; skipping embedding export")
        return
    model.eval()
    all_rows = []
    with torch.no_grad():
        count = 0
        for batch in loader:
            masks = batch["masks"].to(next(model.parameters()).device)
            mu, logvar = model.encode(masks)
            for i in range(mu.shape[0]):
                all_rows.append(mu[i].cpu().numpy())
                count += 1
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break
    if not all_rows:
        return
    import numpy as np

    df = pd.DataFrame(np.stack(all_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"Saved embeddings to {path}")


def make_run_dir(base: Path, args: argparse.Namespace) -> Path:
    """Construct a descriptive run directory based on key hyperparameters."""
    run_name = (
        f"ld{args.latent_dim}_bf{args.base_filters}_wu{args.kl_warmup_steps}"
        f"_lr{args.lr:g}_mb{args.max_beta:g}_fb{args.kl_free_bits:g}_e2{int(not args.no_e2cnn)}"
    )
    return base / run_name


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SolarShapeVAE (Stage 1)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_warmup_steps", type=int, default=2000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--group_N", type=int, default=8)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--no_e2cnn", action="store_true", help="Use plain Conv2d encoder (diagnostic only)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic masks for smoke testing")
    parser.add_argument("--num_cells", type=int, default=256, help="Number of synthetic cells when --synthetic")
    parser.add_argument("--channels", nargs="+", default=["TOM20"], help="Dummy channels for synthetic data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mask_manifest",
        type=Path,
        default=None,
        help=(
            "TSV with nucleus_mask_path [tab] cell_mask_path [tab] sample_id. "
            "If only one path is given, it is used for both nucleus and cell; sample_id defaults to 0."
        ),
    )
    parser.add_argument(
        "--mask_root",
        type=Path,
        default=None,
        help="Optional root to prepend to relative paths in mask_manifest",
    )
    parser.add_argument("--high_res_size", type=int, default=256, help="High-res mask canvas size")
    parser.add_argument("--low_res_size", type=int, default=128, help="Low-res mask canvas size (model input)")
    parser.add_argument("--combined_mask", action="store_true", help="Manifest paths are single label maps (0/1/2)")
    parser.add_argument("--val_background", type=int, default=0, help="Value for background in combined mask")
    parser.add_argument("--val_cytoplasm", type=int, default=1, help="Value for cytoplasm in combined mask")
    parser.add_argument("--val_nucleus", type=int, default=2, help="Value for nucleus in combined mask")
    parser.add_argument(
        "--class_weights",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Class weights for background, cytoplasm, nucleus (used in multi-class recon loss)",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/solar_shape_vae.pt"))
    parser.add_argument(
        "--embeddings_out", type=Path, default=None, help="Optional parquet path to dump mu embeddings"
    )
    parser.add_argument("--embed_limit", type=int, default=256, help="Max rows to dump to parquet")
    parser.add_argument("--log_dir", type=Path, default=Path("runs/solarshape_vae"), help="TensorBoard log directory")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Max global grad norm (0 to disable)")
    parser.add_argument("--max_beta", type=float, default=1.0, help="Cap on KL weight beta")
    parser.add_argument("--beta_cycle_steps", type=int, default=0, help="Triangular beta cycle length (0 disables)")
    parser.add_argument("--kl_free_bits", type=float, default=0.0, help="Free-bits threshold per latent dim (nats)")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 disables)")
    return parser.parse_args()


def log_weight_histograms(model: SolarShapeVAE, writer: SummaryWriter, global_step: int) -> None:
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, global_step)


def log_recon_grid(
    model: SolarShapeVAE,
    dataset,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    seed: int,
    max_items: int = 16,
) -> None:
    if len(dataset) == 0:
        return
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(dataset), generator=g)[: max_items]
    batch = [dataset[int(i)] for i in idxs]
    masks = torch.stack([b["masks"] for b in batch]).to(device)
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(masks)
    # Build discrete labels from masks and predictions.
    target_labels = build_targets(masks)  # (B,H,W)
    pred_labels = recon.argmax(dim=1)  # (B,H,W)

    def colorize(labels: torch.Tensor) -> torch.Tensor:
        # labels: (B,H,W) ints in {0,1,2}; returns (B,3,H,W) RGB in [0,1]
        b, h, w = labels.shape
        rgb = torch.zeros((b, 3, h, w), device=labels.device, dtype=torch.float32)
        rgb[:, 1] = (labels == 1).float()  # green for cytoplasm
        rgb[:, 2] = (labels == 2).float()  # blue for nucleus
        return rgb

    grid_in = make_grid(colorize(target_labels), nrow=4, normalize=False, scale_each=False)
    grid_out = make_grid(colorize(pred_labels), nrow=4, normalize=False, scale_each=False)
    writer.add_image("recon/combined_input", grid_in, global_step)
    writer.add_image("recon/combined_output", grid_out, global_step)


def make_synthetic_dataset(args: argparse.Namespace) -> SolarDataset:
    g = torch.Generator().manual_seed(args.seed)
    cells = _make_synthetic_cells(args.num_cells, args.channels, g)
    cfg = SolarDatasetConfig(channel_names=args.channels)
    return SolarDataset(cells=cells, config=cfg)


def sample_ids_for_subset(subset) -> list[int]:
    if not hasattr(subset, "indices"):
        raise ValueError("Expected a Subset with 'indices' attribute")
    base_ds = subset.dataset
    return [int(base_ds.sample_ids[i]) for i in subset.indices]


def build_targets(masks: torch.Tensor) -> torch.Tensor:
    # masks shape: (B, 2, H, W) with channel0=nucleus, channel1=cell (includes nucleus)
    nucleus = masks[:, 0]
    cell = masks[:, 1]
    cytoplasm = (cell > 0.5) & (nucleus <= 0.5)
    nucleus_mask = nucleus > 0.5
    target = torch.zeros_like(cell, dtype=torch.long)
    target[cytoplasm] = 1
    target[nucleus_mask] = 2
    return target


def reconstruction_loss(
    probs: torch.Tensor, masks: torch.Tensor, class_weights: torch.Tensor, normalize_class_weights: bool = True
) -> torch.Tensor:
    """Weighted categorical NLL summed over pixels (matches SCALER objective scale)."""
    target = build_targets(masks)  # (B,H,W)
    weight = class_weights.to(probs.device)
    if normalize_class_weights:
        # Normalize weights to sum to num_classes to keep loss scale comparable across settings.
        weight = weight * (weight.numel() / weight.sum().clamp_min(1e-6))
    logp = torch.log(probs.clamp_min(1e-6))
    nll = F.nll_loss(logp, target, weight=weight, reduction="none")  # (B,H,W)
    return nll.sum(dim=(1, 2)).mean()


def parse_manifest_line(line: str) -> tuple[Path, Path, str]:
    line = line.strip()
    if not line or line.startswith("#"):
        raise ValueError("skip")
    parts = line.split("\t")
    if len(parts) == 3:
        nuc, cell, sid = parts
        return Path(nuc), Path(cell), sid
    if len(parts) == 2:
        nuc, sid = parts
        return Path(nuc), Path(nuc), sid
    if len(parts) == 1:
        nuc = parts[0]
        return Path(nuc), Path(nuc), 0
    raise ValueError(f"Manifest line must have 1-3 tab-separated fields, got {len(parts)}: {parts}")


def load_real_dataset(args: argparse.Namespace) -> SolarDataset:
    if args.mask_manifest is None:
        raise SystemExit("Provide --mask_manifest for real-data training or use --synthetic.")
    lines = args.mask_manifest.read_text().splitlines()
    cells = []
    raw_cells = []
    for line in lines:
        try:
            nuc_path, cell_path, sid = parse_manifest_line(line)
        except ValueError as e:
            if str(e) == "skip":
                continue
            raise
        if args.mask_root:
            nuc_path = args.mask_root / nuc_path
            cell_path = args.mask_root / cell_path
        raw_cells.append((nuc_path, cell_path, sid))
    if not raw_cells:
        raise SystemExit("Mask manifest produced zero cells; check paths and formatting.")

    # Map potentially string sample identifiers to stable ints.
    uniq: dict[str, int] = {}
    for _, _, sid in raw_cells:
        label = str(sid)
        if label not in uniq:
            uniq[label] = len(uniq)

    for nuc_path, cell_path, sid in raw_cells:
        sid_int = uniq[str(sid)]
        if args.combined_mask:
            cells.append(
                {
                    "combined_mask": nuc_path,  # nuc_path holds the label map
                    "organelle_channels": {},
                    "sample_id": sid_int,
                }
            )
        else:
            cells.append(
                {
                    "nucleus_mask": nuc_path,
                    "cell_mask": cell_path,
                    "organelle_channels": {},
                    "sample_id": sid_int,
                }
            )

    cfg = SolarDatasetConfig(
        channel_names=[],
        high_res_size=args.high_res_size,
        low_res_size=args.low_res_size,
        normalize_channels=False,
        mask_only=True,
        combined_mask_values=(
            {
                "background": args.val_background,
                "cytoplasm": args.val_cytoplasm,
                "nucleus": args.val_nucleus,
            }
            if args.combined_mask
            else None
        ),
    )
    return SolarDataset(cells=cells, config=cfg)


def evaluate(model: SolarShapeVAE, loader: DataLoader, device: torch.device, beta: float, class_weights: torch.Tensor) -> tuple[float, float, float]:
    if loader is None or len(loader.dataset) == 0:
        return 0.0, 0.0, 0.0
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_kl = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            masks = batch["masks"].to(device)
            recon, mu, logvar = model(masks)
            rec = reconstruction_loss(recon, masks, class_weights)
            kl = kl_with_free_bits(mu, logvar, free_bits=0.0)  # evaluation uses full KL without free bits
            loss = rec + beta * kl
            total_loss += loss.item()
            total_bce += rec.item()
            total_kl += kl.item()
            num_batches += 1
    total_loss /= max(1, num_batches)
    total_bce /= max(1, num_batches)
    total_kl /= max(1, num_batches)
    return total_loss, total_bce, total_kl


def train() -> None:
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        dataset = make_synthetic_dataset(args)
    else:
        dataset = load_real_dataset(args)

    class_weights = torch.tensor(args.class_weights, dtype=torch.float32)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    g = torch.Generator().manual_seed(args.seed)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=g)
    print(f"Dataset split -> train: {n_train}, test: {n_test}")

    train_sample_ids = sample_ids_for_subset(train_ds)
    sampler = BalancedBatchSampler(train_sample_ids, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SolarShapeVAE(
        latent_dim=args.latent_dim,
        group_N=args.group_N,
        input_size=args.low_res_size,
        base_filters=args.base_filters,
        use_e2cnn=not args.no_e2cnn,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = make_run_dir(args.log_dir, args)
    ckpt_root = Path(args.checkpoint)
    ckpt_dir = (ckpt_root if ckpt_root.suffix == "" else ckpt_root.parent) / run_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base = run_dir.name
    writer = SummaryWriter(log_dir=run_dir)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_rec = 0.0
        running_kl = 0.0
        for batch in train_loader:
            masks = batch["masks"].to(device)
            recon, mu, logvar = model(masks)
            rec = reconstruction_loss(recon, masks, class_weights.to(device))
            factor = beta_factor(global_step, args)
            kl = kl_with_free_bits(mu, logvar, free_bits=args.kl_free_bits)
            loss = rec + factor * kl
            opt.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()

            running_loss += loss.item()
            running_rec += rec.item()
            running_kl += kl.item()
            global_step += 1
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            writer.add_scalar("train/loss_rec", rec.item(), global_step)
            writer.add_scalar("train/loss_kl", kl.item(), global_step)
            writer.add_scalar("train/warmup_beta", factor, global_step)
        num_batches = len(train_loader)
        print(
            f"Epoch {epoch+1}/{args.epochs} | Loss {running_loss/num_batches:.4f} | "
            f"Rec {running_rec/num_batches:.4f} | KL {running_kl/num_batches:.4f} | Warmup {factor:.2f}"
        )

        log_weight_histograms(model, writer, global_step)
        # Use a fixed beta=1.0 for evaluation to reflect the full objective without warmup scaling.
        test_loss, test_rec, test_kl = evaluate(model, test_loader, device, beta=1.0, class_weights=class_weights.to(device))
        writer.add_scalar("test/loss_total", test_loss, global_step)
        writer.add_scalar("test/loss_rec", test_rec, global_step)
        writer.add_scalar("test/loss_kl", test_kl, global_step)

        log_recon_grid(model, test_ds, device, writer, global_step, seed=args.seed)

        if args.save_every and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            epoch_path = ckpt_dir / f"{ckpt_base}_epoch{epoch+1:04d}.pt"
            save_checkpoint(model, epoch_path)
            print(f"Saved checkpoint to {epoch_path}")

    final_path = ckpt_dir / f"{ckpt_base}_epoch{args.epochs:04d}.pt"
    save_checkpoint(model, final_path)
    print(f"Saved checkpoint to {final_path}")

    writer.close()

    if args.embeddings_out is not None:
        maybe_save_embeddings(model, test_loader, args.embed_limit, args.embeddings_out)


if __name__ == "__main__":
    train()
