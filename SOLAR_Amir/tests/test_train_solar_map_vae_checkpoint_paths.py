from __future__ import annotations

from pathlib import Path

from solar.train.train_solar_map_vae import resolve_checkpoint_dir


def test_checkpoint_paths_stage2_convention() -> None:
    ckpt_root = Path("/abs/SOLAR/SolarMapVAE")
    run_name = "ld16_bf32_wu3000_lr0.0001_mb1_fb0_e21"

    root, ckpt_dir = resolve_checkpoint_dir(ckpt_root, run_name)
    assert root == ckpt_root
    assert ckpt_dir == ckpt_root / run_name

    epoch_path = ckpt_dir / f"{run_name}_epoch0030.pt"
    final_path = ckpt_dir / f"{run_name}_final.pt"

    assert str(epoch_path) == "/abs/SOLAR/SolarMapVAE/ld16_bf32_wu3000_lr0.0001_mb1_fb0_e21/ld16_bf32_wu3000_lr0.0001_mb1_fb0_e21_epoch0030.pt"
    assert str(final_path) == "/abs/SOLAR/SolarMapVAE/ld16_bf32_wu3000_lr0.0001_mb1_fb0_e21/ld16_bf32_wu3000_lr0.0001_mb1_fb0_e21_final.pt"
