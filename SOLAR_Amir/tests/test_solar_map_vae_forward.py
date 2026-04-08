from __future__ import annotations

import torch

from solar.models.solar_map_vae import SolarMapVAE, SolarMapVAEConfig


def test_solarmap_forward_shapes():
    config = SolarMapVAEConfig(
        num_channels=3,
        input_size=128,
        latent_dim=8,
        cond_morph_dim=4,
        cond_cell_dim=2,
        cond_sample_dim=1,
        base_filters=8,
        num_blocks=3,
        hidden_dim=64,
    )
    model = SolarMapVAE(config)
    x = torch.randn(2, 3, 128, 128)
    mu_shape = torch.randn(2, 4)
    logvar_shape = torch.randn(2, 4)
    cond_cell = torch.randn(2, 2)
    cond_sample = torch.randn(2, 1)

    recon, mu, logvar, z, cond_morph = model(x, mu_shape, logvar_shape, cond_cell, cond_sample)

    assert recon.shape == x.shape
    assert mu.shape == (2, 3, 8)
    assert logvar.shape == (2, 3, 8)
    assert z.shape == (2, 3, 8)
    assert cond_morph.shape == mu_shape.shape

    kl = model.kl_total(mu, logvar)
    assert kl.dim() == 0
    rec = model.reconstruction_loss(recon, x)
    assert rec.dim() == 0

