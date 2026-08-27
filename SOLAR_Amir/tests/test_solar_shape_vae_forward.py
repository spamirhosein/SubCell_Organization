from __future__ import annotations

import pytest
import torch

e2cnn = pytest.importorskip("e2cnn")

from solar.models.solar_shape_vae import SolarShapeVAE


def test_forward_shapes():
    model = SolarShapeVAE(latent_dim=32, group_N=8, input_size=128)
    x = torch.rand(4, 2, 128, 128)
    recon, mu, logvar = model(x)
    assert recon.shape == (4, 3, 128, 128)
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)
    assert torch.all((recon >= 0.0) & (recon <= 1.0))
