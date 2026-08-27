from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SolarMapVAEConfig:
    num_channels: int
    input_size: int = 128
    latent_dim: int = 16
    cond_morph_dim: int = 32
    cond_cell_dim: int = 0
    cond_sample_dim: int = 0
    base_filters: int = 32
    num_blocks: int = 4
    hidden_dim: int = 256


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, base_filters: int, num_blocks: int, input_size: int, hidden_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_channels
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = out_ch
        self.net = nn.Sequential(*layers)
        spatial = input_size // (2 ** num_blocks)
        self.out_dim = ch * spatial * spatial
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(self.out_dim, hidden_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.proj(self.net(x))


class ConvDecoder(nn.Module):
    def __init__(self, base_filters: int, num_blocks: int, input_size: int) -> None:
        super().__init__()
        self.start_channels = base_filters * (2 ** (num_blocks - 1))
        self.spatial = input_size // (2 ** num_blocks)
        layers: List[nn.Module] = []
        ch = self.start_channels
        for i in range(num_blocks):
            out_ch = max(base_filters, ch // 2)
            layers.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch = out_ch
        layers.append(nn.Conv2d(ch, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b = h.shape[0]
        h = h.view(b, self.start_channels, self.spatial, self.spatial)
        return self.net(h)


class MarkerVAE(nn.Module):
    def __init__(self, config: SolarMapVAEConfig, cond_dim: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder(
            in_channels=1,
            base_filters=config.base_filters,
            num_blocks=config.num_blocks,
            input_size=config.input_size,
            hidden_dim=config.hidden_dim,
        )
        self.fc_mu = nn.Linear(config.hidden_dim + cond_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim + cond_dim, config.latent_dim)
        self.fc_dec = nn.Linear(config.latent_dim + cond_dim, self.encoder.out_dim)
        self.decoder = ConvDecoder(
            base_filters=config.base_filters,
            num_blocks=config.num_blocks,
            input_size=config.input_size,
        )

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = torch.cat([h, cond], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, cond], dim=1)
        h = self.fc_dec(h)
        return self.decoder(h)


class SolarMapVAE(nn.Module):
    """Conditional per-marker VAE for localization maps."""

    def __init__(self, config: SolarMapVAEConfig) -> None:
        super().__init__()
        self.config = config
        cond_dim = config.cond_morph_dim + config.cond_cell_dim + config.cond_sample_dim
        self.markers = nn.ModuleList([MarkerVAE(config, cond_dim=cond_dim) for _ in range(config.num_channels)])

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=-1).mean()

    def forward(
        self,
        x: torch.Tensor,
        mu_shape: torch.Tensor,
        logvar_shape: torch.Tensor,
        cond_cell: torch.Tensor,
        cond_sample: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        cond_morph = self.reparameterize(mu_shape, logvar_shape)
        cond = torch.cat([cond_morph, cond_cell, cond_sample], dim=1)

        recon_parts: List[torch.Tensor] = []
        mu_list: List[torch.Tensor] = []
        logvar_list: List[torch.Tensor] = []
        z_list: List[torch.Tensor] = []

        for idx, marker in enumerate(self.markers):
            xi = x[:, idx : idx + 1]
            mu, logvar = marker.encode(xi, cond)
            z = self.reparameterize(mu, logvar)
            recon_i = marker.decode(z, cond)
            recon_parts.append(recon_i)
            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)

        recon = torch.cat(recon_parts, dim=1)
        mu_all = torch.stack(mu_list, dim=1)
        logvar_all = torch.stack(logvar_list, dim=1)
        z_all = torch.stack(z_list, dim=1)
        return recon, mu_all, logvar_all, z_all, cond_morph

    def reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, target, reduction="mean")

    def kl_total(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=-1).mean()