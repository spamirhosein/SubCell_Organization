from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as enn


@dataclass
class SolarShapeVAEConfig:
    latent_dim: int = 32
    group_N: int = 8
    input_size: int = 128
    base_filters: int = 32
    nlayers: int = 3
    kernel_size: int = 5
    use_e2cnn: bool = True


class SolarShapeVAE(nn.Module):
    """Rotation-invariant VAE for cell/nucleus masks using e2cnn."""

    def __init__(
        self,
        latent_dim: int = 32,
        group_N: int = 8,
        input_size: int = 128,
        base_filters: int = 32,
        nlayers: int = 3,
        kernel_size: int = 5,
        use_e2cnn: bool = True,
    ) -> None:
        super().__init__()
        self.config = SolarShapeVAEConfig(
            latent_dim=latent_dim,
            group_N=group_N,
            input_size=input_size,
            base_filters=base_filters,
            nlayers=nlayers,
            kernel_size=kernel_size,
            use_e2cnn=use_e2cnn,
        )
        self.use_e2cnn = use_e2cnn
        if self.use_e2cnn:
            # Use dihedral group (rotations + flips) to enforce equivariance to rotations and reflections.
            self.r2_act = gspaces.FlipRot2dOnR2(N=group_N)
            self.input_type = enn.FieldType(self.r2_act, 2 * [self.r2_act.trivial_repr])
            self.encoder, enc_out_type = self._build_encoder_e2()
            self.gpool = enn.GroupPooling(enc_out_type)
        else:
            # Plain Conv2d backbone (diagnostic mode; not equivariant).
            self.r2_act = None
            self.input_type = None
            self.encoder = self._build_encoder_plain()
            self.gpool = None

        # Use lazy linears to infer the correct flattened dimension after pooling.
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_logvar = nn.LazyLinear(latent_dim)

        top_channels = self.config.base_filters * (2 ** (self.config.nlayers - 1))
        spatial_dim = self.config.input_size // (2 ** self.config.nlayers)
        self.fc_dec = nn.Linear(latent_dim, top_channels * spatial_dim * spatial_dim)

        dec_layers = []
        in_ch = top_channels
        for i in range(self.config.nlayers):
            out_ch = max(self.config.base_filters, in_ch // 2)
            dec_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            dec_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        dec_layers.append(nn.Conv2d(in_ch, 3, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

    def _build_encoder_e2(self) -> Tuple[enn.SequentialModule, enn.FieldType]:
        layers = []
        in_type = self.input_type
        out_type = None
        for i in range(self.config.nlayers):
            c = self.config.base_filters * (2 ** i)
            out_type = enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])
            layers.append(
                enn.R2Conv(
                    in_type,
                    out_type,
                    kernel_size=self.config.kernel_size,
                    padding=self.config.kernel_size // 2,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(enn.InnerBatchNorm(out_type))
            layers.append(enn.ReLU(out_type, inplace=True))
            in_type = out_type

        return enn.SequentialModule(*layers), out_type  # type: ignore[arg-type]

    def _build_encoder_plain(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_ch = 2
        for i in range(self.config.nlayers):
            out_ch = self.config.base_filters * (2 ** i)
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=self.config.kernel_size,
                    padding=self.config.kernel_size // 2,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_e2cnn:
            gx = enn.GeometricTensor(x, self.input_type)
            gx = self.encoder(gx)
            gx = self.gpool(gx)
            h = gx.tensor.flatten(1)
        else:
            h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        spatial_dim = self.config.input_size // (2 ** self.config.nlayers)
        h = self.fc_dec(z)
        top_channels = self.config.base_filters * (2 ** (self.config.nlayers - 1))
        h = h.view(-1, top_channels, spatial_dim, spatial_dim)
        x = self.decoder(h)
        return torch.softmax(x, dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean()
