"""PilotNet (Bojarski et al. 2016) with the vehicle speed concatenated at the fully connected stage."""

from __future__ import annotations

import torch
from torch import nn

from ads import config


class PilotNet(nn.Module):
    """5 conv layers + FC 100/50/10, ~250k parameters.

    Input: YUV image, NCHW, 3x66x200, uint8 or float in [0, 255]; speed in km/h, shape (N,).
    Output: steering in MetaDrive's normalised units, shape (N,). Not squashed; clamp at use.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
            nn.Flatten(),
        )
        n_features = 64 * 1 * 18  # spatial size after the conv stack for a 66x200 input
        self.head = nn.Sequential(
            nn.Linear(n_features + 1, 100),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, image: torch.Tensor, speed_kmh: torch.Tensor) -> torch.Tensor:
        x = image.float() / 127.5 - 1.0
        features = self.features(x)
        speed = (speed_kmh.float() / config.SPEED_SCALE_KMH).reshape(-1, 1)
        return self.head(torch.cat([features, speed], dim=1)).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
