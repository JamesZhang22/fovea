"""Ordinal defocus net: 96px grayscale eye patch -> distribution over blur levels.

Tiny CNN per the Google microscopy blueprint. Soft-argmax over the ordered levels
gives a continuous blur radius, distribution entropy gives confidence for free.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fovea.core.score.model import DEFOCUS_RADII_PX

PATCH_PX = 96
N_LEVELS = len(DEFOCUS_RADII_PX)
TARGET_SIGMA_LEVELS = 0.7  # gaussian spread of the soft ordinal target


class FocusNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Linear(64 * 16, N_LEVELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def radius_to_index(radius_px: float) -> float:
    """Continuous position on the ordinal level axis via piecewise-linear interpolation."""
    radii = DEFOCUS_RADII_PX
    if radius_px <= radii[0]:
        return 0.0
    for i in range(len(radii) - 1):
        if radius_px <= radii[i + 1]:
            frac = (radius_px - radii[i]) / (radii[i + 1] - radii[i])
            return i + frac
    return float(len(radii) - 1)


def index_to_radius(index: np.ndarray) -> np.ndarray:
    """Inverse of radius_to_index, vectorized."""
    radii = np.array(DEFOCUS_RADII_PX)
    idx = np.clip(index, 0, len(radii) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(radii) - 1)
    frac = idx - lo
    return radii[lo] * (1 - frac) + radii[hi] * frac


def soft_target(indices: torch.Tensor) -> torch.Tensor:
    """Gaussian-smoothed target over levels for fractional level positions, (N,) -> (N, K)."""
    levels = torch.arange(N_LEVELS, dtype=torch.float32, device=indices.device)
    t = torch.exp(-((levels[None, :] - indices[:, None]) ** 2) / (2 * TARGET_SIGMA_LEVELS**2))
    return t / t.sum(dim=1, keepdim=True)


def ordinal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")


def decode(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous level index via soft-argmax and confidence in [0, 1] from entropy."""
    p = F.softmax(logits, dim=1)
    levels = torch.arange(N_LEVELS, dtype=torch.float32, device=logits.device)
    index = (p * levels).sum(dim=1)
    entropy = -(p * (p + 1e-9).log()).sum(dim=1)
    confidence = 1.0 - entropy / np.log(N_LEVELS)
    return index, confidence
