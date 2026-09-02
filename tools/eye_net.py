"""SimCC-style eye keypoint net, coordinate classification over x and y bins.

RTMPose's core idea in plain PyTorch: the head predicts a distribution over sub-pixel
bins per axis, soft-argmax gives a smooth coordinate, distribution entropy gives
confidence for free.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

INPUT_PX = 192
CROP_PX = 256  # stored crop size, augmentation crops down to INPUT_PX
BINS_PER_AXIS = INPUT_PX * 2  # half-pixel resolution
TARGET_SIGMA_BINS = 8.0  # gaussian spread of the soft training target


class EyeCrops(Dataset):
    """Stored square crops with eye points, random flip and crop-window augmentation.

    rows: {"id": str, "eyes": [[x, y], ...]} with coordinates in CROP_PX space,
    images live at <img_dir>/<id>.jpg. Samples whose eye leaves the window get (-1, -1),
    filter them with collate_valid.
    """

    def __init__(self, img_dir: Path, rows: list[dict], augment: bool) -> None:
        self.img_dir = img_dir
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        rec = self.rows[i]
        im = Image.open(self.img_dir / f"{rec['id']}.jpg")
        eyes = [list(e) for e in rec["eyes"]]
        rng = np.random.default_rng()

        if not self.augment:
            # whole-crop resize matches inference, keeps every eval sample
            im = im.resize((INPUT_PX, INPUT_PX), Image.BILINEAR)
            s = INPUT_PX / CROP_PX
            eyes = [[ex * s, ey * s] for ex, ey in eyes]
            eye = eyes[0]
        else:
            if rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                eyes = [[CROP_PX - ex, ey] for ex, ey in eyes]
            max_off = CROP_PX - INPUT_PX
            ox, oy = rng.integers(0, max_off + 1), rng.integers(0, max_off + 1)
            im = im.crop((ox, oy, ox + INPUT_PX, oy + INPUT_PX))
            eyes = [[ex - ox, ey - oy] for ex, ey in eyes]
            inside = [e for e in eyes if 0 <= e[0] < INPUT_PX and 0 <= e[1] < INPUT_PX]
            eye = inside[int(rng.integers(0, len(inside)))] if inside else [-1.0, -1.0]

        arr = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return arr, torch.tensor(eye, dtype=torch.float32)


def collate_valid(batch):
    """Drop samples whose eye left the augmentation crop."""
    xs = torch.stack([b[0] for b in batch])
    eyes = torch.stack([b[1] for b in batch])
    valid = eyes[:, 0] >= 0
    return xs[valid], eyes[valid]


class EyeNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.neck = nn.Conv2d(576, 128, 1)
        flat = 128 * (INPUT_PX // 32) ** 2
        self.head_x = nn.Linear(flat, BINS_PER_AXIS)
        self.head_y = nn.Linear(flat, BINS_PER_AXIS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = F.relu(self.neck(self.backbone(x))).flatten(1)
        return self.head_x(feat), self.head_y(feat)


def soft_target(coords_px: torch.Tensor) -> torch.Tensor:
    """Gaussian-smoothed one-hot over bins for coordinates in input pixels, (N,) -> (N, BINS)."""
    bins = torch.arange(BINS_PER_AXIS, dtype=torch.float32, device=coords_px.device)
    centers = coords_px * (BINS_PER_AXIS / INPUT_PX)
    t = torch.exp(-((bins[None, :] - centers[:, None]) ** 2) / (2 * TARGET_SIGMA_BINS**2))
    return t / t.sum(dim=1, keepdim=True)


def simcc_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")


def decode(logits_x: torch.Tensor, logits_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-argmax coordinates in input pixels and a confidence in [0, 1] per sample."""
    coords = []
    confs = []
    for logits in (logits_x, logits_y):
        p = F.softmax(logits, dim=1)
        bins = torch.arange(BINS_PER_AXIS, dtype=torch.float32, device=logits.device)
        coords.append((p * bins).sum(dim=1) * (INPUT_PX / BINS_PER_AXIS))
        confs.append(p.max(dim=1).values)
    xy = torch.stack(coords, dim=1)
    conf = torch.stack(confs, dim=1).min(dim=1).values / soft_target_peak()
    return xy, conf.clamp(max=1.0)


def soft_target_peak() -> float:
    """Peak probability of a perfectly-predicted soft target, normalizes confidence to ~1."""
    total = sum(math.exp(-(i**2) / (2 * TARGET_SIGMA_BINS**2)) for i in range(-40, 41))
    return 1.0 / total
