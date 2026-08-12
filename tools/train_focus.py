"""Train the ordinal focus model on synthetically degraded sharp eye patches.

Usage: uv run python tools/train_focus.py

On-the-fly degradation: each sample is a random corpus patch convolved with a
random-radius disc PSF (80%) or a linear motion kernel (20%, labeled at its
equivalent severity). Split is by patch, patches come one per burst.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from focus_degrade import (
    DEFOCUS_RADII_PX,
    apply_kernel,
    disc_kernel,
    motion_kernel,
)
from focus_net import FocusNet, decode, index_to_radius, ordinal_loss, radius_to_index, soft_target

CORPUS = Path("data/sharp-eyes")
MODEL_OUT = Path("models/focus_v1.pt")
VAL_FRACTION = 0.2
MOTION_SHARE = 0.2
EPOCHS = 15
STEPS_PER_EPOCH = 250
BATCH = 64
LR = 1e-3
MAX_RADIUS = DEFOCUS_RADII_PX[-1]


def load_patches() -> list[np.ndarray]:
    rejected = set(json.loads((CORPUS / "rejected.json").read_text()))
    kept = [r for r in json.loads((CORPUS / "index.json").read_text()) if r["id"] not in rejected]
    return [
        np.asarray(Image.open(CORPUS / "patches" / f"{r['id']}.png").convert("L"), dtype=np.float32)
        for r in kept
    ]


class DegradedEyes(Dataset):
    """Infinite-style sampler: random patch x random degradation each draw."""

    def __init__(self, patches: list[np.ndarray], n_samples: int, seed: int) -> None:
        self.patches = patches
        self.n = n_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        gray = self.patches[int(rng.integers(len(self.patches)))]
        if rng.random() < 0.5:
            gray = gray[:, ::-1]

        max_index = len(DEFOCUS_RADII_PX) - 1
        if rng.random() < MOTION_SHARE:
            length = float(rng.uniform(2.0, 2 * MAX_RADIUS))
            kernel = motion_kernel(length, float(rng.uniform(0, 180)))
            radius = length / 2
        else:
            index = float(rng.uniform(0, max_index))
            radius = float(index_to_radius(np.array([index]))[0])
            kernel = disc_kernel(radius)
        out = apply_kernel(gray, kernel, rng)
        target_index = radius_to_index(min(radius, MAX_RADIUS))
        x = torch.from_numpy((out / 255.0).copy())[None]
        return x, torch.tensor(target_index, dtype=torch.float32)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    patches = load_patches()
    rng = np.random.default_rng(7)
    order = rng.permutation(len(patches))
    n_val = int(len(patches) * VAL_FRACTION)
    val_patches = [patches[i] for i in order[:n_val]]
    train_patches = [patches[i] for i in order[n_val:]]
    print(f"{len(train_patches)} train / {len(val_patches)} val patches")

    train_dl = DataLoader(
        DegradedEyes(train_patches, EPOCHS * STEPS_PER_EPOCH * BATCH, seed=1),
        batch_size=BATCH,
        num_workers=6,
        persistent_workers=True,
    )
    val_dl = DataLoader(DegradedEyes(val_patches, 2000, seed=2), batch_size=BATCH, num_workers=4)

    model = FocusNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e3:.0f}k parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * STEPS_PER_EPOCH)

    best = 1e9
    epoch = 0
    total = n = 0
    for step, (xs, idx) in enumerate(train_dl, start=1):
        model.train()
        xs, idx = xs.to(device), idx.to(device)
        loss = ordinal_loss(model(xs), soft_target(idx))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total += loss.item() * len(xs)
        n += len(xs)

        if step % STEPS_PER_EPOCH == 0:
            epoch += 1
            model.eval()
            errs = []
            with torch.no_grad():
                for vx, vidx in val_dl:
                    pred_idx, _ = decode(model(vx.to(device)))
                    pred_r = index_to_radius(pred_idx.cpu().numpy())
                    true_r = index_to_radius(vidx.numpy())
                    errs.extend(np.abs(pred_r - true_r).tolist())
            mae = float(np.mean(errs))
            print(
                f"epoch {epoch:2d}/{EPOCHS} loss {total / n:.4f} | val MAE {mae:.2f}px blur radius",
                flush=True,
            )
            total = n = 0
            if mae < best:
                best = mae
                MODEL_OUT.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), MODEL_OUT)
    print(f"best val MAE {best:.2f}px -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
