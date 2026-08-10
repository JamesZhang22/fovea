"""Train the in-domain eye model on our own labels.

Usage:
  uv run python tools/train_eye_own.py <folder-with-eye_labels.jsonl>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from eye_net import (
    CROP_PX,
    INPUT_PX,
    EyeCrops,
    EyeNet,
    collate_valid,
    decode,
    simcc_loss,
    soft_target,
)

from fovea.core.group.burst import group_bursts
from fovea.core.ingest import cr3
from fovea.core.ingest.cache import Cache
from fovea.core.ingest.decode import roi_native
from fovea.core.scan import scan_folder

PREP = Path("data/own-prep")
MODEL_OUT = Path("models/eye_own_v1.pt")
BOX_PAD_FRACTION = 0.15
VAL_FRACTION = 0.15
EPOCHS = 60
BATCH = 32
LR = 8e-4


def square_region(box: dict) -> tuple[int, int, int, int, float]:
    """Padded square around a bird box in full-res pixels, plus its side length."""
    w, h = box["x1"] - box["x0"], box["y1"] - box["y0"]
    side = max(w, h) * (1 + 2 * BOX_PAD_FRACTION)
    cx, cy = (box["x0"] + box["x1"]) / 2, (box["y0"] + box["y1"]) / 2
    return int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2), side


def prep(labels_path: Path, folder: Path) -> list[dict]:
    """Extract crops for every labeled row, tag each with its burst for the split."""
    rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
    rows = [r for r in rows if r["eye"]]

    entries = scan_folder(folder, Cache(folder / ".fovea" / "cache.sqlite"))
    burst_of = {}
    for b_id, burst in enumerate(group_bursts(entries)):
        for e in burst:
            burst_of[e["path"]] = b_id

    PREP.mkdir(parents=True, exist_ok=True)
    out = []
    jpeg_cache: dict[str, bytes] = {}
    for r in rows:
        x0, y0, x1, y1, side = square_region(r["box"])
        if r["path"] not in jpeg_cache:
            p = Path(r["path"])
            jpeg_cache.clear()
            jpeg_cache[r["path"]] = cr3.read_range(p, cr3.read_previews(p).full)
        crop = roi_native(jpeg_cache[r["path"]], (x0, y0, x1, y1)).resize(
            (CROP_PX, CROP_PX), Image.LANCZOS
        )
        scale = CROP_PX / side
        ex, ey = (r["eye"][0] - x0) * scale, (r["eye"][1] - y0) * scale
        if not (0 <= ex < CROP_PX and 0 <= ey < CROP_PX):
            continue
        safe_id = r["id"].replace("#", "_")
        crop.save(PREP / f"{safe_id}.jpg", quality=92)
        out.append(
            {
                "id": safe_id,
                "eyes": [[ex, ey]],
                "burst": burst_of.get(r["path"], -1),
                "side_px": side,
                "box_px": max(r["box"]["x1"] - r["box"]["x0"], r["box"]["y1"] - r["box"]["y0"]),
            }
        )
    (PREP / "index.json").write_text(json.dumps(out))
    return out


def burst_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Whole bursts go to train or val, never both."""
    bursts = sorted({r["burst"] for r in rows})
    rng = np.random.default_rng(7)
    rng.shuffle(bursts)
    n_val = max(1, round(len(bursts) * VAL_FRACTION))
    val_bursts = set(bursts[:n_val])
    train = [r for r in rows if r["burst"] not in val_bursts]
    val = [r for r in rows if r["burst"] in val_bursts]
    return train, val


def train(rows: list[dict]) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_rows, val_rows = burst_split(rows)
    print(f"{len(train_rows)} train / {len(val_rows)} val (burst-aware split)")

    train_dl = DataLoader(
        EyeCrops(PREP, train_rows, True),
        batch_size=BATCH,
        shuffle=True,
        num_workers=6,
        collate_fn=collate_valid,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        EyeCrops(PREP, val_rows, False),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_valid,
        persistent_workers=True,
    )
    model = EyeNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(train_dl))

    best = 1e9
    for epoch in range(EPOCHS):
        model.train()
        total = n = 0
        for xs, eyes in train_dl:
            xs, eyes = xs.to(device), eyes.to(device)
            lx, ly = model(xs)
            loss = simcc_loss(lx, soft_target(eyes[:, 0])) + simcc_loss(ly, soft_target(eyes[:, 1]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xs)
            n += len(xs)

        model.eval()
        errs_norm = []
        with torch.no_grad():
            for i, (xs, eyes) in enumerate(val_dl):
                xy, _ = decode(*model(xs.to(device)))
                err_px = ((xy.cpu() - eyes) ** 2).sum(dim=1).sqrt()
                for j, e in enumerate(err_px.tolist()):
                    rec = val_rows[i * BATCH + j]
                    full_px = e * rec["side_px"] / INPUT_PX
                    errs_norm.append(full_px / rec["box_px"])
        errs_t = torch.tensor(errs_norm)
        median = errs_t.median().item()
        pck5 = (errs_t < 0.05).float().mean().item()
        pck10 = (errs_t < 0.10).float().mean().item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"epoch {epoch + 1:2d}/{EPOCHS} loss {total / n:.4f} | val median "
                f"{median:.1%} of box | within 5%: {pck5:.1%} | within 10%: {pck10:.1%}",
                flush=True,
            )
        if median < best:
            best = median
            MODEL_OUT.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
    print(f"best val median {best:.1%} of box -> {MODEL_OUT}")


if __name__ == "__main__":
    folder = Path(sys.argv[1]).expanduser().resolve()
    rows = prep(folder / "eye_labels.jsonl", folder)
    print(f"prepped {len(rows)} labeled crops")
    train(rows)
