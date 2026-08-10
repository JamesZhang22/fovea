"""CUB-200-2011 eye keypoint prototype.

Usage:
  uv run python tools/train_eye_prototype.py prep    # extract square bird crops once
  uv run python tools/train_eye_prototype.py train   # train + eval, saves models/eye_prototype.pt

Validates the bird-crop -> SimCC eye keypoint architecture end to end before any of
our own labels exist.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

CUB = Path("data/cub/CUB_200_2011")
PREP = Path("data/cub/prep")
MODEL_OUT = Path("models/eye_prototype.pt")
LEFT_EYE, RIGHT_EYE = 7, 11
BOX_PAD_FRACTION = 0.15
EPOCHS = 20
BATCH = 64
LR = 1e-3


def load_annotations() -> list[dict]:
    """One record per image with bbox, visible eyes, and train/test flag."""
    images = {}
    for line in (CUB / "images.txt").read_text().splitlines():
        i, name = line.split()
        images[i] = {"id": i, "file": name, "eyes": []}
    for line in (CUB / "bounding_boxes.txt").read_text().splitlines():
        i, x, y, w, h = line.split()
        images[i]["box"] = [float(x), float(y), float(w), float(h)]
    for line in (CUB / "train_test_split.txt").read_text().splitlines():
        i, is_train = line.split()
        images[i]["train"] = is_train == "1"
    for line in (CUB / "parts" / "part_locs.txt").read_text().splitlines():
        i, part, x, y, visible = line.split()
        if int(part) in (LEFT_EYE, RIGHT_EYE) and visible == "1":
            images[i]["eyes"].append([float(x), float(y)])
    return [r for r in images.values() if r["eyes"]]


def square_crop_region(box: list[float]) -> tuple[float, float, float]:
    """Left, top, and side of the padded square around a CUB bbox."""
    x, y, w, h = box
    side = max(w, h) * (1 + 2 * BOX_PAD_FRACTION)
    cx, cy = x + w / 2, y + h / 2
    return cx - side / 2, cy - side / 2, side


def prep() -> None:
    records = load_annotations()
    PREP.mkdir(parents=True, exist_ok=True)

    def one(rec: dict) -> dict | None:
        im = Image.open(CUB / "images" / rec["file"]).convert("RGB")
        left, top, side = square_crop_region(rec["box"])
        crop = im.crop((round(left), round(top), round(left + side), round(top + side)))
        crop = crop.resize((CROP_PX, CROP_PX), Image.LANCZOS)
        scale = CROP_PX / side
        eyes = [[(ex - left) * scale, (ey - top) * scale] for ex, ey in rec["eyes"]]
        eyes = [[ex, ey] for ex, ey in eyes if 0 <= ex < CROP_PX and 0 <= ey < CROP_PX]
        if not eyes:
            return None
        crop.save(PREP / f"{rec['id']}.jpg", quality=92)
        return {"id": rec["id"], "eyes": eyes, "train": rec["train"]}

    with ThreadPoolExecutor(max_workers=10) as pool:
        rows = [r for r in pool.map(one, records) if r]
    (PREP / "index.json").write_text(json.dumps(rows))
    n_train = sum(1 for r in rows if r["train"])
    print(f"prepped {len(rows)} crops ({n_train} train / {len(rows) - n_train} test)")


def train() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = json.loads((PREP / "index.json").read_text())
    train_rows = [r for r in rows if r["train"]]
    test_rows = [r for r in rows if not r["train"]]
    train_dl = DataLoader(
        EyeCrops(PREP, train_rows, True),
        batch_size=BATCH,
        shuffle=True,
        num_workers=6,
        collate_fn=collate_valid,
        persistent_workers=True,
    )
    test_dl = DataLoader(
        EyeCrops(PREP, test_rows, False),
        batch_size=BATCH,
        shuffle=False,
        num_workers=4,
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
        errs = []
        with torch.no_grad():
            for xs, eyes in test_dl:
                xy, _ = decode(*model(xs.to(device)))
                errs.extend(((xy.cpu() - eyes) ** 2).sum(dim=1).sqrt().tolist())
        errs_t = torch.tensor(errs)
        median = errs_t.median().item()
        pck = (errs_t < 0.05 * INPUT_PX).float().mean().item()
        print(
            f"epoch {epoch + 1:2d}/{EPOCHS} loss {total / n:.4f} "
            f"| test median {median:.1f}px | PCK@5% {pck:.1%}",
            flush=True,
        )
        if median < best:
            best = median
            MODEL_OUT.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
    print(f"best median {best:.1f}px on {INPUT_PX}px crops -> {MODEL_OUT}")


if __name__ == "__main__":
    {"prep": prep, "train": train}[sys.argv[1]]()
