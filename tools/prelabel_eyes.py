"""Pre-label bird eyes with the prototype model for faster hand-correction.

Usage: uv run python tools/prelabel_eyes.py <folder> [--weights models/eye_prototype.pt]
           [--out <folder>/eye_prelabels.jsonl] [--eval <folder>/eye_labels.jsonl]

Writes predicted eye points (full-resolution image coordinates plus confidence) for every
detected bird crop. The labeler shows them as suggestions to confirm or correct.
With --eval, instead compares predictions against existing hand labels and reports error.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eye_net import INPUT_PX, EyeNet, decode

from fovea.core.ingest import cr3
from fovea.core.ingest.cache import Cache
from fovea.core.ingest.decode import roi_native
from fovea.core.pipeline import PipelineConfig, run_pipeline

BOX_PAD_FRACTION = 0.15  # square padded crop


def crop_region(box: dict) -> tuple[int, int, int, int]:
    """Padded square around the bird box in full-resolution pixels."""
    w, h = box["x1"] - box["x0"], box["y1"] - box["y0"]
    side = max(w, h) * (1 + 2 * BOX_PAD_FRACTION)
    cx, cy = (box["x0"] + box["x1"]) / 2, (box["y0"] + box["y1"]) / 2
    return (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))


def predict_folder(folder: Path, weights: Path) -> list[dict]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = EyeNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    entries = run_pipeline(
        folder,
        PipelineConfig(score=False, export=False, detect=True),
        Cache(folder / ".fovea" / "cache.sqlite"),
    )
    rows = []
    for e in entries:
        path = Path(e["path"])
        jpeg = None
        for i, box in enumerate(e.get("birds") or []):
            if jpeg is None:
                jpeg = cr3.read_range(path, cr3.read_previews(path).full)
            x0, y0, x1, y1 = crop_region(box)
            crop = roi_native(jpeg, (x0, y0, x1, y1)).resize((INPUT_PX, INPUT_PX))
            arr = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
            with torch.no_grad():
                xy, conf = decode(*model(torch.from_numpy(arr)[None].to(device)))
            ex = x0 + float(xy[0, 0]) / INPUT_PX * (x1 - x0)
            ey = y0 + float(xy[0, 1]) / INPUT_PX * (y1 - y0)
            rows.append(
                {
                    "id": f"{path.name}#{i}",
                    "path": str(path),
                    "box": box,
                    "eye": [ex, ey],
                    "confidence": round(float(conf[0]), 3),
                }
            )
            print(f"\r{len(rows)} predicted", end="", file=sys.stderr)
    print(file=sys.stderr)
    return rows


def evaluate(rows: list[dict], labels_path: Path) -> None:
    """Prediction error against hand labels, normalized by bird box size."""
    truth = {}
    for line in labels_path.read_text().splitlines():
        r = json.loads(line)
        if r["eye"] is not None:
            truth[r["id"]] = r
    errs = []
    for row in rows:
        t = truth.get(row["id"])
        if t is None:
            continue
        d = ((row["eye"][0] - t["eye"][0]) ** 2 + (row["eye"][1] - t["eye"][1]) ** 2) ** 0.5
        size = max(t["box"]["x1"] - t["box"]["x0"], t["box"]["y1"] - t["box"]["y0"])
        errs.append((d / size, d, row["confidence"]))
    if not errs:
        print("no overlapping labels to evaluate against")
        return
    norm = sorted(e[0] for e in errs)
    n = len(norm)
    print(f"{n} predictions vs hand labels")
    print(
        f"median error: {norm[n // 2]:.1%} of box size ({sorted(e[1] for e in errs)[n // 2]:.0f}px)"
    )
    for pct in (0.05, 0.10):
        print(f"within {pct:.0%} of box: {sum(v < pct for v in norm) / n:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="pre-label eyes with the prototype model")
    parser.add_argument("folder")
    parser.add_argument("--weights", default="models/eye_prototype.pt")
    parser.add_argument("--out", help="output JSONL (default <folder>/eye_prelabels.jsonl)")
    parser.add_argument("--eval", dest="eval_labels", help="compare against hand labels instead")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    rows = predict_folder(folder, Path(args.weights))
    if args.eval_labels:
        evaluate(rows, Path(args.eval_labels))
    else:
        out = Path(args.out) if args.out else folder / "eye_prelabels.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{len(rows)} pre-labels -> {out}")


if __name__ == "__main__":
    main()
