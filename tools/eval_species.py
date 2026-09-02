"""Render an HTML sheet of per-burst species predictions for eyeball evaluation.

Usage: uv run python tools/eval_species.py [--region north-america] <folder>...

One card per burst: the classified bird crop and the top-3 with confidences. The
reviewer knows what they shot, so wrong IDs jump out immediately. Writes
<folder>/.fovea/species_sheet.html per folder.
"""

import argparse
import base64
import html
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fovea.core.group.burst import group_bursts
from fovea.core.ingest import cr3, decode
from fovea.core.ingest.cache import Cache
from fovea.core.pipeline import PipelineConfig, best_species_frame, run_pipeline
from fovea.core.species.classify import CROP_PAD_FRACTION

THUMB_WIDTH_PX = 260
CONFIDENT = 0.9  # card confidence coloring thresholds.
SHAKY = 0.5

STYLE = """
body { background: #111; color: #ddd; font: 13px system-ui; margin: 16px; }
.grid { display: flex; flex-wrap: wrap; gap: 10px; }
.card { width: 260px; background: #1a1a1a; border-radius: 6px; overflow: hidden; }
.card img { width: 260px; display: block; }
.card .body { padding: 8px; }
.name { color: #888; font-size: 11px; margin-bottom: 4px; }
.pred { display: flex; justify-content: space-between; }
.conf-high { color: #6bff9e; } .conf-mid { color: #d7b24d; } .conf-low { color: #ff6b6b; }
"""


def crop_thumb(entry: dict) -> str | None:
    """Base64 JPEG of the padded bird crop the classifier saw."""
    path = Path(entry["path"])
    previews = cr3.read_previews(path)
    src = previews.full or previews.prvw
    if src is None:
        return None
    im = decode.decode_scaled(cr3.read_range(path, src), 3480)
    scale = im.width / (entry["meta"].get("ImageWidth") or im.width)
    box = max(entry["birds"], key=lambda b: b["confidence"])
    w, h = box["x1"] - box["x0"], box["y1"] - box["y0"]
    crop = im.crop(
        (
            max(0, int((box["x0"] - CROP_PAD_FRACTION * w) * scale)),
            max(0, int((box["y0"] - CROP_PAD_FRACTION * h) * scale)),
            min(im.width, int((box["x1"] + CROP_PAD_FRACTION * w) * scale)),
            min(im.height, int((box["y1"] + CROP_PAD_FRACTION * h) * scale)),
        )
    )
    crop.thumbnail((THUMB_WIDTH_PX, THUMB_WIDTH_PX * 3))
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def card(entry: dict) -> str | None:
    preds = entry.get("species")
    if not preds:
        return None
    thumb = crop_thumb(entry)
    if thumb is None:
        return None
    top = preds[0]["confidence"]
    cls = "conf-high" if top >= CONFIDENT else "conf-mid" if top >= SHAKY else "conf-low"
    rows = "".join(
        f'<div class="pred"><span>{html.escape(p["common"] or p["scientific"])}</span>'
        f"<span>{p['confidence']:.2f}</span></div>"
        for p in preds
    )
    return (
        f'<div class="card"><img src="data:image/jpeg;base64,{thumb}">'
        f'<div class="body {cls}"><div class="name">{html.escape(Path(entry["path"]).name)}'
        f"</div>{rows}</div></div>"
    )


def build_sheet(folder: Path, region: str | None) -> None:
    cache = Cache(folder / ".fovea" / "cache.sqlite")
    config = PipelineConfig(
        detect=True, eye=True, species=True, species_region=region, export=False
    )
    entries = run_pipeline(folder, config, cache)
    bursts = group_bursts(entries, config.gap_seconds)
    cards = [c for b in bursts if (best := best_species_frame(b)) and (c := card(best))]
    out = folder / ".fovea" / f"species_sheet_{region or 'all'}.html"
    out.write_text(
        f"<!doctype html><style>{STYLE}</style><h3>{html.escape(str(folder))} · "
        f"{region or 'all regions'} · {len(cards)} bursts</h3>"
        f'<div class="grid">{"".join(cards)}</div>'
    )
    print(f"{len(cards)} bursts -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()
    for f in args.folders:
        build_sheet(Path(f).expanduser().resolve(), args.region)
