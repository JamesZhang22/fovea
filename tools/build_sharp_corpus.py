"""Assemble the tack-sharp eye patch corpus that anchors the focus score.

Usage: uv run python tools/build_sharp_corpus.py <label-folder>... [--out data/sharp-eyes]

Extracts fixed-size native-resolution patches around hand-labeled eyes (no resizing,
blur radius stays in native pixels), ranks them by sharpness, keeps the best per burst
plus the global top, and writes a contact sheet for human confirmation.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fovea.core.group.burst import group_bursts
from fovea.core.ingest import cr3
from fovea.core.ingest.cache import Cache
from fovea.core.ingest.decode import roi_native
from fovea.core.scan import scan_folder
from fovea.core.score.classical import metrics, to_gray

PATCH_PX = 96  # fixed native-resolution crop, the focus model's input size
MIN_BOX_PX = 400  # below this the eye is too small to define "sharp"
MIN_MEAN_LUMA = 35.0  # darker patches are noise traps for contrast-normalized metrics
MIN_GRAY_STD = 18.0  # flat patches carry no edges worth anchoring on


def collect(folders: list[Path]) -> list[dict]:
    """One record per labeled eye with its burst id and native-res patch metrics."""
    rows = []
    for fi, folder in enumerate(folders):
        labels = [
            json.loads(line) for line in (folder / "eye_labels.jsonl").read_text().splitlines()
        ]
        labels = [r for r in labels if r["eye"]]
        entries = scan_folder(folder, Cache(folder / ".fovea" / "cache.sqlite"))
        burst_of = {}
        for b_id, burst in enumerate(group_bursts(entries)):
            for e in burst:
                burst_of[e["path"]] = fi * 100_000 + b_id

        jpeg_path = None
        jpeg = None
        for r in labels:
            box_px = max(r["box"]["x1"] - r["box"]["x0"], r["box"]["y1"] - r["box"]["y0"])
            if box_px < MIN_BOX_PX:
                continue
            if r["path"] != jpeg_path:
                p = Path(r["path"])
                jpeg = cr3.read_range(p, cr3.read_previews(p).full)
                jpeg_path = r["path"]
            ex, ey = r["eye"]
            half = PATCH_PX // 2
            region = (int(ex - half), int(ey - half), int(ex + half), int(ey + half))
            patch = roi_native(jpeg, region)
            if patch.size != (PATCH_PX, PATCH_PX):
                continue
            gray = to_gray(patch)
            if gray.mean() < MIN_MEAN_LUMA or gray.std() < MIN_GRAY_STD:
                continue
            m = metrics(gray)
            rows.append(
                {
                    "id": r["id"].replace("#", "_"),
                    "path": r["path"],
                    "eye": r["eye"],
                    "box_px": box_px,
                    "burst": burst_of.get(r["path"], -1),
                    "brenner": round(m["brenner"], 1),
                    "edge_sharpness": round(m["edge_sharpness"], 4),
                }
            )
            print(f"\r{len(rows)} patches measured", end="", file=sys.stderr)
    print(file=sys.stderr)
    return rows


def select_sharp(rows: list[dict]) -> list[dict]:
    """Best per burst by combined metric rank, both metrics must agree it is sharp."""

    def add_combined(pool: list[dict]) -> None:
        for key in ("brenner", "edge_sharpness"):
            order = sorted(pool, key=lambda r: r[key])
            for i, r in enumerate(order):
                r[f"_{key}_pct"] = i / max(len(order) - 1, 1)
        for r in pool:
            r["combined"] = round(r["_brenner_pct"] * r["_edge_sharpness_pct"], 4)

    add_combined(rows)
    best_per_burst: dict[int, dict] = {}
    for r in rows:
        cur = best_per_burst.get(r["burst"])
        if cur is None or r["combined"] > cur["combined"]:
            best_per_burst[r["burst"]] = r
    # within-burst metric ranking is trustworthy, the global cutoff is not — human
    # curation via the sheet is the only cross-subject filter
    return sorted(best_per_burst.values(), key=lambda r: -r["combined"])


def write_corpus(selected: list[dict], out: Path) -> None:
    patches_dir = out / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    jpeg_path = None
    jpeg = None
    for r in sorted(selected, key=lambda r: r["path"]):
        if r["path"] != jpeg_path:
            p = Path(r["path"])
            jpeg = cr3.read_range(p, cr3.read_previews(p).full)
            jpeg_path = r["path"]
        ex, ey = r["eye"]
        half = PATCH_PX // 2
        patch = roi_native(jpeg, (int(ex - half), int(ey - half), int(ex + half), int(ey + half)))
        patch.save(patches_dir / f"{r['id']}.png")  # lossless, no second jpeg pass
    (out / "index.json").write_text(json.dumps(selected))


def write_sheet(selected: list[dict], out: Path) -> Path:
    rejected_path = out / "rejected.json"
    rejected = set(json.loads(rejected_path.read_text())) if rejected_path.exists() else set()
    cells = "\n".join(
        f'<figure data-id="{r["id"]}"{" class=rejected" if r["id"] in rejected else ""}>'
        f'<img src="patches/{r["id"]}.png">'
        f"<figcaption>score {r['combined']:.3f}</figcaption></figure>"
        for r in sorted(selected, key=lambda r: -r["combined"])
    )
    html = f"""<!doctype html><meta charset="utf-8"><title>curate sharp eye corpus</title>
<style>
body {{ background: #141414; color: #999; font: 13px system-ui; margin: 16px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fill, 200px); gap: 10px; }}
img {{ width: 192px; image-rendering: pixelated; display: block; }}
figure {{ margin: 0; cursor: pointer; border: 2px solid transparent; border-radius: 4px; }}
figure.rejected {{ border-color: #c33; }} figure.rejected img {{ opacity: 0.25; }}
figcaption {{ padding: 2px; }}
#bar {{ position: sticky; top: 0; background: #141414; padding: 8px 0; }}
button {{ background: #2b6cb0; color: #fff; border: 0; border-radius: 5px; padding: 6px 14px;
          font: inherit; cursor: pointer; }}
</style>
<div id="bar">
  <b>{len(selected)} burst champions — click every patch that is NOT critically sharp</b>
  <button onclick="copyRejects()">copy rejected ids</button> <span id="n"></span>
</div>
<main>{cells}</main>
<script>
const count = () => document.getElementById("n").textContent =
  document.querySelectorAll(".rejected").length + " rejected";
document.querySelectorAll("figure").forEach(f =>
  f.addEventListener("click", () => {{ f.classList.toggle("rejected"); count(); }}));
function copyRejects() {{
  const ids = [...document.querySelectorAll(".rejected")].map(f => f.dataset.id);
  navigator.clipboard.writeText(JSON.stringify(ids));
  document.getElementById("n").textContent = ids.length + " copied";
}}
count();
</script>"""
    sheet = out / "sheet.html"
    sheet.write_text(html)
    return sheet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", default="data/sharp-eyes")
    args = parser.parse_args()
    out = Path(args.out)

    rows = collect([Path(f).expanduser().resolve() for f in args.folders])
    selected = select_sharp(rows)
    write_corpus(selected, out)
    sheet = write_sheet(selected, out)
    print(f"{len(rows)} candidates -> {len(selected)} selected -> {out}")
    print(f"review: open {sheet}")
