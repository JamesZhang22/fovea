import html
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import ImageDraw

from fovea.core.ingest import cr3, decode

THUMB_WIDTH_PX = 800

# overlay colors
IN_FOCUS = (255, 40, 40)  # red, AF points the camera reports as in focus
REPORTED = (255, 220, 0)  # yellow, AF points reported but not in focus
BIRD = (60, 220, 60)  # green, detected bird bounding boxes

PAGE = """<!doctype html>
<meta charset="utf-8"><title>fovea contact sheet</title>
<style>
body {{ background: #111; color: #ccc; font: 13px system-ui; margin: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 12px; }}
figure {{ margin: 0; }}
img {{ width: 100%; display: block; border-radius: 4px; }}
figcaption {{ padding: 4px 2px; color: #999; }}
</style>
<h1>fovea contact sheet — {n} files</h1>
<div class="grid">
{cells}
</div>
"""


def render_file(entry: dict, out_dir: Path) -> str:
    """Write one thumbnail with AF boxes drawn, returns the image filename or '' on failure."""
    path = Path(entry["path"])
    previews = cr3.read_previews(path)
    src = previews.full or previews.prvw
    if src is None:
        return ""
    im = decode.thumbnail(cr3.read_range(path, src), THUMB_WIDTH_PX)

    img_w = entry["meta"].get("ImageWidth") or im.width
    scale = im.width / max(img_w, 1)
    draw = ImageDraw.Draw(im)

    for b in entry.get("birds") or []:
        box = [b["x0"] * scale, b["y0"] * scale, b["x1"] * scale, b["y1"] * scale]
        draw.rectangle(box, outline=BIRD, width=3)

    af = entry["af"]
    if af and af["display_points"]:
        for p in af["display_points"]:
            x0 = (p["cx"] - p["w"] / 2) * scale
            y0 = (p["cy"] - p["h"] / 2) * scale
            x1 = (p["cx"] + p["w"] / 2) * scale
            y1 = (p["cy"] + p["h"] / 2) * scale
            color = IN_FOCUS if p["in_focus"] else REPORTED
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    name = path.stem + ".jpg"
    im.save(out_dir / "images" / name, quality=85)
    return name


def write_contactsheet(entries: list[dict], out_dir: Path) -> Path:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        names = list(pool.map(lambda e: render_file(e, out_dir), entries))

    cells = []
    for entry, name in zip(entries, names, strict=True):
        if not name:
            continue
        af = entry["af"]
        caption = Path(entry["path"]).name
        if af:
            mode = entry["meta"].get("AFAreaMode")
            tag = "lattice" if af["lattice"] else f"{af['n_points']} pt"
            caption += f" — mode {mode}, {tag}, {len(af['display_points'])} shown"
        cells.append(
            f'<figure><img loading="lazy" src="images/{name}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )

    out = out_dir / "index.html"
    out.write_text(PAGE.format(n=len(cells), cells="\n".join(cells)))
    return out
