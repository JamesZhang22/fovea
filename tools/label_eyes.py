"""Click-the-eye labeling tool.

Usage: uv run python tools/label_eyes.py <folder> [--out <folder>/eye_labels.jsonl] [--port 7333]

Serves one bird crop at a time. Click the eye center (loupe follows the cursor), the label
is a single point, `s` skips, arrows navigate, clicking a labeled frame relabels it.
Labels append to JSONL with eye coordinates in full-resolution image pixels, rerunning
resumes at the first unlabeled crop.
"""

import argparse
import io
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fovea.core.ingest import cr3, decode
from fovea.core.ingest.cache import Cache
from fovea.core.pipeline import PipelineConfig, run_pipeline

BOX_PAD_FRACTION = 0.15  # context around the bird box so head poses near the edge stay visible.
DISPLAY_WIDTH_PX = 1600
HOST = "127.0.0.1"  # localhost only, never exposed to the network.

PAGE = """<!doctype html>
<meta charset="utf-8"><title>fovea eye labeler</title>
<style>
body { background: #111; color: #ccc; font: 14px system-ui; margin: 0; text-align: center; }
#bar { padding: 8px; color: #999; }
#stage { position: relative; display: inline-block; cursor: crosshair; }
#img { max-width: 96vw; max-height: 84vh; display: block; }
#cursor { position: absolute; width: 22px; height: 22px; margin: -11px; border: 2px solid #f33;
          border-radius: 50%; pointer-events: none; display: none; }
#mark { position: absolute; width: 22px; height: 22px; margin: -11px; border: 2px solid #f33;
        border-radius: 50%; pointer-events: none; display: none; }
#mark::after { content: ""; position: absolute; left: 50%; top: 50%; width: 2px; height: 2px;
               margin: -1px; background: #f33; }
#hint { position: absolute; width: 22px; height: 22px; margin: -11px; border: 2px dashed #fa0;
        border-radius: 50%; pointer-events: none; display: none; }
kbd { background: #333; padding: 1px 5px; border-radius: 3px; }
</style>
<div id="bar"></div>
<div id="stage"><img id="img"><div id="cursor"></div><div id="mark"></div><div id="hint"></div>
</div>
<div><kbd>click</kbd> label eye &nbsp; <kbd>enter</kbd> accept suggestion &nbsp; <kbd>s</kbd> skip
&nbsp; <kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate &nbsp; <kbd>u</kbd> undo last</div>
<script>
let cur = null;
const img = document.getElementById("img"), cursor = document.getElementById("cursor"),
      mark = document.getElementById("mark"), hint = document.getElementById("hint"),
      bar = document.getElementById("bar");

function place(el, pt) {
  if (!cur || !pt) { el.style.display = "none"; return; }
  const r = img.getBoundingClientRect();
  el.style.display = "block";
  el.style.left = pt.x * r.width + "px";
  el.style.top = pt.y * r.height + "px";
}

function placeMark() {
  place(mark, cur && cur.label);
  place(hint, cur && cur.suggest);
}

async function load(i) {
  const r = await fetch("/item?i=" + i);
  cur = await r.json();
  let state = cur.skipped ? "SKIPPED" : cur.label ? "labeled" : "unlabeled";
  bar.textContent = `${cur.progress}  ·  ${cur.id}  ·  conf ${cur.confidence}  ·  ` +
    `${cur.i + 1}/${cur.n}  ·  ${state}`;
  img.onload = placeMark;
  img.src = cur.img;
}

img.addEventListener("mousemove", e => {
  const r = img.getBoundingClientRect();
  cursor.style.display = "block";
  cursor.style.left = e.clientX - r.left + "px";
  cursor.style.top = e.clientY - r.top + "px";
});
img.addEventListener("mouseleave", () => cursor.style.display = "none");

img.addEventListener("click", async e => {
  const r = img.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  const resp = await fetch("/label", {method: "POST",
    body: JSON.stringify({id: cur.id, x, y})});
  load((await resp.json()).next);
});

document.addEventListener("keydown", async e => {
  if (!cur) return;
  if (e.key === "Enter" && cur.suggest) {
    const resp = await fetch("/label", {method: "POST",
      body: JSON.stringify({id: cur.id, accept: true})});
    load((await resp.json()).next);
  } else if (e.key === "s") {
    const resp = await fetch("/label", {method: "POST",
      body: JSON.stringify({id: cur.id, skip: true})});
    load((await resp.json()).next);
  } else if (e.key === "ArrowLeft" && cur.i > 0) {
    load(cur.i - 1);
  } else if (e.key === "ArrowRight" && cur.i < cur.n - 1) {
    load(cur.i + 1);
  } else if (e.key === "u") {
    const resp = await fetch("/undo", {method: "POST"});
    load((await resp.json()).next);
  }
});
fetch("/start").then(r => r.json()).then(d => load(d.i));
</script>
"""


class Labeler:
    def __init__(self, folder: Path, out: Path, prelabels: Path | None = None) -> None:
        self.out = out
        entries = run_pipeline(
            folder,
            PipelineConfig(score=False, export=False, detect=True),
            Cache(folder / ".fovea" / "cache.sqlite"),
        )
        self.items = []
        for e in entries:
            for i, box in enumerate(e.get("birds") or []):
                item_id = f"{Path(e['path']).name}#{i}"
                self.items.append({"id": item_id, "path": e["path"], "box": box})
        self.labels: dict[str, dict] = {}
        if out.exists():
            for line in out.read_text().splitlines():
                row = json.loads(line)
                self.labels[row["id"]] = row
        self.suggestions: dict[str, dict] = {}
        if prelabels and prelabels.exists():
            for line in prelabels.read_text().splitlines():
                row = json.loads(line)
                self.suggestions[row["id"]] = row
        self.lock = threading.Lock()

    def first_unlabeled(self) -> int:
        return next((i for i, it in enumerate(self.items) if it["id"] not in self.labels), 0)

    def next_unlabeled_after(self, i: int) -> int:
        """First unlabeled index after i, wrapping, stays at i when everything is labeled."""
        n = len(self.items)
        for step in range(1, n + 1):
            j = (i + step) % n
            if self.items[j]["id"] not in self.labels:
                return j
        return i

    def progress(self) -> str:
        return f"{len(self.labels)}/{len(self.items)} labeled"

    def crop_region(self, item: dict) -> tuple[int, int, int, int]:
        """Padded bird box in full-resolution pixel coordinates."""
        b = item["box"]
        pw = (b["x1"] - b["x0"]) * BOX_PAD_FRACTION
        ph = (b["y1"] - b["y0"]) * BOX_PAD_FRACTION
        return (int(b["x0"] - pw), int(b["y0"] - ph), int(b["x1"] + pw), int(b["y1"] + ph))

    def crop_jpeg(self, item: dict) -> bytes:
        previews = cr3.read_previews(Path(item["path"]))
        jpeg = cr3.read_range(Path(item["path"]), previews.full)
        im = decode.roi_native(jpeg, self.crop_region(item))
        if im.width > DISPLAY_WIDTH_PX:
            im = im.resize((DISPLAY_WIDTH_PX, round(im.height * DISPLAY_WIDTH_PX / im.width)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        return buf.getvalue()

    def _to_crop_fraction(self, item: dict, eye: list[float]) -> dict:
        x0, y0, x1, y1 = self.crop_region(item)
        return {"x": (eye[0] - x0) / (x1 - x0), "y": (eye[1] - y0) / (y1 - y0)}

    def item_state(self, i: int) -> dict:
        """Item payload for the UI, label and suggestion as crop-fraction coordinates."""
        item = self.items[i]
        row = self.labels.get(item["id"])
        label = None
        skipped = False
        if row:
            if row["eye"] is None:
                skipped = True
            else:
                label = self._to_crop_fraction(item, row["eye"])
        suggest = None
        srow = self.suggestions.get(item["id"])
        if srow and not row:
            suggest = self._to_crop_fraction(item, srow["eye"])
            suggest["conf"] = srow.get("confidence")
        return {
            "i": i,
            "n": len(self.items),
            "id": item["id"],
            "confidence": round(item["box"]["confidence"], 2),
            "img": f"/img/{i}",
            "label": label,
            "skipped": skipped,
            "suggest": suggest,
            "progress": self.progress(),
        }

    def record(self, item: dict, eye_fullres: tuple[float, float] | None) -> None:
        """Append or replace the label row for this item and rewrite the JSONL."""
        row = {
            "id": item["id"],
            "path": item["path"],
            "box": item["box"],
            "eye": list(eye_fullres) if eye_fullres else None,
        }
        with self.lock:
            self.labels[item["id"]] = row
            self._flush()

    def undo(self) -> int:
        """Remove the most recent label, returns its index for the UI to show."""
        with self.lock:
            if not self.labels:
                return self.first_unlabeled()
            last_id = next(reversed(self.labels))
            del self.labels[last_id]
            self._flush()
        return next(i for i, it in enumerate(self.items) if it["id"] == last_id)

    def _flush(self) -> None:
        self.out.write_text("".join(json.dumps(r) + "\n" for r in self.labels.values()))


def make_handler(labeler: Labeler):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _json(self, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE.encode())
            elif url.path == "/start":
                self._json({"i": labeler.first_unlabeled()})
            elif url.path == "/item":
                i = int(parse_qs(url.query)["i"][0])
                self._json(labeler.item_state(i))
            elif url.path.startswith("/img/"):
                jpeg = labeler.crop_jpeg(labeler.items[int(url.path.removeprefix("/img/"))])
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(jpeg)
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)) or b"{}")
            if self.path == "/label":
                i = next(i for i, it in enumerate(labeler.items) if it["id"] == body["id"])
                item = labeler.items[i]
                if body.get("skip"):
                    labeler.record(item, None)
                elif body.get("accept"):
                    labeler.record(item, tuple(labeler.suggestions[item["id"]]["eye"]))
                else:
                    x0, y0, x1, y1 = labeler.crop_region(item)
                    labeler.record(item, (x0 + body["x"] * (x1 - x0), y0 + body["y"] * (y1 - y0)))
                self._json({"next": labeler.next_unlabeled_after(i)})
            elif self.path == "/undo":
                self._json({"next": labeler.undo()})
            else:
                self.send_error(404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="label bird eye keypoints in the browser")
    parser.add_argument("folder")
    parser.add_argument("--out", help="labels JSONL path (default <folder>/eye_labels.jsonl)")
    parser.add_argument(
        "--prelabels", help="suggestions JSONL (default <folder>/eye_prelabels.jsonl)"
    )
    parser.add_argument("--port", type=int, default=7333)
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    out = Path(args.out) if args.out else folder / "eye_labels.jsonl"
    prelabels = Path(args.prelabels) if args.prelabels else folder / "eye_prelabels.jsonl"
    labeler = Labeler(folder, out, prelabels)
    if not labeler.items:
        sys.exit("no bird detections in this folder")

    url = f"http://{HOST}:{args.port}"
    print(f"{len(labeler.items)} crops, {labeler.progress()} -> {url}")
    webbrowser.open(url)
    server = ThreadingHTTPServer((HOST, args.port), make_handler(labeler))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped, {labeler.progress()}, labels in {out}")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
